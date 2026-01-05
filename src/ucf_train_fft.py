import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
# 跳过ssl
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# 假设 model.py 中已经包含了修改后的 CLIPVAD 类 (带 SCDA 和 dist_to_anchor 输出)
from model_fft import CLIPVAD
from ucf_test_fft import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option


# =========================================================================
# [新增] TF-VAD 核心损失函数模块
# 替代原有的 CLASM 和 CLAS2，实现 Ranking + Semantic + Center Loss
# =========================================================================
class TFVADLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, margin=100.0):
        super(TFVADLoss, self).__init__()
        self.alpha = alpha  # Ranking Loss 权重
        self.beta = beta  # Semantic Class Loss 权重
        self.gamma = gamma  # Center Loss 权重 (核心创新)
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits1, logits2, dist_to_anchor, text_labels, lengths):
        """
        logits1: [B, T, 1] - 异常评分
        logits2: [B, T, K] - 语义相似度
        dist_to_anchor: [B, T] - 到 Normal Anchor 的距离
        text_labels: [B, ClassNum] - One-hot 标签 (index 0 is Normal)
        """
        # --- 1. 数据预处理 ---
        # 这里的 text_labels 是 One-hot 的。
        # Normal 样本: text_labels[:, 0] == 1
        # Anomaly 样本: text_labels[:, 0] == 0
        is_normal = (text_labels[:, 0] == 1)
        is_anomaly = (text_labels[:, 0] == 0)

        # 生成 Mask 处理 Padding (长度之外的设为忽略)
        batch_size, max_len = logits1.shape[:2]
        mask = torch.arange(max_len, device=logits1.device).expand(batch_size, max_len) < lengths.unsqueeze(1)

        # 将 Padding 区域的 Logits 设为极小值，以免干扰 Max 计算
        logits1_masked = logits1.squeeze(-1).clone()
        logits1_masked[~mask] = -1e9

        # --- 2. MIL Ranking Loss (对应原代码的 loss1/CLAS2) ---
        # 逻辑: 异常视频最高分 > 正常视频最高分 + Margin
        top_scores, _ = torch.max(logits1_masked, dim=1)  # [B]

        normal_scores = top_scores[is_normal]
        anomaly_scores = top_scores[is_anomaly]

        loss_rank = torch.tensor(0.0, device=logits1.device)
        if len(normal_scores) > 0 and len(anomaly_scores) > 0:
            # Hinge Loss: max(0, margin + max_norm - max_ano)
            # 我们希望 max_ano 越大越好，max_norm 越小越好
            loss_rank = torch.relu(self.margin + normal_scores.mean() - anomaly_scores.mean())

        # --- 3. Semantic Classification Loss (对应原代码的 loss2/CLASM) ---
        # 逻辑: 异常视频中最显著的那一帧，应该被分类为正确的异常类别
        loss_cls = torch.tensor(0.0, device=logits1.device)
        if is_anomaly.sum() > 0:
            # 获取异常样本的真实类别索引 (0-13, 0是Normal, 所以要 -1 得到 0-12 的异常类索引)
            # text_labels[is_anomaly] shape: [N_ano, 14] -> argmax -> indices 1~13
            ano_labels_raw = torch.argmax(text_labels[is_anomaly], dim=1)
            ano_cls_targets = ano_labels_raw - 1  # Shift to 0-based for CE Loss

            # 找到异常得分最高的帧索引
            _, max_indices = torch.max(logits1_masked[is_anomaly], dim=1)  # [N_ano]

            # 取出这些帧对应的语义预测 logits2
            # logits2 shape: [B, T, K_anomaly]
            # 我们需要 gather [N_ano, K_anomaly]
            # 注意: logits2 只有异常类别的维度 (通常 13 类)

            # 这是一个 tricky 的点: 需要确认 logits2 的最后一维大小。
            # 原代码中 logits2 是 visual @ text。text 包括 Normal。
            # 如果 logits2 包含 Normal 类，我们需要排除它，或者根据 model 输出调整。
            # 假设 model 输出的 logits2 对应所有 prompt (14类)。

            logits2_ano_samples = logits2[is_anomaly]  # [N_ano, T, 14]
            # 选取 Max Frame
            pred_cls_frames = logits2_ano_samples[torch.arange(logits2_ano_samples.size(0)), max_indices]

            # 去掉第0列 (Normal)，只看异常类的分布，或者直接对全类做 CE
            # 这里为了对齐语义，我们让它预测具体的异常类 (1-13)
            # 既然 target 是 1-13，我们切片 logits 取 [:, 1:]
            pred_cls_abnormal = pred_cls_frames[:, 1:]

            if pred_cls_abnormal.shape[1] == ano_cls_targets.max() + 1:
                loss_cls = self.ce_loss(pred_cls_abnormal, ano_cls_targets)

        # --- 4. Hypersphere Center Loss (TF-VAD 核心创新) ---
        # 逻辑: 正常视频的所有有效帧，距离 Normal Anchor 应该尽可能近
        loss_center = torch.tensor(0.0, device=logits1.device)
        if is_normal.sum() > 0:
            normal_dists = dist_to_anchor[is_normal]  # [N_norm, T]
            normal_masks = mask[is_normal]  # [N_norm, T]

            # 只计算非 Padding 区域
            valid_dists = normal_dists[normal_masks]
            if valid_dists.numel() > 0:
                loss_center = torch.mean(valid_dists ** 2)

        # 总损失
        total_loss = self.alpha * loss_rank + self.beta * loss_cls + self.gamma * loss_center
        return total_loss, loss_rank, loss_cls, loss_center


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)

    # [新增] 初始化 TF-VAD 损失函数
    # 参数可根据实验调整，这里给出一组推荐值
    criterion = TFVADLoss(alpha=1.0, beta=1.0, gamma=0.001, margin=100.0).to(device)

    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch + 1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        # 记录器重置
        loss_meter = {'total': 0, 'rank': 0, 'cls': 0, 'center': 0}

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            # 1. 数据加载与拼接
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            # 拼接长度
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)

            # 处理标签: text_labels 是 one-hot 形式
            raw_labels = list(normal_label) + list(anomaly_label)
            text_labels = get_batch_label(raw_labels, prompt_text, label_map).to(device)

            # 2. 模型前向传播 (注意接收第4个返回值 dist_to_anchor)
            # text_features, logits1, logits2, dist_to_anchor
            text_features, logits1, logits2, dist_to_anchor = model(visual_features, None, prompt_text, feat_lengths)

            # 3. [修改] 使用 TF-VAD Loss 计算损失
            loss, l_rank, l_cls, l_center = criterion(logits1, logits2, dist_to_anchor, text_labels, feat_lengths)

            # 记录损失以便打印
            loss_meter['total'] += loss.item()
            loss_meter['rank'] += l_rank.item()
            loss_meter['cls'] += l_cls.item()
            loss_meter['center'] += l_center.item()

            # 4. 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += i * normal_loader.batch_size * 2

            # 5. 打印日志与测试
            if step % 1280 == 0 and step != 0:
                print(f"epoch: {e + 1} | step: {step} | "
                      f"Total: {loss_meter['total'] / (i + 1):.4f} | "
                      f"Rank: {loss_meter['rank'] / (i + 1):.4f} | "
                      f"Cls: {loss_meter['cls'] / (i + 1):.4f} | "
                      f"Center: {loss_meter['center'] / (i + 1):.4f}")

                # 测试代码保持不变 (注意: test 函数内部可能需要适配新的模型返回值，但如果 test 只用 logits1，通常兼容)
                # 如果 ucf_test.py 里的 test 函数里调用 model 没解包第4个参数，可能会报错
                # 建议在 ucf_test.py 中也做类似修改: _, logits, _, _ = model(...)
                AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)

                if AUC > ap_best:  # 这里通常用 AUC 或 AP 作为保存标准
                    ap_best = AUC
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)

        scheduler.step()

        # 保存当前 Epoch 模型
        torch.save(model.state_dict(), 'model1/model_fft_cur.pth')
        # 重载最优模型以保证训练稳定性 (可选策略)
        # checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        # model.load_state_dict(checkpoint['model_state_dict'])

    # 训练结束保存最终模型
    if args.use_checkpoint:
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        torch.save(checkpoint['model_state_dict'], args.model_path)
    else:
        torch.save(model.state_dict(), args.model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault',
                      'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting',
                      'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting',
                      'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 确保这里的 CLIPVAD 是修改后的包含 TF-VAD 模块的版本
    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head,
                    args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
