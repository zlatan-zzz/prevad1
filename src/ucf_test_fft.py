import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# 确保导入的是修改后的模型
from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    model.to(device)
    model.eval()

    element_logits2_stack = []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = item[2]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            # =============================================================
            # [修改点 1]: 正确解包模型返回的 4 个参数
            # text_features, logits1, logits2, dist_to_anchor
            # =============================================================
            _, logits1, logits2, dist_to_anchor = model(visual, padding_mask, prompt_text, lengths)

            # Reshape 处理
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            dist_to_anchor = dist_to_anchor.reshape(dist_to_anchor.shape[0] * dist_to_anchor.shape[1])

            # 计算分数
            # prob2: 基于文本相似度的异常概率 (1 - Normal的概率)
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))

            # prob1: 基于 MIL 分类器的异常概率
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            # [可选优化]: 如果你想测试 "距离原点越远越异常" 的效果，可以解除下面的注释
            # prob_dist = (dist_to_anchor[0:len_cur] - dist_to_anchor[0:len_cur].min()) / (dist_to_anchor[0:len_cur].max() - dist_to_anchor[0:len_cur].min() + 1e-6)
            # prob1 = (prob1 + prob_dist) / 2  # 融合距离分数

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    # 计算 AUC 和 AP 指标
    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP / (i + 1)
    print('average MAP: {:.2f}'.format(averageMAP))

    # 返回 AUC1 作为主要的监控指标 (对应 ucf_train_fft.py 中的 AP = AUC)
    return ROC1, AP1


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()

    label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault',
                      'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting',
                      'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting',
                      'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})

    testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # 实例化模型
    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head,
                    args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    # 加载权重
    # 注意：如果加载的是旧的权重文件（没有TF-VAD模块参数），这里可能会报错 key mismatch。
    # 只有加载 ucf_train_fft.py 训练出的新权重才完全匹配。
    # 如果是测试旧权重，需要在 model.py 里加 strict=False，但那样就没有频域效果了。
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param)

    test(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)