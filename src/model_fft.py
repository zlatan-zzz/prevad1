from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.fft  # Added for TF-VAD Frequency operations
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj


# =========================================================================
# 新增模块：TF-VAD 核心频域组件
# Spectral-Convolutional Dynamics Attention (SCDA)
# =========================================================================
class SpectralConvAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super(SpectralConvAttention, self).__init__()
        self.d_model = d_model

        # 谱卷积层: 在频域(F轴)上进行1D卷积，捕捉局部频率模式
        # 输入维度是 2*d_model (实部+虚部)
        self.spec_conv = nn.Sequential(
            nn.Conv1d(in_channels=2 * d_model, out_channels=2 * d_model,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(2 * d_model),
            nn.ReLU()
        )

        # 跨频带注意力: 学习全局频率依赖
        self.attn = nn.MultiheadAttention(embed_dim=2 * d_model,
                                          num_heads=n_heads,
                                          dropout=dropout,
                                          batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(2 * d_model)

    def forward(self, x):
        """
        x: [Batch, Time, Dim]
        """
        B, T, D = x.shape

        # 1. RFFT (时域 -> 频域)
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")  # [B, F, D]

        # 2. 复数转实数拼接
        x_spec = torch.cat([x_fft.real, x_fft.imag], dim=-1)  # [B, F, 2D]

        # 3. 谱卷积 (需要 permute 适配 Conv1d)
        x_spec_conv = x_spec.permute(0, 2, 1)  # [B, 2D, F]
        x_spec_conv = self.spec_conv(x_spec_conv)
        x_spec_conv = x_spec_conv.permute(0, 2, 1)  # [B, F, 2D]

        # 4. 跨频带注意力
        attn_out, _ = self.attn(x_spec_conv, x_spec_conv, x_spec_conv)
        x_spec_refined = self.norm(x_spec_conv + self.dropout(attn_out))

        # 5. iRFFT (频域 -> 时域)
        real, imag = torch.split(x_spec_refined, [D, D], dim=-1)
        x_fft_refined = torch.complex(real, imag)
        v_freq = torch.fft.irfft(x_fft_refined, n=T, dim=1, norm="ortho")

        return v_freq


# =========================================================================
# 原有基础模块 (保持不变)
# =========================================================================

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# =========================================================================
# 主模型 CLIPVAD (增量修改)
# =========================================================================

class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        # --- Original Temporal Module ---
        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        # --- Original GCN Modules ---
        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()

        # [MODIFIED] 原来的 linear 用于 GCN 后的映射，现在我们可能有新的融合层
        self.linear_gcn = nn.Linear(visual_width, visual_width)  # Renamed for clarity, logic kept
        self.gelu = QuickGELU()

        # --- [NEW] TF-VAD Frequency Module (SCDA) ---
        # 插入频域增强模块
        self.scda_module = SpectralConvAttention(visual_width)

        # --- [NEW] Fusion Layer ---
        # 用于融合 GCN 特征 (Spatial-Temporal) 和 SCDA 特征 (Frequency)
        # 输入维度翻倍，输出回 visual_width
        self.tf_fusion = nn.Linear(visual_width * 2, visual_width)
        self.fusion_norm = nn.LayerNorm(visual_width)

        # --- [NEW] Normal Anchor (语义原点) ---
        # 初始化一个可学习的参数，代表正常特征的中心
        self.normal_anchor = nn.Parameter(torch.zeros(1, visual_width))
        nn.init.xavier_normal_(self.normal_anchor)

        # --- Original Classifier Logic ---
        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)
        # linear_gcn 是原来的 self.linear，需要初始化
        nn.init.xavier_normal_(self.linear_gcn.weight)
        nn.init.xavier_normal_(self.tf_fusion.weight)

    def build_attention_mask(self, attn_window):
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0
        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))  # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2
        return output

    def encode_video(self, images, padding_mask, lengths):
        # --- 1. Preprocessing & Temporal Transformer (Unchanged) ---
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        # x: [T, B, D] coming out of Transformer
        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)  # [B, T, D]

        # --- 2. GCN Branch (Original Logic) ---
        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])

        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x_gcn_cat = torch.cat((x1, x2), 2)
        x_gcn = self.linear_gcn(x_gcn_cat)  # [B, T, D]

        # --- 3. [NEW] Frequency Branch (TF-VAD Logic) ---
        # 我们使用 Transformer 后的特征 x 进入频域模块
        # 这增加了频域动力学信息
        x_freq = self.scda_module(x)  # [B, T, D]

        # --- 4. [NEW] Fusion ---
        # 将 GCN 提取的空间关系与 SCDA 提取的频率动力学融合
        x_combined = torch.cat([x_gcn, x_freq], dim=-1)  # [B, T, 2D]
        x_final = self.tf_fusion(x_combined)  # [B, T, D]
        x_final = self.fusion_norm(x_final)

        return x_final

    def encode_textprompt(self, text):
        # (Unchanged)
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat(
            [len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)
        return text_features

    def forward(self, visual, padding_mask, text, lengths):
        # 1. 提取视频特征 (现在包含了 时域+GCN+频域 的融合信息)
        visual_features = self.encode_video(visual, padding_mask, lengths)

        # 2. 原有的 MIL 分类器逻辑 (logits1)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        # 3. 提取文本特征
        text_features_ori = self.encode_textprompt(text)

        # 4. 视觉-文本 交叉注意力 (Original Logic)
        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)

        # 原有的相似度 Logits (logits2)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        # --- [NEW] TF-VAD Specific Scoring ---
        # 计算特征到 "Normal Anchor" 的欧氏距离
        # dist: [B, T]
        dist_to_anchor = torch.norm(visual_features - self.normal_anchor, p=2, dim=-1)

        # 返回值增加了 dist_to_anchor，用于计算 TF-VAD 的 Loss
        # 依然返回 logits1, logits2 以兼容原来的训练逻辑
        return text_features_ori, logits1, logits2, dist_to_anchor