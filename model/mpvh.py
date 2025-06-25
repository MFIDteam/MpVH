import torch
import math
import timm
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from timm.models.layers import to_2tuple
from models.adapter import MambaAdapter


# 定义图像到 patch 的 Embedding 层
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        # 计算图像划分后的 patch grid size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 定义卷积操作来实现图像到 patch embedding 的映射
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 通过卷积得到图像的 patch embedding
        x = self.proj(x)  # 输出形状：(B, embed_dim, H', W')
        # 将最后两维展平，将形状变为 (B, embed_dim, num_patches)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC，其中 N=num_patches
        return x


class VisionTransformerRetrieval(nn.Module):
    def __init__(self, code_length=32, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., drop_rate=0.1, embed_layer=PatchEmbed):
        super().__init__()

        self.code_length = code_length
        self.embed_dim = embed_dim
        self.patch_embed = embed_layer(img_size=224, patch_size=16, embed_dim=embed_dim)

        # 定义一个可学习的加权参数，用于加权前一层adapter输出与ViT输出（原有逻辑保留）
        self.alpha = nn.Parameter(torch.tensor(0.35), requires_grad=True)

        # 使用适配器模块
        self.adapter = MambaAdapter(dim=embed_dim, d_state=16, ssm_ratio=1.0, dropout=0.1)

        self.adapter_weight = nn.Parameter(torch.zeros(embed_dim))  # 每个通道一个权重，初始化为1
        init.uniform_(self.adapter_weight, a=-0.1, b=0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 加载预训练的 ViT 模型
        pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.blocks = pretrained_vit.blocks  # 直接加载预训练的 Transformer blocks
        self.pos_embed = pretrained_vit.pos_embed  # 加载预训练的位置编码
        self.cls_token.data.copy_(pretrained_vit.cls_token)  # 加载预训练的分类 token

        self.norm = nn.LayerNorm(embed_dim)

        self.hash_layer_active = nn.Sequential(
            nn.Tanh(),
        )
        self.W_G = nn.Parameter(torch.Tensor(code_length, 256))
        torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))
        self.fc = nn.Linear(embed_dim, 256)
        self.bn = nn.BatchNorm1d(256)

        # 新增多头注意力参数
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.conv_q = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,  # 保持维度一致
            kernel_size=1,
            bias=False
        )
        self.conv_k = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            bias=False
        )
        self.conv_v = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            bias=False
        )
        # 总层数 L
        self.num_blocks = len(self.blocks)
        # 我们在每一层中将选择 num_routing = L//2+1 个路由专家
        self.num_routing = self.num_blocks // 2 + 1

    def forward_features(self, x):
        x = self.patch_embed(x)  # (B, N, C)
        B, N, C = x.shape

        # 添加分类 token 并应用位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + N, C)
        x = x + self.pos_embed  # 加入预训练的位置编码
        x = self.pos_drop(x)

        # 定义变量存储前一层adapter的输出以及保存所有前层的 Q（作为路由专家）
        prev_adapter_output = None
        prev_Q_list = []  # 每个元素形状 (B, N, C)

        for i, block in enumerate(self.blocks):
            # 计算adapter的输入：第一层直接使用ViT输出，否则融合上一层adapter输出与ViT输出
            if prev_adapter_output is None:
                adapter_input = x[:, 1:, :].clone()  # (B, N, C)
            else:
                adapter_input = self.alpha * prev_adapter_output + (1 - self.alpha) * x[:, 1:, :]

            # 获取当前层的adapter输出
            adapter_output = self.adapter(adapter_input)  # (B, N, C)
            prev_adapter_output = adapter_output.clone()

            # ----- 计算当前层 Q -----
            q_input = adapter_output.permute(0, 2, 1).unsqueeze(-1)  # (B, C, N, 1)
            q_current = self.conv_q(q_input)  # (B, C, N, 1)
            q_current = q_current.squeeze(-1).permute(0, 2, 1)   # (B, N, C)

            # 同时计算当前层的 K 用于路由选择（基于当前adapter输出）
            k_input_current = adapter_output.permute(0, 2, 1).unsqueeze(-1)  # (B, C, N, 1)
            current_K = self.conv_k(k_input_current)  # (B, C, N, 1)
            current_K = current_K.squeeze(-1).permute(0, 2, 1)  # (B, N, C)

            # ----- 路由专家选择 -----
            # 固定专家：当前层的 q_current
            # 路由专家候选：之前所有层保存的 Q，保存在 prev_Q_list 中
            # 对每个候选专家，我们计算其与当前 K 的平均点积（作为相似度评分），选择评分最高的 num_routing 个
            if len(prev_Q_list) > 0:
                candidate_scores = []
                for candidate in prev_Q_list:
                    # 计算每个候选专家与当前K的相似度，取均值得到标量评分
                    score = (candidate * current_K).mean()
                    candidate_scores.append(score)
                candidate_scores = torch.stack(candidate_scores)  # (num_candidates,)
                top_k = min(self.num_routing, len(prev_Q_list))
                top_values, top_indices = torch.topk(candidate_scores, top_k, largest=True)
                selected_experts = [prev_Q_list[idx] for idx in top_indices.tolist()]
                # 将固定专家与选中的路由专家进行简单平均
                fused_q = q_current
                for expert in selected_experts:
                    fused_q = fused_q + expert
                fused_q = fused_q / (1 + top_k)
            else:
                fused_q = q_current

            # 将当前层的 q_current 存入列表，以便后续作为路由专家候选
            prev_Q_list.append(q_current.clone())

            # ----- 用融合后的 Q 计算 K 和 V -----
            fused_Q_input = fused_q.permute(0, 2, 1).unsqueeze(-1)  # (B, C, N, 1)
            K = self.conv_k(fused_Q_input)  # (B, C, N, 1)
            V = self.conv_v(fused_Q_input)  # (B, C, N, 1)
            # 恢复形状并分割多头
            K = K.squeeze(-1).permute(0, 2, 1)
            K = K.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            V = V.squeeze(-1).permute(0, 2, 1)
            V = V.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            # 计算注意力
            Q = q_current.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # 固定专家计算的 Q
            attn = (Q @ K.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = attn.softmax(dim=-1)

            # 注意力作用到 V 上
            attended_v = (attn @ V).permute(0, 2, 1, 3)  # (B, N, h, d)
            attended_v = attended_v.reshape(B, N, C)  # 合并多头

            # 应用通道权重和归一化
            attended_v = self.norm(attended_v * self.adapter_weight.unsqueeze(0).unsqueeze(0))

            # 更新特征，不使用原地操作
            updated_tokens = x[:, 1:, :] + attended_v
            x = torch.cat((x[:, :1, :], updated_tokens), dim=1)

            # 通过 Transformer 块
            x = block(x)

        # 提取分类 token 的特征，并归一化
        cls_features = x[:, 0]  # (B, C)
        cls_features = self.norm(cls_features)

        return cls_features

    def forward(self, x):
        features = self.forward_features(x)  # 提取分类 token 特征
        features = self.fc(features)
        features = self.bn(features)

        deep_S_G = F.linear(features, self.W_G)
        ret = self.hash_layer_active(deep_S_G)
        return ret  # 返回图像嵌入用于检索


# 初始化模型
def vit_base_patch16_224_retrieval(code_length=32, embed_dim=768, depth=12,
                                   num_heads=12, mlp_ratio=4., drop_rate=0.1):
    model = VisionTransformerRetrieval(
        code_length=code_length,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate
    )
    return model
