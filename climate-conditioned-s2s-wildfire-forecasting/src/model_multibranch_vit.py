"""
Multi-Branch ViT: 基于 ViT-B/16 的三分支模型
支持 Local / Global / OCIs 三种输入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from typing import Optional


class PatchEmbedding(nn.Module):
    """Conv-based Patch Embedding"""
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, H'*W')
        x = x.transpose(1, 2)  # (B, H'*W', embed_dim)
        return x


class ImprovedTCN(nn.Module):
    """
    改进的时序卷积网络（TCN）

    特点：
    1. 因果卷积：避免未来信息泄露（只看过去和当前）
    2. 多层堆叠：增加感受野和建模能力
    3. 残差连接：保留原始信息
    4. 深度可分离卷积：减少参数量
    """
    def __init__(self, channels: int, kernel_size: int = 5, num_layers: int = 2, use_causal: bool = True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.use_causal = use_causal

        # 计算padding
        if use_causal:
            # 因果卷积：只在左侧padding，确保只看到过去的信息
            self.padding = (kernel_size - 1)
        else:
            # 普通卷积：两侧padding
            self.padding = (kernel_size - 1) // 2

        # 构建多层TCN
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                # 深度可分离1D卷积
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=self.padding if not use_causal else 0,  # 因果卷积手动padding
                    groups=channels,  # 深度可分离
                    bias=False
                ),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                # Pointwise卷积（1x1卷积）
                nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(channels),
            )
            self.tcn_layers.append(layer)

        # 残差连接的可学习门控
        self.gates = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])

    def forward(self, x):
        """
        Args:
            x: (B, C, T) - 时序特征

        Returns:
            out: (B, C, T) - TCN处理后的特征
        """
        out = x

        for i, layer in enumerate(self.tcn_layers):
            residual = out

            # 因果卷积：手动padding
            if self.use_causal:
                out = F.pad(out, (self.padding, 0))  # 只在左侧padding

            # TCN层
            out = layer(out)

            # 因果卷积：裁剪到原始长度
            if self.use_causal and out.size(2) > residual.size(2):
                out = out[:, :, :residual.size(2)]

            # 残差连接（带门控）
            out = residual + self.gates[i] * out

        return out


class TemporalTransformer(nn.Module):
    """
    时序Transformer：对每个空间位置的时序序列进行显式建模

    核心思想：在空间建模之前，先提取时序特征
    - 对每个patch位置，建模其在不同时间步的演化
    - 捕捉短期变化和长期趋势
    - 时序-空间解耦，避免时序信息被空间信息淹没

    输入: (B, T, N, C) - T个时间步，N个空间位置
    输出: (B, T, N, C) - 时序增强的特征
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,  # FFN维度
            dropout=0.0,
            activation='gelu',
            batch_first=False,  # (T, B*N, C)
            norm_first=True  # Pre-LN，更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 最终归一化
        self.norm = nn.LayerNorm(embed_dim)

        print(f"  ✓ 时序Transformer: {num_layers}层, {num_heads}头, 显式建模时序依赖")

    def forward(self, x):
        """
        Args:
            x: (B, T, N, C) - T个时间步，N个空间位置

        Returns:
            out: (B, T, N, C) - 时序增强的特征
        """
        B, T, N, C = x.shape

        # 重排维度: (B, T, N, C) -> (B*N, T, C) -> (T, B*N, C)
        # 将每个空间位置的时序序列作为独立样本
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)  # (B*N, T, C)
        x = x.permute(1, 0, 2)  # (T, B*N, C)

        # 时序Transformer：对每个空间位置的T个时间步建模
        x = self.transformer(x)  # (T, B*N, C)
        x = self.norm(x)

        # 重排回原始维度: (T, B*N, C) -> (B*N, T, C) -> (B, N, T, C) -> (B, T, N, C)
        x = x.permute(1, 0, 2)  # (B*N, T, C)
        x = x.reshape(B, N, T, C).permute(0, 2, 1, 3)  # (B, T, N, C)

        return x


class SpatialTokenMixer2D(nn.Module):
    """
    空间Token混合器：在patch-grid上进行局部交互

    目的：patch embedding后的10×10 token grid是完全不重叠的patch，
    局部连续性弱。使用轻量depthwise 3×3 conv增强空间结构感知。

    输入：(B*T, C, Hgrid, Wgrid) - local是10×10，global是3×6
    输出：(B*T, C, Hgrid, Wgrid)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gn = nn.GroupNorm(32, dim)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)

    def forward(self, x):  # (B, D, H, W)
        """
        Args:
            x: (B*T, dim, Hgrid, Wgrid)
        Returns:
            out: (B*T, dim, Hgrid, Wgrid)
        """
        y = self.pw(F.gelu(self.dw(self.gn(x))))
        return x + y


class TemporalTokenMixerTCN(nn.Module):
    """
    时序Token混合器：每个patch位置沿时间建模

    目的：显式建模每个空间位置随时间的变化模式。
    只在T=4上操作，不增加token数，显存开销小。

    输入：(B, T, N, D) - local N=100，global N=18
    输出：(B, T, N, D)
    """
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.pw = nn.Conv1d(dim, dim, 1)

    def forward(self, x):  # (B, T, N, D)
        """
        Args:
            x: (B, T, N, D)
        Returns:
            out: (B, T, N, D)
        """
        B, T, N, D = x.shape
        y = x.permute(0, 2, 3, 1).reshape(B*N, D, T)      # (B*N, D, T)
        y = self.pw(F.gelu(self.dw(y)))                   # (B*N, D, T)
        y = y.reshape(B, N, D, T).permute(0, 3, 1, 2)     # (B, T, N, D)
        return x + y


class TemporalAttentionPool(nn.Module):
    """
    改进的时序注意力池化模块（多头注意力）

    使用多头自注意力机制来聚合时序信息，相比简单的卷积注意力：
    1. 多头机制：不同的头可以关注不同的时间模式
    2. 自注意力：时间步之间可以相互交互
    3. 更强的时序建模能力
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        # 多头注意力层
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=False  # 输入格式: (T, B*H*W, C)
        )

        # 层归一化
        self.norm = nn.LayerNorm(channels)

        # 可学习的查询向量（用于池化）
        self.query = nn.Parameter(torch.randn(1, 1, channels))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) - 时序特征

        Returns:
            out: (B, C, H, W) - 加权聚合后的特征
        """
        B, C, T, H, W = x.shape

        # 重排维度: (B, C, T, H, W) -> (T, B*H*W, C)
        x = x.permute(2, 0, 3, 4, 1)  # (T, B, H, W, C)
        x = x.reshape(T, B * H * W, C)  # (T, B*H*W, C)

        # 扩展查询向量: (1, 1, C) -> (1, B*H*W, C)
        query = self.query.expand(1, B * H * W, C)  # (1, B*H*W, C)

        # 多头注意力：query关注所有时间步
        # attn_output: (1, B*H*W, C)
        attn_output, _ = self.multihead_attn(
            query=query,      # (1, B*H*W, C)
            key=x,            # (T, B*H*W, C)
            value=x,          # (T, B*H*W, C)
        )

        # 层归一化
        attn_output = self.norm(attn_output)  # (1, B*H*W, C)

        # 重排回原始空间维度: (1, B*H*W, C) -> (B, C, H, W)
        attn_output = attn_output.squeeze(0)  # (B*H*W, C)
        attn_output = attn_output.view(B, H, W, C)  # (B, H, W, C)
        attn_output = attn_output.permute(0, 3, 1, 2)  # (B, C, H, W)

        return attn_output


class TemporalAttentionPoolMHA(nn.Module):
    """
    基于Query的多头注意力时序池化

    对每个空间位置，用可学习query做MultiheadAttention聚合T个时间步。

    输入：(B, T, N, D)
    输出：(B, N, D)
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):  # (B, T, N, D)
        """
        Args:
            x: (B, T, N, D)
        Returns:
            out: (B, N, D)
        """
        B, T, N, D = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(B*N, T, D)  # (B*N, T, D)
        q = self.query.expand(B*N, 1, D)               # (B*N, 1, D)
        out, _ = self.mha(q, xt, xt)                   # (B*N, 1, D)
        out = self.ln(out)
        return out.squeeze(1).reshape(B, N, D)         # (B, N, D)


class TemporalLocalEncoder(nn.Module):
    """
    Local编码器：集成空间和时序Token混合器

    输入: (B, T, 14, 80, 80) - T个时间步
    输出: (B, N, 768) = (B, 100, 768) - 时序池化后的patch tokens

    改进策略：
    1. 批处理所有时间步的patch embedding (B*T, 14, 80, 80) -> (B*T, 768, 10, 10)
    2. 应用SpatialTokenMixer2D增强空间结构感知
    3. Reshape成(B, T, N, D)格式
    4. 应用TemporalTokenMixerTCN建模时序变化
    5. 应用TemporalAttentionPoolMHA池化时间维度 -> (B, N, D)
    """
    def __init__(self, in_channels: int = 14, temporal_steps: int = 4, embed_dim: int = 768,
                 patch_size: int = 8, use_spatial_mixer: bool = True, use_temporal_mixer: bool = True):
        super().__init__()
        self.temporal_steps = temporal_steps
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_spatial_mixer = use_spatial_mixer
        self.use_temporal_mixer = use_temporal_mixer

        # 2D Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 空间Token混合器
        if use_spatial_mixer:
            self.spatial_mixer = SpatialTokenMixer2D(dim=embed_dim)
            print(f"  ✓ Local编码器启用SpatialTokenMixer2D")

        # 时序Token混合器
        if use_temporal_mixer:
            self.temporal_mixer = TemporalTokenMixerTCN(dim=embed_dim, kernel_size=3)
            print(f"  ✓ Local编码器启用TemporalTokenMixerTCN")

        # 时序注意力池化
        self.temporal_pool = TemporalAttentionPoolMHA(dim=embed_dim, num_heads=4)
        print(f"  ✓ Local编码器启用TemporalAttentionPoolMHA")

    def forward(self, x):
        """
        Args:
            x: (B, T, 14, 80, 80) - T个时间步
        Returns:
            tokens: (B, 100, 768) - 时序池化后的patch tokens
        """
        B, T, C, H, W = x.shape

        # 步骤1: 批处理所有时间步的patch embedding
        x_reshaped = x.reshape(B * T, C, H, W)  # (B*T, 14, 80, 80)
        patches = self.patch_embed(x_reshaped)  # (B*T, 768, 10, 10)

        # 步骤2: 空间Token混合器（在10×10 grid上增强局部连续性）
        if self.use_spatial_mixer:
            patches = self.spatial_mixer(patches)  # (B*T, 768, 10, 10)

        # 步骤3: Reshape成(B, T, N, D)格式用于时序建模
        h_grid = H // self.patch_size
        w_grid = W // self.patch_size
        patches = patches.view(B, T, self.embed_dim, h_grid, w_grid)  # (B, T, 768, 10, 10)
        patches = patches.permute(0, 1, 3, 4, 2)  # (B, T, 10, 10, 768)
        patches = patches.reshape(B, T, h_grid * w_grid, self.embed_dim)  # (B, T, 100, 768)

        # 步骤4: 时序Token混合器（每个空间位置沿时间建模）
        if self.use_temporal_mixer:
            patches = self.temporal_mixer(patches)  # (B, T, 100, 768)

        # 步骤5: 时序注意力池化 (B, T, N, D) -> (B, N, D)
        tokens = self.temporal_pool(patches)  # (B, 100, 768)

        return tokens


class TemporalGlobalEncoder(nn.Module):
    """
    Global编码器：集成空间和时序Token混合器

    输入: (B, T, 14, 180, 360) - T个时间步
    输出: (B, N, 768) = (B, 18, 768) - 时序池化后的patch tokens

    改进策略：
    1. 批处理所有时间步的patch embedding (B*T, 14, 180, 360) -> (B*T, 768, 3, 6)
    2. 应用SpatialTokenMixer2D增强空间结构感知
    3. Reshape成(B, T, N, D)格式
    4. 应用TemporalTokenMixerTCN建模时序变化
    5. 应用TemporalAttentionPoolMHA池化时间维度 -> (B, N, D)
    """
    def __init__(self, in_channels: int = 14, temporal_steps: int = 4, embed_dim: int = 768,
                 patch_size: int = 60, use_spatial_mixer: bool = True, use_temporal_mixer: bool = True):
        super().__init__()
        self.temporal_steps = temporal_steps
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_spatial_mixer = use_spatial_mixer
        self.use_temporal_mixer = use_temporal_mixer

        # 2D Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 空间Token混合器
        if use_spatial_mixer:
            self.spatial_mixer = SpatialTokenMixer2D(dim=embed_dim)
            print(f"  ✓ Global编码器启用SpatialTokenMixer2D")

        # 时序Token混合器
        if use_temporal_mixer:
            self.temporal_mixer = TemporalTokenMixerTCN(dim=embed_dim, kernel_size=3)
            print(f"  ✓ Global编码器启用TemporalTokenMixerTCN")

        # 时序注意力池化
        self.temporal_pool = TemporalAttentionPoolMHA(dim=embed_dim, num_heads=4)
        print(f"  ✓ Global编码器启用TemporalAttentionPoolMHA")

    def forward(self, x):
        """
        Args:
            x: (B, T, 14, 180, 360) - T个时间步
        Returns:
            tokens: (B, 18, 768) - 时序池化后的patch tokens
        """
        B, T, C, H, W = x.shape

        # 步骤1: 批处理所有时间步的patch embedding
        x_reshaped = x.reshape(B * T, C, H, W)  # (B*T, 14, 180, 360)
        patches = self.patch_embed(x_reshaped)  # (B*T, 768, 3, 6)

        # 步骤2: 空间Token混合器（在3×6 grid上增强局部连续性）
        if self.use_spatial_mixer:
            patches = self.spatial_mixer(patches)  # (B*T, 768, 3, 6)

        # 步骤3: Reshape成(B, T, N, D)格式用于时序建模
        h_grid = H // self.patch_size
        w_grid = W // self.patch_size
        patches = patches.view(B, T, self.embed_dim, h_grid, w_grid)  # (B, T, 768, 3, 6)
        patches = patches.permute(0, 1, 3, 4, 2)  # (B, T, 3, 6, 768)
        patches = patches.reshape(B, T, h_grid * w_grid, self.embed_dim)  # (B, T, 18, 768)

        # 步骤4: 时序Token混合器（每个空间位置沿时间建模）
        if self.use_temporal_mixer:
            patches = self.temporal_mixer(patches)  # (B, T, 18, 768)

        # 步骤5: 时序注意力池化 (B, T, N, D) -> (B, N, D)
        tokens = self.temporal_pool(patches)  # (B, 18, 768)

        return tokens


class OCIEncoder(nn.Module):
    """
    新版OCI编码器：将10×10的OCI数据转换为100个patch tokens

    输入: (B, 10, 10) - 10个变量 × 10个时间步
    输出: (B, 100, 768) - 100个patch tokens

    策略：
    1. 将10×10的数据看作一个"特征图"
    2. 每个单元(i,j)代表第i个变量在第j个时间步的值
    3. 每个单元通过MLP投影到768维
    4. 添加变量维度和时间维度的位置编码
    """
    def __init__(self, n_oci_vars: int = 10, oci_window: int = 10, embed_dim: int = 768):
        super().__init__()
        self.n_oci_vars = n_oci_vars
        self.oci_window = oci_window
        self.embed_dim = embed_dim

        # 将每个标量值投影到embed_dim维度
        self.value_proj = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # 变量维度的位置编码（10个变量）
        self.var_pos_embed = nn.Parameter(
            torch.zeros(1, n_oci_vars, 1, embed_dim)
        )
        nn.init.trunc_normal_(self.var_pos_embed, std=0.02)

        # 时间维度的位置编码（10个时间步）
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, 1, oci_window, embed_dim)
        )
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

    def forward(self, x_oci):
        """
        Args:
            x_oci: (B, 10, 10) - 10个变量 × 10个时间步

        Returns:
            tokens: (B, 100, 768) - 100个patch tokens
        """
        B = x_oci.size(0)

        # 将每个标量值投影到embed_dim
        # (B, 10, 10) → (B, 10, 10, 1) → (B, 10, 10, 768)
        x = x_oci.unsqueeze(-1)  # (B, 10, 10, 1)
        x = self.value_proj(x)  # (B, 10, 10, 768)

        # 添加2D位置编码
        # 变量维度位置编码: (1, 10, 1, 768) 广播到 (B, 10, 10, 768)
        # 时间维度位置编码: (1, 1, 10, 768) 广播到 (B, 10, 10, 768)
        x = x + self.var_pos_embed + self.time_pos_embed

        # Reshape成token序列: (B, 10, 10, 768) → (B, 100, 768)
        tokens = x.reshape(B, self.n_oci_vars * self.oci_window, self.embed_dim)

        return tokens


class ContextFiLM(nn.Module):
    """
    Context FiLM (Feature-wise Linear Modulation)

    目的：让Global和OCI的宏观背景信息直接调制Local tokens
    使模型更容易利用全球尺度drivers去解释局部火灾概率

    输入：
        - local_tokens: (B, N_local, D) - Local tokens
        - global_tokens: (B, N_global, D) - Global tokens
        - oci_tokens: (B, N_oci, D) - OCI tokens
    输出：
        - modulated_local: (B, N_local, D) - 调制后的Local tokens
    """
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*dim, 2*dim),
            nn.GELU(),
            nn.Linear(2*dim, 2*dim),
        )
        print(f"  ✓ 启用ContextFiLM: Global/OCI → Local调制")

    def forward(self, local_tokens, global_tokens, oci_tokens):  # (B,Nl,D),(B,Ng,D),(B,No,D)
        """
        Args:
            local_tokens: (B, N_local, D)
            global_tokens: (B, N_global, D)
            oci_tokens: (B, N_oci, D)
        Returns:
            modulated_local: (B, N_local, D)
        """
        g = global_tokens.mean(dim=1)  # (B,D)
        o = oci_tokens.mean(dim=1)     # (B,D)
        ctx = torch.cat([g, o], dim=1) # (B,2D)
        gamma_beta = self.mlp(ctx)     # (B,2D)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # (B,D),(B,D)
        gamma = gamma.unsqueeze(1)
        beta  = beta.unsqueeze(1)
        return local_tokens * (1.0 + gamma) + beta


class PatchwiseSegHead(nn.Module):
    """
    Patch-wise分割头（TeleViT1.0风格）

    每个local token线性解码到自己patch的空间分辨率，再组装整图。
    这样可以自动适应不同的patch_size（8/16/20等）。

    输入: (B, N, D) - N个patch tokens
    输出: (B, C, H*p, W*p) - 完整分割图
    """
    def __init__(self, dim: int, num_classes: int, patch_size: int, grid_h: int, grid_w: int):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.proj = nn.Linear(dim, num_classes * patch_size * patch_size)

    def forward(self, local_tokens):  # (B, N, D)
        """
        Args:
            local_tokens: (B, N, D) - N个patch tokens
        Returns:
            logits: (B, C, H*p, W*p) - 例如 (B, 3, 80, 80)
        """
        B, N, D = local_tokens.shape
        C = self.num_classes
        p = self.patch_size
        H, W = self.grid_h, self.grid_w
        x = self.proj(local_tokens)  # (B, N, C*p*p)
        x = x.view(B, H, W, C, p, p)                 # (B,H,W,C,p,p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous() # (B,C,H,p,W,p)
        return x.view(B, C, H*p, W*p)                # (B,C,80,80)


class ProgressiveSegmentationHead(nn.Module):
    """
    渐进式上采样分割头

    特点：
    1. 在特征空间进行上采样，而不是在 logits 空间
    2. 每次上采样后都进行卷积提取特征
    3. 使用 3×3 卷积保持空间信息
    4. 10×10 → 20×20 → 40×40 → 80×80
    5. 支持多时间步长预测（multi-horizon）
    """
    def __init__(self, embed_dim: int = 768, num_classes: int = 3, num_lead_times: int = 1):
        super().__init__()
        self.num_lead_times = num_lead_times

        # 第一阶段：降维 (10×10)
        self.stage1 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 第二阶段：上采样到 20×20
        self.stage2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 第三阶段：上采样到 40×40
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 第四阶段：上采样到 80×80
        self.stage4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 最终分类层（为每个 lead time 创建独立的分类器）
        if num_lead_times > 1:
            self.classifiers = nn.ModuleList([
                nn.Conv2d(32, num_classes, kernel_size=1) for _ in range(num_lead_times)
            ])
        else:
            self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, 768, 10, 10) - ViT encoder 输出

        Returns:
            logits: (B, 3, 80, 80) if num_lead_times==1
                    (B, L, 3, 80, 80) if num_lead_times>1
        """
        # Stage 1: 10×10 → 10×10 (降维)
        x = self.stage1(x)  # (B, 256, 10, 10)

        # Stage 2: 10×10 → 20×20
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 256, 20, 20)
        x = self.stage2(x)  # (B, 128, 20, 20)

        # Stage 3: 20×20 → 40×40
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 128, 40, 40)
        x = self.stage3(x)  # (B, 64, 40, 40)

        # Stage 4: 40×40 → 80×80
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 64, 80, 80)
        x = self.stage4(x)  # (B, 32, 80, 80)

        # 分类（多时间步长）
        if self.num_lead_times > 1:
            logits_list = []
            for i in range(self.num_lead_times):
                logits_t = self.classifiers[i](x)  # (B, 3, 80, 80)
                logits_list.append(logits_t)
            logits = torch.stack(logits_list, dim=1)  # (B, L, 3, 80, 80)
        else:
            logits = self.classifier(x)  # (B, 3, 80, 80)

        return logits


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        attention = self.conv(x)  # (B, 1, H, W)
        return x * attention


class ImprovedSegmentationHead(nn.Module):
    """
    改进的分割头：多尺度特征 + 空间注意力

    特点：
    1. 多尺度特征提取（3个卷积分支）
    2. 空间注意力机制
    3. 渐进式上采样
    """
    def __init__(self, embed_dim: int = 768, num_classes: int = 3):
        super().__init__()

        # 多尺度特征提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 空间注意力
        self.attention = SpatialAttention(64)

        # 最终分类层
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, embed_dim, H, W) - 例如 (B, 768, 5, 5)

        Returns:
            logits: (B, num_classes, H_out, W_out) - 例如 (B, 3, 80, 80)
        """
        # 多尺度特征提取
        x = self.conv1(x)  # (B, 256, 5, 5)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)  # (B, 256, 20, 20)

        x = self.conv2(x)  # (B, 128, 20, 20)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 128, 40, 40)

        x = self.conv3(x)  # (B, 64, 40, 40)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 64, 80, 80)

        # 空间注意力
        x = self.attention(x)  # (B, 64, 80, 80)

        # 分类
        logits = self.classifier(x)  # (B, 3, 80, 80)

        return logits


class DecoderBlock(nn.Module):
    """UNet 解码器块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


class UNetSegmentationHead(nn.Module):
    """
    UNet-style 分割头 + 多尺度特征融合

    特点：
    1. 从 ViT encoder 的多个层提取特征（第 3、6、9、12 层）
    2. 使用跳跃连接融合多尺度特征
    3. 渐进式解码器上采样
    4. 10×10 → 20×20 → 40×40 → 80×80
    """
    def __init__(self, embed_dim: int = 768, num_classes: int = 3, patch_grid_size: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_grid_size = patch_grid_size  # 例如 10×10

        # 解码器路径（从深到浅）
        self.decoder1 = DecoderBlock(embed_dim, 512)      # 10×10 → 20×20
        self.decoder2 = DecoderBlock(512 + 512, 256)      # 20×20 → 40×40 (512 + skip)
        self.decoder3 = DecoderBlock(256 + 256, 128)      # 40×40 → 80×80 (256 + skip)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 跳跃连接的投影层（将不同层的特征投影到对应维度）
        self.skip_conv1 = nn.Conv2d(embed_dim, 512, kernel_size=1)  # layer9 → 512
        self.skip_conv2 = nn.Conv2d(embed_dim, 256, kernel_size=1)  # layer6 → 256
        self.skip_conv3 = nn.Conv2d(embed_dim, 128, kernel_size=1)  # layer3 → 128

        # 最终分类层
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, features_list):
        """
        Args:
            features_list: 多尺度特征列表（已经处理成 2D feature maps）
                - features_list[0]: 第 3 层特征 (B, 768, 10, 10)
                - features_list[1]: 第 6 层特征 (B, 768, 10, 10)
                - features_list[2]: 第 9 层特征 (B, 768, 10, 10)
                - features_list[3]: 第 12 层特征 (B, 768, 10, 10)

        Returns:
            logits: (B, num_classes, 80, 80)
        """
        feat_layer3 = features_list[0]   # (B, 768, 10, 10) - 浅层特征
        feat_layer6 = features_list[1]   # (B, 768, 10, 10) - 中层特征
        feat_layer9 = features_list[2]   # (B, 768, 10, 10) - 深层特征
        feat_layer12 = features_list[3]  # (B, 768, 10, 10) - 最深层特征

        # ========== 解码器路径（从最深层开始，逐步上采样并融合浅层特征）==========

        # Stage 1: 从最深层开始 (10×10 → 20×20)
        x = self.decoder1(feat_layer12)  # (B, 512, 20, 20)

        # 跳跃连接 1: 融合第 9 层特征
        skip1 = self.skip_conv1(feat_layer9)  # (B, 512, 10, 10)
        skip1 = F.interpolate(skip1, size=(20, 20), mode='bilinear', align_corners=False)  # (B, 512, 20, 20)
        x = torch.cat([x, skip1], dim=1)  # (B, 1024, 20, 20) → decoder2 会处理

        # Stage 2: 20×20 → 40×40
        x = self.decoder2(x)  # (B, 256, 40, 40)

        # 跳跃连接 2: 融合第 6 层特征
        skip2 = self.skip_conv2(feat_layer6)  # (B, 256, 10, 10)
        skip2 = F.interpolate(skip2, size=(40, 40), mode='bilinear', align_corners=False)  # (B, 256, 40, 40)
        x = torch.cat([x, skip2], dim=1)  # (B, 512, 40, 40)

        # Stage 3: 40×40 → 80×80
        x = self.decoder3(x)  # (B, 128, 80, 80)

        # 跳跃连接 3: 融合第 3 层特征（最浅层，细节最丰富）
        skip3 = self.skip_conv3(feat_layer3)  # (B, 128, 10, 10)
        skip3 = F.interpolate(skip3, size=(80, 80), mode='bilinear', align_corners=False)  # (B, 128, 80, 80)
        x = x + skip3  # (B, 128, 80, 80) - 残差连接

        # Stage 4: 最终特征提取
        x = self.decoder4(x)  # (B, 64, 80, 80)

        # 分类
        logits = self.classifier(x)  # (B, 3, 80, 80)
        return logits


class ASPPModule(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling) 模块"""
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: list):
        super().__init__()
        modules = []

        # 1×1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # 多个膨胀卷积
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # 全局平均池化
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        self.convs = nn.ModuleList(modules)

        # 融合所有分支
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # 全局池化分支需要上采样
        res[-1] = F.interpolate(res[-1], size=x.shape[2:], mode='bilinear', align_corners=False)

        res = torch.cat(res, dim=1)
        return self.project(res)


class ASPPSegmentationHead(nn.Module):
    """
    ASPP 分割头

    特点：
    1. 使用 ASPP 捕获多尺度上下文信息
    2. 渐进式解码器上采样
    3. 不需要多尺度特征（只用最后一层）
    """
    def __init__(self, embed_dim: int = 768, num_classes: int = 3, atrous_rates: list = [6, 12, 18]):
        super().__init__()

        # ASPP 模块
        self.aspp = ASPPModule(embed_dim, 256, atrous_rates)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 最终分类层
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, embed_dim, H, W) - 例如 (B, 768, 10, 10)

        Returns:
            logits: (B, num_classes, 80, 80)
        """
        # ASPP 提取多尺度特征
        x = self.aspp(x)  # (B, 256, 10, 10)

        # 解码器
        x = self.decoder(x)  # (B, 64, 10, 10)

        # 上采样到目标尺寸
        x = F.interpolate(x, size=(80, 80), mode='bilinear', align_corners=False)  # (B, 64, 80, 80)

        # 分类
        logits = self.classifier(x)  # (B, 3, 80, 80)
        return logits


class MultiBranchViT(nn.Module):
    """
    Multi-Branch Vision Transformer for SeasFire (新版)

    输入：
    - x_local: (B, T, 14, 80, 80) - T个时间步
    - x_global: (B, T, 14, 180, 360) - T个时间步
    - x_oci: (B, 10, 10) - 10个变量 × 10个时间步

    输出：
    - logits: (B, 3, 80, 80)  # 3 classes: no fire, fire, sea/masked
    """

    def __init__(
        self,
        use_local: bool = True,
        use_global: bool = True,
        use_oci: bool = True,
        load_pretrained_backbone: bool = False,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        local_patch_size: int = 8,
        global_patch_size: int = 60,
        temporal_steps: int = 4,
        oci_window: int = 10,
        use_improved_head: bool = False,
        seg_head_type: str = 'basic',  # 'basic', 'improved', 'unet', 'aspp'
        num_lead_times: int = 1,  # 多时间步长预测
        use_tcn: bool = False,  # 【改进】是否使用时序卷积网络
        tcn_kernel_size: int = 3,  # 【改进】TCN卷积核大小
        tcn_num_layers: int = 1,  # 【新增】TCN层数
        use_causal_conv: bool = False,  # 【新增】是否使用因果卷积
        use_temporal_transformer: bool = False,  # 【激进优化】是否使用时序Transformer
        temporal_transformer_layers: int = 2,  # 【激进优化】时序Transformer层数
        temporal_transformer_heads: int = 8,  # 【激进优化】时序Transformer注意力头数
    ):
        super().__init__()

        self.use_local = use_local
        self.use_global = use_global
        self.use_oci = use_oci
        self.embed_dim = embed_dim
        self.depth = depth
        self.temporal_steps = temporal_steps
        self.use_improved_head = use_improved_head
        self.seg_head_type = seg_head_type
        self.num_lead_times = num_lead_times
        self.use_tcn = use_tcn
        self.use_temporal_transformer = use_temporal_transformer

        # ========== Patch Embeddings ==========
        # Local: (B, T, 14, 80, 80) -> (B, 100, 768) [时序池化后]
        self.local_embed = TemporalLocalEncoder(
            in_channels=14,
            temporal_steps=temporal_steps,
            embed_dim=embed_dim,
            patch_size=local_patch_size,
            use_spatial_mixer=True,
            use_temporal_mixer=True
        )
        self.n_local_tokens = (80 // local_patch_size) ** 2  # 100 (不再乘以T)
        self.local_grid_size = 80 // local_patch_size  # 10

        # Global: (B, T, 14, 180, 360) -> (B, 18, 768) [时序池化后]
        self.global_embed = TemporalGlobalEncoder(
            in_channels=14,
            temporal_steps=temporal_steps,
            embed_dim=embed_dim,
            patch_size=global_patch_size,
            use_spatial_mixer=True,
            use_temporal_mixer=True
        )
        self.n_global_tokens = (180 // global_patch_size) * (360 // global_patch_size)  # 18 (不再乘以T)

        # OCI: (B, 10, 10) -> (B, 100, 768)
        self.oci_encoder = OCIEncoder(
            n_oci_vars=10,
            oci_window=oci_window,
            embed_dim=embed_dim
        )
        self.n_oci_tokens = 10 * oci_window  # 动态计算：10个变量 × oci_window个时间步

        # 【新增】ContextFiLM模块
        self.context_film = ContextFiLM(dim=embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 总 token 数: 1 + 100 + 18 + 100 = 219 (时序池化后)
        self.n_total_tokens = 1 + self.n_local_tokens + self.n_global_tokens + self.n_oci_tokens

        # Positional Embedding（可学习）
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_total_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 分支类型编码（让模型知道tokens来自哪个分支）
        self.local_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.global_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.oci_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.local_type_embed, std=0.02)
        nn.init.trunc_normal_(self.global_type_embed, std=0.02)
        nn.init.trunc_normal_(self.oci_type_embed, std=0.02)

        # ========== ViT Transformer Encoder ==========
        if load_pretrained_backbone:
            print("  加载 ViT-B/16 ImageNet 预训练权重...")
            from torchvision.models.vision_transformer import Encoder

            # 1. 先构建自己的encoder（使用正确的token数量）
            self.encoder = Encoder(
                seq_length=self.n_total_tokens,  # 573 tokens
                num_layers=depth,
                num_heads=num_heads,
                hidden_dim=embed_dim,
                mlp_dim=int(embed_dim * mlp_ratio),
                dropout=0.1,
                attention_dropout=0.1,
            )

            # 2. 加载预训练的ViT backbone
            vit_backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            pretrained_encoder = vit_backbone.encoder

            # 3. 只复制Transformer层的权重（跳过pos_embedding）
            print("  复制预训练的 Transformer 层权重...")
            for i in range(depth):
                # 复制每一层的权重
                self.encoder.layers[i].load_state_dict(
                    pretrained_encoder.layers[i].state_dict()
                )

            # 复制最后的LayerNorm
            self.encoder.ln.load_state_dict(pretrained_encoder.ln.state_dict())

            print(f"  ✓ 已加载预训练权重（12层Transformer + LayerNorm）")
            print(f"  ⚠️ 位置编码从头训练（token数量: 573 vs 预训练: 197）")
        else:
            # 手动构建 ViT encoder
            from torchvision.models.vision_transformer import Encoder
            self.encoder = Encoder(
                seq_length=self.n_total_tokens,
                num_layers=depth,
                num_heads=num_heads,
                hidden_dim=embed_dim,
                mlp_dim=int(embed_dim * mlp_ratio),
                dropout=0.1,
                attention_dropout=0.1,
            )
            print(f"  ✓ ViT Encoder dropout=0.1, attention_dropout=0.1")

        # ========== Segmentation Head ==========
        # 根据 seg_head_type 选择分割头
        if seg_head_type == 'patchwise':
            print(f"  使用 Patchwise 分割头（TeleViT1.0风格，自动适应patch_size）")
            self.seg_head = PatchwiseSegHead(
                dim=embed_dim,
                num_classes=3,
                patch_size=local_patch_size,
                grid_h=self.local_grid_size,
                grid_w=self.local_grid_size
            )
        elif seg_head_type == 'unet':
            print(f"  ⚠️  警告: UNet 分割头与新架构不兼容（需要多层特征，但时序池化已在encoder内完成）")
            print(f"  ⚠️  自动切换到 Patchwise 分割头")
            self.seg_head = PatchwiseSegHead(
                dim=embed_dim,
                num_classes=3,
                patch_size=local_patch_size,
                grid_h=self.local_grid_size,
                grid_w=self.local_grid_size
            )
            self.seg_head_type = 'patchwise'  # 更新类型
        elif seg_head_type == 'aspp':
            print(f"  使用 ASPP 分割头（多尺度上下文）")
            self.seg_head = ASPPSegmentationHead(embed_dim=embed_dim, num_classes=3)
        elif seg_head_type == 'improved' or use_improved_head:
            print(f"  使用改进的分割头（空间注意力）")
            self.seg_head = ImprovedSegmentationHead(embed_dim=embed_dim, num_classes=3)
        elif seg_head_type == 'progressive':
            print(f"  使用渐进式上采样分割头（在特征空间上采样）")
            if num_lead_times > 1:
                print(f"  多时间步长预测: {num_lead_times} 个 lead times")
            self.seg_head = ProgressiveSegmentationHead(embed_dim=embed_dim, num_classes=3, num_lead_times=num_lead_times)
        else:
            # 默认使用 patchwise 分割头
            print(f"  使用 Patchwise 分割头（默认，TeleViT1.0风格）")
            self.seg_head = PatchwiseSegHead(
                dim=embed_dim,
                num_classes=3,
                patch_size=local_patch_size,
                grid_h=self.local_grid_size,
                grid_w=self.local_grid_size
            )

        # 初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x_local, x_global, x_oci):
        """
        Args:
            x_local: (B, T, 14, 80, 80) - T个时间步的Local输入
            x_global: (B, T, 14, 180, 360) - T个时间步的Global输入
            x_oci: (B, 10, 10) - OCI数据

        Returns:
            logits: (B, 3, 80, 80)  # 3 classes for CrossEntropyLoss
        """
        B = x_local.size(0)
        tokens = []

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        tokens.append(cls_tokens)

        # ========== Local Tokens ==========
        # 时序池化已在encoder内部完成
        if self.use_local:
            local_tokens = self.local_embed(x_local)  # (B, 100, 768)
        else:
            local_tokens = torch.zeros(B, self.n_local_tokens, self.embed_dim, device=x_local.device)

        # ========== Global Tokens ==========
        # 时序池化已在encoder内部完成
        if self.use_global:
            global_tokens = self.global_embed(x_global)  # (B, 18, 768)
        else:
            global_tokens = torch.zeros(B, self.n_global_tokens, self.embed_dim, device=x_local.device)

        # ========== OCI Tokens ==========
        if self.use_oci:
            oci_tokens = self.oci_encoder(x_oci)  # (B, 100, 768)
        else:
            oci_tokens = torch.zeros(B, self.n_oci_tokens, self.embed_dim, device=x_local.device)

        # 【新增】ContextFiLM: 使用Global和OCI调制Local tokens
        local_tokens = self.context_film(local_tokens, global_tokens, oci_tokens)

        # 添加分支类型编码
        local_tokens = local_tokens + self.local_type_embed
        tokens.append(local_tokens)

        global_tokens = global_tokens + self.global_type_embed
        tokens.append(global_tokens)

        oci_tokens = oci_tokens + self.oci_type_embed
        tokens.append(oci_tokens)

        # 拼接所有 tokens: (B, 1+100+18+100=219, 768)
        x = torch.cat(tokens, dim=1)

        # 添加位置编码
        x = x + self.pos_embed

        # ========== ViT Encoder ==========
        x = self.encoder(x)  # (B, 219, 768)

        # 提取 local tokens（跳过 CLS）
        local_out = x[:, 1:1+self.n_local_tokens, :]  # (B, 100, 768)

        # 通过分割头
        if self.seg_head_type == 'patchwise' or (self.seg_head_type not in ['unet', 'aspp', 'improved', 'progressive']):
            # Patchwise分割头：直接处理tokens
            logits = self.seg_head(local_out)  # (B, 3, 80, 80)
        else:
            # 其他分割头：需要reshape成2D feature map
            h = w = self.local_grid_size
            local_out = local_out.transpose(1, 2).reshape(B, self.embed_dim, h, w)  # (B, 768, 10, 10)
            logits = self.seg_head(local_out)  # (B, 3, 80, 80)

        return logits


def create_model(config: dict) -> MultiBranchViT:
    """根据配置创建模型"""
    # 解析 lead_time_steps
    lead_time_steps = config['data'].get('lead_time_steps', 1)
    if isinstance(lead_time_steps, int):
        num_lead_times = 1
    elif isinstance(lead_time_steps, (list, tuple)):
        num_lead_times = len(lead_time_steps)
    else:
        num_lead_times = 1

    # 【改进】解析TCN配置
    use_tcn = config['model'].get('use_tcn', False)
    tcn_kernel_size = config['model'].get('tcn_kernel_size', 3)
    tcn_num_layers = config['model'].get('tcn_num_layers', 1)  # 【新增】
    use_causal_conv = config['model'].get('use_causal_conv', False)  # 【新增】

    # 【激进优化】解析时序Transformer配置
    use_temporal_transformer = config['model'].get('use_temporal_transformer', False)
    temporal_transformer_layers = config['model'].get('temporal_transformer_layers', 2)
    temporal_transformer_heads = config['model'].get('temporal_transformer_heads', 8)

    model = MultiBranchViT(
        use_local=config['model']['use_local'],
        use_global=config['model']['use_global'],
        use_oci=config['model']['use_oci'],
        load_pretrained_backbone=config['model']['load_pretrained_backbone'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model']['mlp_ratio'],
        local_patch_size=config['model']['local_patch_size'],
        global_patch_size=config['model']['global_patch_size'],
        temporal_steps=config['data'].get('temporal_steps', 4),
        oci_window=config['data']['oci_window'],
        use_improved_head=config['model'].get('use_improved_head', False),
        seg_head_type=config['model'].get('seg_head_type', 'basic'),
        num_lead_times=num_lead_times,
        use_tcn=use_tcn,  # 【改进】TCN开关
        tcn_kernel_size=tcn_kernel_size,  # 【改进】TCN卷积核大小
        tcn_num_layers=tcn_num_layers,  # 【新增】TCN层数
        use_causal_conv=use_causal_conv,  # 【新增】因果卷积开关
        use_temporal_transformer=use_temporal_transformer,  # 【激进优化】时序Transformer开关
        temporal_transformer_layers=temporal_transformer_layers,  # 【激进优化】时序Transformer层数
        temporal_transformer_heads=temporal_transformer_heads,  # 【激进优化】时序Transformer注意力头数
    )

    # 打印模型信息
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {n_params / 1e6:.2f}M")

    return model


if __name__ == '__main__':
    # 测试模型
    print("=" * 60)
    print("测试新版 MultiBranchViT 模型")
    print("=" * 60)

    model = MultiBranchViT(
        use_local=True,
        use_global=True,
        use_oci=True,
        load_pretrained_backbone=False,
        local_patch_size=8,
        global_patch_size=60,
        temporal_steps=4,
        oci_window=10,
    )

    # 随机输入（新版维度）
    batch_size = 2
    temporal_steps = 4
    x_local = torch.randn(batch_size, temporal_steps, 14, 80, 80)
    x_global = torch.randn(batch_size, temporal_steps, 14, 180, 360)
    x_oci = torch.randn(batch_size, 10, 10)

    print(f"\n输入维度:")
    print(f"  x_local:  {x_local.shape}")
    print(f"  x_global: {x_global.shape}")
    print(f"  x_oci:    {x_oci.shape}")

    # 前向传播
    print(f"\n前向传播...")
    logits = model(x_local, x_global, x_oci)
    print(f"✓ 输出形状: {logits.shape}")  # (2, 3, 80, 80)

    # 参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {n_params / 1e6:.2f}M")

    print("\n" + "=" * 60)
    print("测试通过！")
    print("=" * 60)
