"""Brief implementation note."""
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
    """Brief implementation note."""
    def __init__(self, channels: int, kernel_size: int = 5, num_layers: int = 2, use_causal: bool = True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.use_causal = use_causal

        
        if use_causal:
            
            self.padding = (kernel_size - 1)
        else:
            
            self.padding = (kernel_size - 1) // 2

        
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=self.padding if not use_causal else 0,  
                    groups=channels,  
                    bias=False
                ),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                
                nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(channels),
            )
            self.tcn_layers.append(layer)

        
        self.gates = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])

    def forward(self, x):
        """Brief implementation note."""
        out = x

        for i, layer in enumerate(self.tcn_layers):
            residual = out

            
            if self.use_causal:
                out = F.pad(out, (self.padding, 0))  

            
            out = layer(out)

            
            if self.use_causal and out.size(2) > residual.size(2):
                out = out[:, :, :residual.size(2)]

            
            out = residual + self.gates[i] * out

        return out


class TemporalTransformer(nn.Module):
    """Brief implementation note."""
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,  
            dropout=0.0,
            activation='gelu',
            batch_first=False,  # (T, B*N, C)
            norm_first=True  
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        
        self.norm = nn.LayerNorm(embed_dim)

        print(f"Status")

    def forward(self, x):
        """Brief implementation note."""
        B, T, N, C = x.shape

        
        
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, C)  # (B*N, T, C)
        x = x.permute(1, 0, 2)  # (T, B*N, C)

        
        x = self.transformer(x)  # (T, B*N, C)
        x = self.norm(x)

        
        x = x.permute(1, 0, 2)  # (B*N, T, C)
        x = x.reshape(B, N, T, C).permute(0, 2, 1, 3)  # (B, T, N, C)

        return x


class SpatialTokenMixer2D(nn.Module):
    """Brief implementation note."""
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
    """Brief implementation note."""
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
    """Brief implementation note."""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=False  
        )

        
        self.norm = nn.LayerNorm(channels)

        
        self.query = nn.Parameter(torch.randn(1, 1, channels))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x):
        """Brief implementation note."""
        B, C, T, H, W = x.shape

        
        x = x.permute(2, 0, 3, 4, 1)  # (T, B, H, W, C)
        x = x.reshape(T, B * H * W, C)  # (T, B*H*W, C)

        
        query = self.query.expand(1, B * H * W, C)  # (1, B*H*W, C)

        
        # attn_output: (1, B*H*W, C)
        attn_output, _ = self.multihead_attn(
            query=query,      # (1, B*H*W, C)
            key=x,            # (T, B*H*W, C)
            value=x,          # (T, B*H*W, C)
        )

        
        attn_output = self.norm(attn_output)  # (1, B*H*W, C)

        
        attn_output = attn_output.squeeze(0)  # (B*H*W, C)
        attn_output = attn_output.view(B, H, W, C)  # (B, H, W, C)
        attn_output = attn_output.permute(0, 3, 1, 2)  # (B, C, H, W)

        return attn_output


class TemporalAttentionPoolMHA(nn.Module):
    """Brief implementation note."""
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
    """Brief implementation note."""
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

        
        if use_spatial_mixer:
            self.spatial_mixer = SpatialTokenMixer2D(dim=embed_dim)
            print(f"Status")

        
        if use_temporal_mixer:
            self.temporal_mixer = TemporalTokenMixerTCN(dim=embed_dim, kernel_size=3)
            print(f"Status")

        
        self.temporal_pool = TemporalAttentionPoolMHA(dim=embed_dim, num_heads=4)
        print(f"Status")

    def forward(self, x):
        """Brief implementation note."""
        B, T, C, H, W = x.shape

        
        x_reshaped = x.reshape(B * T, C, H, W)  # (B*T, 14, 80, 80)
        patches = self.patch_embed(x_reshaped)  # (B*T, 768, 10, 10)

        
        if self.use_spatial_mixer:
            patches = self.spatial_mixer(patches)  # (B*T, 768, 10, 10)

        
        h_grid = H // self.patch_size
        w_grid = W // self.patch_size
        patches = patches.view(B, T, self.embed_dim, h_grid, w_grid)  # (B, T, 768, 10, 10)
        patches = patches.permute(0, 1, 3, 4, 2)  # (B, T, 10, 10, 768)
        patches = patches.reshape(B, T, h_grid * w_grid, self.embed_dim)  # (B, T, 100, 768)

        
        if self.use_temporal_mixer:
            patches = self.temporal_mixer(patches)  # (B, T, 100, 768)

        
        tokens = self.temporal_pool(patches)  # (B, 100, 768)

        return tokens


class TemporalGlobalEncoder(nn.Module):
    """Brief implementation note."""
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

        
        if use_spatial_mixer:
            self.spatial_mixer = SpatialTokenMixer2D(dim=embed_dim)
            print(f"Status")

        
        if use_temporal_mixer:
            self.temporal_mixer = TemporalTokenMixerTCN(dim=embed_dim, kernel_size=3)
            print(f"Status")

        
        self.temporal_pool = TemporalAttentionPoolMHA(dim=embed_dim, num_heads=4)
        print(f"Status")

    def forward(self, x):
        """Brief implementation note."""
        B, T, C, H, W = x.shape

        
        x_reshaped = x.reshape(B * T, C, H, W)  # (B*T, 14, 180, 360)
        patches = self.patch_embed(x_reshaped)  # (B*T, 768, 3, 6)

        
        if self.use_spatial_mixer:
            patches = self.spatial_mixer(patches)  # (B*T, 768, 3, 6)

        
        h_grid = H // self.patch_size
        w_grid = W // self.patch_size
        patches = patches.view(B, T, self.embed_dim, h_grid, w_grid)  # (B, T, 768, 3, 6)
        patches = patches.permute(0, 1, 3, 4, 2)  # (B, T, 3, 6, 768)
        patches = patches.reshape(B, T, h_grid * w_grid, self.embed_dim)  # (B, T, 18, 768)

        
        if self.use_temporal_mixer:
            patches = self.temporal_mixer(patches)  # (B, T, 18, 768)

        
        tokens = self.temporal_pool(patches)  # (B, 18, 768)

        return tokens


class OCIEncoder(nn.Module):
    """Brief implementation note."""
    def __init__(self, n_oci_vars: int = 10, oci_window: int = 10, embed_dim: int = 768):
        super().__init__()
        self.n_oci_vars = n_oci_vars
        self.oci_window = oci_window
        self.embed_dim = embed_dim

        
        self.value_proj = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        
        self.var_pos_embed = nn.Parameter(
            torch.zeros(1, n_oci_vars, 1, embed_dim)
        )
        nn.init.trunc_normal_(self.var_pos_embed, std=0.02)

        
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, 1, oci_window, embed_dim)
        )
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

    def forward(self, x_oci):
        """Brief implementation note."""
        B = x_oci.size(0)

        
        # (B, 10, 10) → (B, 10, 10, 1) → (B, 10, 10, 768)
        x = x_oci.unsqueeze(-1)  # (B, 10, 10, 1)
        x = self.value_proj(x)  # (B, 10, 10, 768)

        
        
        
        x = x + self.var_pos_embed + self.time_pos_embed

        
        tokens = x.reshape(B, self.n_oci_vars * self.oci_window, self.embed_dim)

        return tokens


class ContextFiLM(nn.Module):
    """Brief implementation note."""
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*dim, 2*dim),
            nn.GELU(),
            nn.Linear(2*dim, 2*dim),
        )
        print(f"Status")

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
    """Brief implementation note."""
    def __init__(self, dim: int, num_classes: int, patch_size: int, grid_h: int, grid_w: int):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.proj = nn.Linear(dim, num_classes * patch_size * patch_size)

    def forward(self, local_tokens):  # (B, N, D)
        """Brief implementation note."""
        B, N, D = local_tokens.shape
        C = self.num_classes
        p = self.patch_size
        H, W = self.grid_h, self.grid_w
        x = self.proj(local_tokens)  # (B, N, C*p*p)
        x = x.view(B, H, W, C, p, p)                 # (B,H,W,C,p,p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous() # (B,C,H,p,W,p)
        return x.view(B, C, H*p, W*p)                # (B,C,80,80)


class ProgressiveSegmentationHead(nn.Module):
    """Brief implementation note."""
    def __init__(self, embed_dim: int = 768, num_classes: int = 3, num_lead_times: int = 1):
        super().__init__()
        self.num_lead_times = num_lead_times

        
        self.stage1 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        
        self.stage2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        
        self.stage4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        
        if num_lead_times > 1:
            self.classifiers = nn.ModuleList([
                nn.Conv2d(32, num_classes, kernel_size=1) for _ in range(num_lead_times)
            ])
        else:
            self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        """Brief implementation note."""
        
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
    """Brief implementation note."""
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
    """Brief implementation note."""
    def __init__(self, embed_dim: int = 768, num_classes: int = 3):
        super().__init__()

        
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

        
        self.attention = SpatialAttention(64)

        
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """Brief implementation note."""
        
        x = self.conv1(x)  # (B, 256, 5, 5)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)  # (B, 256, 20, 20)

        x = self.conv2(x)  # (B, 128, 20, 20)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 128, 40, 40)

        x = self.conv3(x)  # (B, 64, 40, 40)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 64, 80, 80)

        
        x = self.attention(x)  # (B, 64, 80, 80)

        
        logits = self.classifier(x)  # (B, 3, 80, 80)

        return logits


class DecoderBlock(nn.Module):
    """Brief implementation note."""
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
    """Brief implementation note."""
    def __init__(self, embed_dim: int = 768, num_classes: int = 3, patch_grid_size: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_grid_size = patch_grid_size  

        
        self.decoder1 = DecoderBlock(embed_dim, 512)      # 10×10 → 20×20
        self.decoder2 = DecoderBlock(512 + 512, 256)      # 20×20 → 40×40 (512 + skip)
        self.decoder3 = DecoderBlock(256 + 256, 128)      # 40×40 → 80×80 (256 + skip)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        
        self.skip_conv1 = nn.Conv2d(embed_dim, 512, kernel_size=1)  # layer9 → 512
        self.skip_conv2 = nn.Conv2d(embed_dim, 256, kernel_size=1)  # layer6 → 256
        self.skip_conv3 = nn.Conv2d(embed_dim, 128, kernel_size=1)  # layer3 → 128

        
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, features_list):
        """Brief implementation note."""
        feat_layer3 = features_list[0]   
        feat_layer6 = features_list[1]   
        feat_layer9 = features_list[2]   
        feat_layer12 = features_list[3]  

        

        
        x = self.decoder1(feat_layer12)  # (B, 512, 20, 20)

        
        skip1 = self.skip_conv1(feat_layer9)  # (B, 512, 10, 10)
        skip1 = F.interpolate(skip1, size=(20, 20), mode='bilinear', align_corners=False)  # (B, 512, 20, 20)
        x = torch.cat([x, skip1], dim=1)  

        # Stage 2: 20×20 → 40×40
        x = self.decoder2(x)  # (B, 256, 40, 40)

        
        skip2 = self.skip_conv2(feat_layer6)  # (B, 256, 10, 10)
        skip2 = F.interpolate(skip2, size=(40, 40), mode='bilinear', align_corners=False)  # (B, 256, 40, 40)
        x = torch.cat([x, skip2], dim=1)  # (B, 512, 40, 40)

        # Stage 3: 40×40 → 80×80
        x = self.decoder3(x)  # (B, 128, 80, 80)

        
        skip3 = self.skip_conv3(feat_layer3)  # (B, 128, 10, 10)
        skip3 = F.interpolate(skip3, size=(80, 80), mode='bilinear', align_corners=False)  # (B, 128, 80, 80)
        x = x + skip3  

        
        x = self.decoder4(x)  # (B, 64, 80, 80)

        
        logits = self.classifier(x)  # (B, 3, 80, 80)
        return logits


class ASPPModule(nn.Module):
    """Brief implementation note."""
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: list):
        super().__init__()
        modules = []

        
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        self.convs = nn.ModuleList(modules)

        
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

        
        res[-1] = F.interpolate(res[-1], size=x.shape[2:], mode='bilinear', align_corners=False)

        res = torch.cat(res, dim=1)
        return self.project(res)


class ASPPSegmentationHead(nn.Module):
    """Brief implementation note."""
    def __init__(self, embed_dim: int = 768, num_classes: int = 3, atrous_rates: list = [6, 12, 18]):
        super().__init__()

        
        self.aspp = ASPPModule(embed_dim, 256, atrous_rates)

        
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

        
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """Brief implementation note."""
        
        x = self.aspp(x)  # (B, 256, 10, 10)

        
        x = self.decoder(x)  # (B, 64, 10, 10)

        
        x = F.interpolate(x, size=(80, 80), mode='bilinear', align_corners=False)  # (B, 64, 80, 80)

        
        logits = self.classifier(x)  # (B, 3, 80, 80)
        return logits


class MultiBranchViT(nn.Module):
    """Brief implementation note."""

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
        num_lead_times: int = 1,  
        use_tcn: bool = False,  
        tcn_kernel_size: int = 3,  
        tcn_num_layers: int = 1,  
        use_causal_conv: bool = False,  
        use_temporal_transformer: bool = False,  
        temporal_transformer_layers: int = 2,  
        temporal_transformer_heads: int = 8,  
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
        
        self.local_embed = TemporalLocalEncoder(
            in_channels=14,
            temporal_steps=temporal_steps,
            embed_dim=embed_dim,
            patch_size=local_patch_size,
            use_spatial_mixer=True,
            use_temporal_mixer=True
        )
        self.n_local_tokens = (80 // local_patch_size) ** 2  
        self.local_grid_size = 80 // local_patch_size  # 10

        
        self.global_embed = TemporalGlobalEncoder(
            in_channels=14,
            temporal_steps=temporal_steps,
            embed_dim=embed_dim,
            patch_size=global_patch_size,
            use_spatial_mixer=True,
            use_temporal_mixer=True
        )
        self.n_global_tokens = (180 // global_patch_size) * (360 // global_patch_size)  

        # OCI: (B, 10, 10) -> (B, 100, 768)
        self.oci_encoder = OCIEncoder(
            n_oci_vars=10,
            oci_window=oci_window,
            embed_dim=embed_dim
        )
        self.n_oci_tokens = 10 * oci_window  

        
        self.context_film = ContextFiLM(dim=embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        
        self.n_total_tokens = 1 + self.n_local_tokens + self.n_global_tokens + self.n_oci_tokens

        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_total_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        
        self.local_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.global_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.oci_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.local_type_embed, std=0.02)
        nn.init.trunc_normal_(self.global_type_embed, std=0.02)
        nn.init.trunc_normal_(self.oci_type_embed, std=0.02)

        # ========== ViT Transformer Encoder ==========
        if load_pretrained_backbone:
            print("Status")
            from torchvision.models.vision_transformer import Encoder

            
            self.encoder = Encoder(
                seq_length=self.n_total_tokens,  # 573 tokens
                num_layers=depth,
                num_heads=num_heads,
                hidden_dim=embed_dim,
                mlp_dim=int(embed_dim * mlp_ratio),
                dropout=0.1,
                attention_dropout=0.1,
            )

            
            vit_backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            pretrained_encoder = vit_backbone.encoder

            
            print("Status")
            for i in range(depth):
                
                self.encoder.layers[i].load_state_dict(
                    pretrained_encoder.layers[i].state_dict()
                )

            
            self.encoder.ln.load_state_dict(pretrained_encoder.ln.state_dict())

            print(f"Status")
            print(f"Status")
        else:
            
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
            print(f"Status")

        # ========== Segmentation Head ==========
        
        if seg_head_type == 'patchwise':
            print(f"Status")
            self.seg_head = PatchwiseSegHead(
                dim=embed_dim,
                num_classes=3,
                patch_size=local_patch_size,
                grid_h=self.local_grid_size,
                grid_w=self.local_grid_size
            )
        elif seg_head_type == 'unet':
            print(f"Status")
            print(f"Status")
            self.seg_head = PatchwiseSegHead(
                dim=embed_dim,
                num_classes=3,
                patch_size=local_patch_size,
                grid_h=self.local_grid_size,
                grid_w=self.local_grid_size
            )
            self.seg_head_type = 'patchwise'  
        elif seg_head_type == 'aspp':
            print(f"Status")
            self.seg_head = ASPPSegmentationHead(embed_dim=embed_dim, num_classes=3)
        elif seg_head_type == 'improved' or use_improved_head:
            print(f"Status")
            self.seg_head = ImprovedSegmentationHead(embed_dim=embed_dim, num_classes=3)
        elif seg_head_type == 'progressive':
            print(f"Status")
            if num_lead_times > 1:
                print(f"Status")
            self.seg_head = ProgressiveSegmentationHead(embed_dim=embed_dim, num_classes=3, num_lead_times=num_lead_times)
        else:
            
            print(f"Status")
            self.seg_head = PatchwiseSegHead(
                dim=embed_dim,
                num_classes=3,
                patch_size=local_patch_size,
                grid_h=self.local_grid_size,
                grid_w=self.local_grid_size
            )

        
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x_local, x_global, x_oci):
        """Brief implementation note."""
        B = x_local.size(0)
        tokens = []

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        tokens.append(cls_tokens)

        # ========== Local Tokens ==========
        
        if self.use_local:
            local_tokens = self.local_embed(x_local)  # (B, 100, 768)
        else:
            local_tokens = torch.zeros(B, self.n_local_tokens, self.embed_dim, device=x_local.device)

        # ========== Global Tokens ==========
        
        if self.use_global:
            global_tokens = self.global_embed(x_global)  # (B, 18, 768)
        else:
            global_tokens = torch.zeros(B, self.n_global_tokens, self.embed_dim, device=x_local.device)

        # ========== OCI Tokens ==========
        if self.use_oci:
            oci_tokens = self.oci_encoder(x_oci)  # (B, 100, 768)
        else:
            oci_tokens = torch.zeros(B, self.n_oci_tokens, self.embed_dim, device=x_local.device)

        
        local_tokens = self.context_film(local_tokens, global_tokens, oci_tokens)

        
        local_tokens = local_tokens + self.local_type_embed
        tokens.append(local_tokens)

        global_tokens = global_tokens + self.global_type_embed
        tokens.append(global_tokens)

        oci_tokens = oci_tokens + self.oci_type_embed
        tokens.append(oci_tokens)

        
        x = torch.cat(tokens, dim=1)

        
        x = x + self.pos_embed

        # ========== ViT Encoder ==========
        x = self.encoder(x)  # (B, 219, 768)

        
        local_out = x[:, 1:1+self.n_local_tokens, :]  # (B, 100, 768)

        
        if self.seg_head_type == 'patchwise' or (self.seg_head_type not in ['unet', 'aspp', 'improved', 'progressive']):
            
            logits = self.seg_head(local_out)  # (B, 3, 80, 80)
        else:
            
            h = w = self.local_grid_size
            local_out = local_out.transpose(1, 2).reshape(B, self.embed_dim, h, w)  # (B, 768, 10, 10)
            logits = self.seg_head(local_out)  # (B, 3, 80, 80)

        return logits


def create_model(config: dict) -> MultiBranchViT:
    """Brief implementation note."""
    
    lead_time_steps = config['data'].get('lead_time_steps', 1)
    if isinstance(lead_time_steps, int):
        num_lead_times = 1
    elif isinstance(lead_time_steps, (list, tuple)):
        num_lead_times = len(lead_time_steps)
    else:
        num_lead_times = 1

    
    use_tcn = config['model'].get('use_tcn', False)
    tcn_kernel_size = config['model'].get('tcn_kernel_size', 3)
    tcn_num_layers = config['model'].get('tcn_num_layers', 1)  
    use_causal_conv = config['model'].get('use_causal_conv', False)  

    
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
        use_tcn=use_tcn,  
        tcn_kernel_size=tcn_kernel_size,  
        tcn_num_layers=tcn_num_layers,  
        use_causal_conv=use_causal_conv,  
        use_temporal_transformer=use_temporal_transformer,  
        temporal_transformer_layers=temporal_transformer_layers,  
        temporal_transformer_heads=temporal_transformer_heads,  
    )

    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Status")

    return model


if __name__ == '__main__':
    
    print("=" * 60)
    print("Status")
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

    
    batch_size = 2
    temporal_steps = 4
    x_local = torch.randn(batch_size, temporal_steps, 14, 80, 80)
    x_global = torch.randn(batch_size, temporal_steps, 14, 180, 360)
    x_oci = torch.randn(batch_size, 10, 10)

    print(f"Status")
    print(f"  x_local:  {x_local.shape}")
    print(f"  x_global: {x_global.shape}")
    print(f"  x_oci:    {x_oci.shape}")

    
    print(f"Status")
    logits = model(x_local, x_global, x_oci)
    print(f"Status")  # (2, 3, 80, 80)

    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Status")

    print("\n" + "=" * 60)
    print("Status")
    print("=" * 60)
