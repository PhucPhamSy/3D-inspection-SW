"""
Model architectures for 3D AI Labelling module.
Supports: TinyUNet, ResUNet++, Mask2Former

All models follow the same interface:
    model = create_model(arch_name, in_channels, num_classes)
    logits = model(input_tensor)  # (B, num_classes, H, W)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TinyUNet  (lightweight baseline, fast iteration)
# ═══════════════════════════════════════════════════════════════════════════════
class TinyUNet(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(16, out_c, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        up_e2 = F.interpolate(self.up(e2), size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([up_e2, e1], dim=1))
        return self.final(d1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ResUNet++  (Jha et al., 2019)
#    Components: Residual blocks + Squeeze-and-Excitation + ASPP + Attention Gates
# ═══════════════════════════════════════════════════════════════════════════════

class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ResBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.se = SqueezeExcite(out_c)
        # Skip connection
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride),
            nn.BatchNorm2d(out_c),
        ) if in_c != out_c or stride != 1 else nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        out = self.se(out)
        return out + skip


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_c, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU(True))
        self.conv3_d6 = nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=6, dilation=6), nn.BatchNorm2d(out_c), nn.ReLU(True))
        self.conv3_d12 = nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=12, dilation=12), nn.BatchNorm2d(out_c), nn.ReLU(True))
        self.conv3_d18 = nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=18, dilation=18), nn.BatchNorm2d(out_c), nn.ReLU(True))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_c, out_c, 1), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(out_c * 5, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU(True))

    def forward(self, x):
        size = x.shape[2:]
        p = F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=True)
        out = torch.cat([self.conv1x1(x), self.conv3_d6(x), self.conv3_d12(x), self.conv3_d18(x), p], dim=1)
        return self.project(out)


class AttentionGate(nn.Module):
    """Attention gate for skip connections."""
    def __init__(self, gate_c, skip_c, inter_c):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(gate_c, inter_c, 1), nn.BatchNorm2d(inter_c))
        self.W_x = nn.Sequential(nn.Conv2d(skip_c, inter_c, 1), nn.BatchNorm2d(inter_c))
        self.psi = nn.Sequential(nn.Conv2d(inter_c, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, gate, skip):
        g = self.W_g(gate)
        x = self.W_x(skip)
        g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        psi = self.psi(F.relu(g + x, inplace=True))
        return skip * psi


class ResUNetPlusPlus(nn.Module):
    """
    ResUNet++ Architecture (Jha et al.)
    Encoder: 4 ResBlocks with SE, stride-2 downsampling
    Bridge:  ASPP
    Decoder: 3 upsampling stages with Attention Gates + ResBlocks
    """
    def __init__(self, in_c=1, out_c=2, filters=None):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128, 256]
        f = filters

        # Encoder
        self.enc1 = ResBlock(in_c, f[0])
        self.enc2 = ResBlock(f[0], f[1], stride=2)
        self.enc3 = ResBlock(f[1], f[2], stride=2)
        self.enc4 = ResBlock(f[2], f[3], stride=2)

        # Bridge
        self.bridge = ASPP(f[3], f[3])

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = AttentionGate(f[3], f[2], f[2] // 2)
        self.dec3 = ResBlock(f[3] + f[2], f[2])

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionGate(f[2], f[1], f[1] // 2)
        self.dec2 = ResBlock(f[2] + f[1], f[1])

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionGate(f[1], f[0], f[0] // 2)
        self.dec1 = ResBlock(f[1] + f[0], f[0])

        self.final = nn.Conv2d(f[0], out_c, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # (B, f0, H, W)
        e2 = self.enc2(e1)         # (B, f1, H/2, W/2)
        e3 = self.enc3(e2)         # (B, f2, H/4, W/4)
        e4 = self.enc4(e3)         # (B, f3, H/8, W/8)

        # Bridge
        b = self.bridge(e4)        # (B, f3, H/8, W/8)

        # Decoder
        d3 = self.up3(b)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        e3_att = self.att3(b, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        e2_att = self.att2(d3, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        e1_att = self.att1(d2, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return self.final(d1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Mask2Former  (Cheng et al., 2022 — simplified for 2D segmentation)
#    Components: Pixel decoder (FPN) + Transformer decoder with masked attention
# ═══════════════════════════════════════════════════════════════════════════════

class _ConvBnReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.layer(x)


class SimpleBackbone(nn.Module):
    """Lightweight CNN backbone producing multi-scale features."""
    def __init__(self, in_c, base=32):
        super().__init__()
        self.stage1 = nn.Sequential(_ConvBnReLU(in_c, base), _ConvBnReLU(base, base))
        self.stage2 = nn.Sequential(nn.MaxPool2d(2), _ConvBnReLU(base, base*2), _ConvBnReLU(base*2, base*2))
        self.stage3 = nn.Sequential(nn.MaxPool2d(2), _ConvBnReLU(base*2, base*4), _ConvBnReLU(base*4, base*4))
        self.stage4 = nn.Sequential(nn.MaxPool2d(2), _ConvBnReLU(base*4, base*8), _ConvBnReLU(base*8, base*8))

    def forward(self, x):
        f1 = self.stage1(x)     # (B, 32,  H, W)
        f2 = self.stage2(f1)    # (B, 64,  H/2, W/2)
        f3 = self.stage3(f2)    # (B, 128, H/4, W/4)
        f4 = self.stage4(f3)    # (B, 256, H/8, W/8)
        return [f1, f2, f3, f4]


class PixelDecoderFPN(nn.Module):
    """Simple FPN-like pixel decoder."""
    def __init__(self, feature_channels, hidden_dim=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, 1) for c in feature_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GroupNorm(8, hidden_dim), nn.ReLU(True))
            for _ in feature_channels
        ])
        self.mask_features = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, features):
        # Top-down path
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, features)]
        # FPN fusion from coarse to fine
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], mode='bilinear', align_corners=True)
            laterals[i] = laterals[i] + up
        # Output convolutions
        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        # Finest-scale mask features
        mask_feats = self.mask_features(outs[0])
        # Memory for transformer = coarsest feature flattened
        memory = outs[-1]
        return mask_feats, memory


class MaskedTransformerDecoderLayer(nn.Module):
    """Single layer of the Mask2Former transformer decoder."""
    def __init__(self, d_model=128, nhead=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries, memory, attn_mask=None):
        # Self-attention among queries
        q2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + q2)
        # Cross-attention: queries attend to pixel memory with optional mask
        q2, _ = self.cross_attn(queries, memory, memory, attn_mask=attn_mask)
        queries = self.norm2(queries + q2)
        # FFN
        queries = self.norm3(queries + self.ff(queries))
        return queries


class Mask2Former(nn.Module):
    """
    Simplified Mask2Former for 2D semantic/instance segmentation.
    
    Architecture:
        1. CNN Backbone → multi-scale features
        2. Pixel Decoder (FPN) → per-pixel features + memory
        3. Transformer Decoder → N learnable queries → N mask embeddings
        4. Dot product of mask embeddings × pixel features → N binary masks
        5. Class prediction per query
        6. Combine into semantic output (argmax across queries per class)
    """
    def __init__(self, in_c=1, out_c=2, hidden_dim=128, num_queries=32,
                 num_heads=4, num_layers=3, base_channels=32):
        super().__init__()
        self.num_classes = out_c
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Backbone
        self.backbone = SimpleBackbone(in_c, base=base_channels)
        feat_channels = [base_channels, base_channels*2, base_channels*4, base_channels*8]

        # Pixel decoder
        self.pixel_decoder = PixelDecoderFPN(feat_channels, hidden_dim)

        # Learnable queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Transformer decoder
        self.decoder_layers = nn.ModuleList([
            MaskedTransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 2)
            for _ in range(num_layers)
        ])

        # Prediction heads
        self.class_head = nn.Linear(hidden_dim, out_c + 1)  # +1 for "no-object"
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Backbone
        features = self.backbone(x)

        # 2. Pixel decoder
        mask_features, memory_2d = self.pixel_decoder(features)
        # mask_features: (B, D, H, W)
        # memory_2d: (B, D, H/8, W/8) -> flatten to sequence
        Bm, Dm, Hm, Wm = memory_2d.shape
        memory = memory_2d.flatten(2).permute(0, 2, 1)  # (B, Hm*Wm, D)

        # 3. Transformer decoder
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)
        for layer in self.decoder_layers:
            queries = layer(queries, memory)

        # 4. Predictions
        class_logits = self.class_head(queries)  # (B, Q, num_classes+1)
        mask_embeds = self.mask_embed(queries)    # (B, Q, D)

        # 5. Mask predictions via dot product
        # mask_features: (B, D, H, W) -> (B, D, H*W)
        mask_feats_flat = mask_features.flatten(2)  # (B, D, H*W)
        mask_logits = torch.bmm(mask_embeds, mask_feats_flat)  # (B, Q, H*W)
        mask_logits = mask_logits.view(B, self.num_queries, H, W)

        # 6. Combine into semantic segmentation output: (B, num_classes, H, W)
        # Use class probabilities (excluding no-object) as weights
        class_probs = F.softmax(class_logits, dim=-1)[:, :, :-1]  # (B, Q, C) remove no-obj
        mask_probs = mask_logits.sigmoid()  # (B, Q, H, W)

        # Weighted sum: for each class, sum mask_probs weighted by class probability
        # class_probs: (B, Q, C) -> (B, C, Q)
        # mask_probs:  (B, Q, H, W) -> (B, Q, H*W)
        semantic = torch.einsum('bqc,bqhw->bchw', class_probs, mask_probs)
        
        # Clamp to avoid sum > 1.0 from overlapping queries, which causes log(negative) -> nan
        semantic = torch.clamp(semantic, 1e-5, 1.0 - 1e-5)
        
        # Convert to logits scale (log-odds) for CrossEntropyLoss compatibility
        semantic = torch.log(semantic) - torch.log(1.0 - semantic)
        
        return semantic


# ═══════════════════════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════════════════════

def create_model(arch: str, in_channels: int, num_classes: int) -> nn.Module:
    """
    Create a model by architecture name.
    
    Args:
        arch: One of 'UNet', 'ResUNet++', 'Mask2Former'
        in_channels: Input channels (1 for grayscale)
        num_classes: Output classes (including background)
    
    Returns:
        nn.Module with forward(x) -> (B, num_classes, H, W) logits
    """
    arch_lower = arch.lower().replace(' ', '')
    
    if arch_lower in ('unet', 'tinyunet'):
        return TinyUNet(in_channels, num_classes)
    
    elif arch_lower in ('resunet++', 'resunetplusplus', 'resunet'):
        return ResUNetPlusPlus(in_channels, num_classes)
    
    elif arch_lower in ('mask2former',):
        return Mask2Former(in_channels, num_classes)
    
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from: UNet, ResUNet++, Mask2Former")
