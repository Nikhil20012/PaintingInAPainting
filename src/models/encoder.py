from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch
import torch.nn as nn


class ViTEncoder(nn.Module):
    def __init__(self, freeze_layers: int = 6):
        super().__init__()

        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # extract internal components we need
        self.patch_embed = vit.conv_proj         # Conv2d: image → patch tokens
        self.class_token = vit.class_token       # learnable CLS token
        self.pos_embed   = vit.encoder.pos_embedding  # positional embeddings
        self.dropout     = vit.encoder.dropout
        self.layers      = vit.encoder.layers    # 12 transformer blocks
        self.norm        = vit.encoder.ln        # final LayerNorm

        self.hidden_dim  = 768   # ViT-B hidden size
        self.num_patches = 196   # 14x14 grid of 16x16 patches

        # freeze bottom N transformer blocks
        for i, layer in enumerate(self.layers):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]

        # step 1: split image into patches and embed each one
        x = self.patch_embed(x)               # (B, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2)      # (B, 196, 768)

        # step 2: prepend CLS token to the sequence
        cls = self.class_token.expand(B, -1, -1)   # (B, 1, 768)
        x   = torch.cat([cls, x], dim=1)            # (B, 197, 768)

        # step 3: add positional embeddings so model knows patch locations
        x = self.dropout(x + self.pos_embed)

        # step 4: pass through all 12 transformer blocks
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # step 5: split outputs — CLS token goes to classifier, patches to detector
        cls_token    = x[:, 0]     # (B, 768)  — global image summary
        patch_tokens = x[:, 1:]    # (B, 196, 768) — spatial features

        return cls_token, patch_tokens