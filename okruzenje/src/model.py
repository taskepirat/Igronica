from pathlib import Path
import torch
import torch.nn as nn

from config import CHECKPOINT_PATH


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 4, emb_dim: int = 64, img_size: int = 28):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class ViTInputEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 4, emb_dim: int = 64, img_size: int = 28):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        return x


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim: int = 64, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed_x = self.norm1(x)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 4,
        emb_dim: int = 64,
        img_size: int = 28,
        num_classes: int = 62,
        depth: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = ViTInputEmbedding(
            in_channels=in_channels, patch_size=patch_size, emb_dim=emb_dim, img_size=img_size
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim=emb_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits


def load_trained_vit(
    device: torch.device, checkpoint_path: Path | str = CHECKPOINT_PATH
) -> VisionTransformer:
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint nije pronađen na putanji: {ckpt_file}")

    model = VisionTransformer(
        in_channels=1,
        patch_size=4,
        emb_dim=64,
        img_size=28,
        num_classes=62,
        depth=4,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model