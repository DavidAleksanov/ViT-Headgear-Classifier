import torch
import torch.nn as nn

class PatchEmbedder(nn.Module):
    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class LinearProjection(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedder = PatchEmbedder(patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedder(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embeddings
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim: int = 768, qkv_dim: int = 64, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.qkv_dim = qkv_dim
        self.scale = qkv_dim ** -0.5
        self.dropout = nn.Dropout(dropout_rate)
        self.wq = nn.Linear(embed_dim, qkv_dim)
        self.wk = nn.Linear(embed_dim, qkv_dim)
        self.wv = nn.Linear(embed_dim, qkv_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)
        x = attn_weights @ v
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads: int = 12, embed_dim: int = 768, qkv_dim: int = 64, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.qkv_dim = qkv_dim
        self.attn_heads = nn.ModuleList([ScaledDotProductAttention(embed_dim, qkv_dim, dropout_rate) for _ in range(n_heads)])
        self.projection = nn.Sequential(nn.Linear(n_heads * qkv_dim, embed_dim), nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_outputs = [attn(x) for attn in self.attn_heads]
        attn_outputs = torch.cat(attn_outputs, dim=-1)
        x = self.projection(attn_outputs)
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim: int = 768, mlp_hidden_size: int = 3072, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_hidden_size)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(mlp_hidden_size, embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 768, qkv_dim: int = 64, mlp_hidden_size: int = 3072, n_heads: int = 12, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mhsa = MultiHeadSelfAttention(n_heads, embed_dim, qkv_dim, dropout_rate)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_size, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mhsa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ViT(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768, qkv_dim: int = 64, mlp_hidden_size: int = 3072, n_layers: int = 12, n_heads: int = 12, n_classes: int = 10) -> None:
        super().__init__()
        self.linear_proj = LinearProjection(image_size, patch_size, in_channels, embed_dim)
        self.encoder_blocks = nn.Sequential(*[EncoderBlock(embed_dim, qkv_dim, mlp_hidden_size, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_proj(x)
        x = self.encoder_blocks(x)
        x = self.ln(x[:, 0])
        x = self.fc(x)
        return x
