import torch
import torch.nn as nn
from torchprofile import profile_macs
import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(42)


# Hybrid Sparse Attention Mechanism
class SparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scaling = embed_dim ** -0.5
        self.global_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value, mask=None):
        # Compute sparse attention
        scores = torch.einsum("bqd,bkd->bqk", query, key) * self.scaling
        batch_size, seq_len, _ = query.size()
        sparse_mask = torch.zeros_like(scores)
        window_size = 10
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            sparse_mask[:, i, start:end] = 1
        scores = scores * sparse_mask
        sparse_attention_weights = torch.softmax(scores, dim=-1)
        sparse_output = torch.einsum("bqk,bkd->bqd", sparse_attention_weights, value)

        # Add global attention
        global_output, _ = self.global_attention(query, key, value)
        return sparse_output + 0.1 * global_output


# Baseline Vision-Language Model
class BaselineVLM(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=10):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.image_encoder = nn.Linear(768, embed_dim)
        self.text_encoder = nn.Embedding(10000, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True), num_layers
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, image_tokens, text_tokens):
        image_features = self.layer_norm(self.image_encoder(image_tokens))
        text_features = self.layer_norm(self.text_encoder(text_tokens))
        repeat_factor = image_features.size(1) // text_features.size(1)
        text_features = text_features.repeat(1, repeat_factor, 1)
        joint_features = torch.cat([image_features, text_features], dim=1)
        output = self.transformer_encoder(joint_features)
        cls_token = output[:, 0]
        return self.classifier(cls_token)


# Optimized Vision-Language Model
class OptimizedVLM(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=10):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.image_encoder = nn.Linear(768, embed_dim)
        self.text_encoder = nn.Embedding(10000, embed_dim)
        self.sparse_attention = SparseAttention(embed_dim, num_heads)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, image_tokens, text_tokens):
        image_features = self.layer_norm(self.image_encoder(image_tokens))
        text_features = self.layer_norm(self.text_encoder(text_tokens))
        repeat_factor = image_features.size(1) // text_features.size(1)
        text_features = text_features.repeat(1, repeat_factor, 1)
        joint_features = torch.cat([image_features, text_features], dim=1)
        sparse_output = self.sparse_attention(joint_features, joint_features, joint_features)
        sparse_output += joint_features  # Residual connection
        cls_token = sparse_output[:, 0]
        return self.classifier(cls_token)


# Function to Compute Alignment Loss
def compute_alignment_loss(baseline_model, optimized_model, image_tokens, text_tokens):
    baseline_output = baseline_model(image_tokens, text_tokens)
    optimized_output = optimized_model(image_tokens, text_tokens)
    return nn.functional.mse_loss(baseline_output, optimized_output)


# Training Loop with Alignment Loss
def train_with_alignment(baseline_model, optimized_model, optimizer, image_tokens, text_tokens):
    optimizer.zero_grad()
    alignment_loss = compute_alignment_loss(baseline_model, optimized_model, image_tokens, text_tokens)
    alignment_loss.backward()
    optimizer.step()
    return alignment_loss.item()

# Simulated Inputs
image_tokens = torch.randn(4, 64, 768)  
text_tokens = torch.randint(0, 10000, (4, 32)) 

# Instantiate Models
baseline_model = BaselineVLM()
optimized_model = OptimizedVLM()

optimizer = torch.optim.Adam(optimized_model.parameters(), lr=1e-3)

print("Training Optimized Model with Alignment Loss")
for epoch in range(10):
    alignment_loss = train_with_alignment(baseline_model, optimized_model, optimizer, image_tokens, text_tokens)
    print(f"Epoch {epoch + 1}, Alignment Loss: {alignment_loss:.4f}")

baseline_macs = profile_macs(baseline_model, (image_tokens, text_tokens))
print(f"Baseline FLOPs: {baseline_macs / 1e9:.2f} GFLOPs")

optimized_macs = profile_macs(optimized_model, (image_tokens, text_tokens))
print(f"Optimized FLOPs: {optimized_macs / 1e9:.2f} GFLOPs")

# Measure Lossiness
lossiness = compute_alignment_loss(baseline_model, optimized_model, image_tokens, text_tokens).item()
print(f"Lossiness (MSE): {lossiness:.4f}")