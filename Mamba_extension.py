import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from einops import rearrange, repeat
import math
import warnings
warnings.filterwarnings("ignore")

# Mamba Model
class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Convolution for local feature extraction
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        # Projections for dynamics and state-space parameters
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        )

        self.A_log = nn.Parameter(A.log())
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states):
        batch, seqlen, dim = hidden_states.shape

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        x, z = xz.chunk(2, dim=1)

        x = self.act(self.conv1d(x)[..., :seqlen])

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = x
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


# MHIST Dataset
class MHISTDataset(Dataset):
    def __init__(self, annotation_file, image_folder, transform=None, split="train"):
        self.data = pd.read_csv(annotation_file)
        self.image_folder = image_folder
        self.transform = transform
        self.data = self.data[self.data["Partition"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_folder, row["Image Name"])
        image = Image.open(image_path).convert("RGB")
        label = 0 if row["Majority Vote Label"] == "SSA" else 1

        if self.transform:
            image = self.transform(image)

        return image, label


# Integrated CNN + Mamba Model
class CNNMambaModel(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, d_model=768, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.patch_size = patch_size

        # CNN Backbone
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final FC layer
        self.feature_dim = 512  # Output from ResNet18

        # Mamba Model
        self.mamba = Mamba(d_model=self.feature_dim, d_state=d_state, d_conv=d_conv, expand=expand)

        # Classification Head
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # Extract features with CNN
        x = self.cnn(x)

        # Prepare for Mamba
        x = rearrange(x, "b d -> b 1 d")  # Add sequence dimension

        # Process with Mamba
        x = self.mamba(x)

        # Classification
        x = x.mean(dim=1)  # Pooling
        return self.classifier(x)


# Training and Testing Functions
def train_and_test_cnn_mamba():
    annotation_file = "mhist_dataset/annotations.csv"
    image_folder = "mhist_dataset/mhist_images"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = MHISTDataset(annotation_file, image_folder, transform, split="train")
    test_dataset = MHISTDataset(annotation_file, image_folder, transform, split="test")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNMambaModel(img_size=128, patch_size=16, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    print("Training complete.")

    # Testing Loop
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=[0,1])

    print("\nTesting Results:")
    print(f"SSA - Precision : {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1-Score: {f1[0]:.4f}")
    print(f"HP - Precision : {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1-Score: {f1[1]:.4f}")
    print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds):.4f}")


if __name__ == "__main__":
    train_and_test_cnn_mamba()
