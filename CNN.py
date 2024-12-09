import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import warnings
warnings.filterwarnings("ignore")


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


# ResNet18 Model for MHIST Classification
class ResNet18MHISTModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18MHISTModel, self).__init__()
        # Load pre-trained ResNet18
        self.resnet18 = models.resnet18(pretrained=True)
        # Replace the final fully connected layer to match the number of classes
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)


# Training and Testing Functions
def train_and_test_resnet18():
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
    model = ResNet18MHISTModel(num_classes=2).to(device)
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
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    print("Training complete.")

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

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1])

    print("\nTesting Results:")
    print(f"SSA - Precision : {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1-Score: {f1[0]:.4f}")
    print(f"HP - Precision : {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1-Score: {f1[1]:.4f}")
    print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds):.4f}")


if __name__ == "__main__":
    train_and_test_resnet18()
