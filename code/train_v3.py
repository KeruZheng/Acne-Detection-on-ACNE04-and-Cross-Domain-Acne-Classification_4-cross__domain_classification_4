import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import random
import torchvision.transforms.functional as TF
from collections import Counter

# === 自定义 Cutout ===
class RandomCutout:
    def __init__(self, num_holes=1, size=30):
        self.num_holes = num_holes
        self.size = size

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        for _ in range(self.num_holes):
            y = random.randint(0, h - self.size)
            x = random.randint(0, w - self.size)
            img[:, y:y+self.size, x:x+self.size] = 0.0
        return img

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")

    # === v3 数据增强：HSV扰动 + 仿射几何 + Cutout ===
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        RandomCutout(num_holes=1, size=30),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # === 计算类别权重 ===
    targets = [label for _, label in dataset.imgs]
    counts = Counter(targets)
    total = counts[0] + counts[1]
    class_weights = [total / counts[i] for i in range(2)]
    class_weights = torch.tensor(class_weights).float().to(device)

    # === Facenet 模型，多卡支持 ===
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=2)
    model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    log_history = []
    cam = GradCAM(model=model, target_layers=[model.module.block8.branch1[-1]])

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": np.mean(train_losses)})

        # === 验证 ===
        model.eval()
        val_labels, val_preds, val_probs = [], [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds)
                val_probs.extend(probs)

        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds)
        rec = recall_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds)
        try:
            auc = roc_auc_score(val_labels, val_probs)
        except:
            auc = float('nan')

        print(f"\n✅ Epoch {epoch+1} Validation:")
        print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUROC: {auc:.4f}")

        log_history.append({"epoch": epoch+1, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc})
        with open(os.path.join(args.output_dir, 'log.json'), 'w') as f:
            json.dump(log_history, f, indent=4)

        # === Grad-CAM 可视化 ===
        sample_image, _ = val_dataset[0]
        input_tensor = sample_image.unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])[0, :]
        rgb_image = sample_image.permute(1, 2, 0).numpy()
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        plt.imsave(os.path.join(args.output_dir, f'gradcam_epoch{epoch+1}.png'), visualization)

        # === 保存最佳模型 ===
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, args.save))
            print(f"🌟 Best model saved at epoch {epoch+1} with F1: {f1:.4f}")

    print("\n🎉 Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/v3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save", type=str, default="best_facenet_v3.pt")
    args = parser.parse_args()
    main(args)
