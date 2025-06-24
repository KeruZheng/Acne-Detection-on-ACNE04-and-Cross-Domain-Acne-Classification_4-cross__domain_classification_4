import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from collections import Counter

class CombinedDataset(Dataset):
    def __init__(self, pos_dirs, neg_dirs, transform=None):
        """
        pos_dirs, neg_dirs: list of directory paths containing positive/negative images
        """
        self.transform = transform
        self.images = []
        self.labels = []
        # Load positive images
        for d in pos_dirs:
            for fname in os.listdir(d):
                path = os.path.join(d, fname)
                if os.path.isfile(path) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.images.append(path)
                    self.labels.append(1)
        # Load negative images
        for d in neg_dirs:
            for fname in os.listdir(d):
                path = os.path.join(d, fname)
                if os.path.isfile(path) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.images.append(path)
                    self.labels.append(0)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    pos_dirs = [args.original_pos_dir, args.converted_pos_dir]
    neg_dirs = [args.original_neg_dir, args.converted_neg_dir]

    dataset = CombinedDataset(pos_dirs, neg_dirs, transform=transform)
    print(f"Dataset size: {len(dataset)} images")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_labels = [train_dataset.dataset.labels[i] for i in train_dataset.indices]
    class_counts = Counter(train_labels)
    print(f"🧮 Class distribution in training set: {dict(class_counts)}")
    total = sum(class_counts.values())
    weights = [total / class_counts[i] for i in range(len(class_counts))]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=2)
    model = nn.DataParallel(model).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    cam = GradCAM(model=model, target_layers=[model.module.block8.branch1[-1]])
    best_f1 = 0.0
    log_history = []

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

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
        prec = precision_score(val_labels, val_preds, zero_division=0)
        rec = recall_score(val_labels, val_preds, zero_division=0)
        f1 = f1_score(val_labels, val_preds, zero_division=0)
        try:
            auc = roc_auc_score(val_labels, val_probs)
        except:
            auc = float('nan')

        print(f"\n✅ Epoch {epoch+1} Validation:")
        print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUROC: {auc:.4f}")

        log_history.append({"epoch": epoch+1, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc})
        with open(os.path.join(args.output_dir, 'log.json'), 'w') as f:
            json.dump(log_history, f, indent=4)

        # Grad-CAM
        sample_image, _ = val_dataset[0]
        input_tensor = sample_image.unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])[0, :]
        rgb_image = sample_image.permute(1, 2, 0).numpy()
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        plt.imsave(os.path.join(args.output_dir, f'gradcam_epoch{epoch+1}.png'), visualization)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, args.save))
            print(f"🌟 Best model saved at epoch {epoch+1} with F1: {f1:.4f}")

    print("\n🎉 Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_pos_dir", type=str, required=True, help="Original positive samples path")
    parser.add_argument("--original_neg_dir", type=str, required=True, help="Original negative samples path")
    parser.add_argument("--converted_pos_dir", type=str, required=True, help="CycleGAN converted positive samples path")
    parser.add_argument("--converted_neg_dir", type=str, required=True, help="CycleGAN converted negative samples path")
    parser.add_argument("--output_dir", type=str, default="outputs/v2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save", type=str, default="best_facenet_v2.pt")
    args = parser.parse_args()
    main(args)
