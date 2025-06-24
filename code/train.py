import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models


def build_model(name):
    if name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr
    save_model_path = os.path.join(args.output_dir, args.save)
    data_dir = args.data_dir
    model_name = args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    log_history = []
    best_f1 = 0.0

    cam = None  # GradCAM 初始化

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': np.mean(train_losses)})

        model.eval()
        val_labels, val_preds, val_probs = [], [], []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
            for images, labels in pbar:
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
            auc = np.nan

        print(f"\n✅ Epoch {epoch+1} Validation:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUROC: {auc:.4f}")

        # === 保存 log.json ===
        log_history.append({
            'epoch': epoch + 1,
            'acc': acc,
            'prec': prec,
            'rec': rec,
            'f1': f1,
            'auc': auc
        })
        with open(os.path.join(args.output_dir, 'log.json'), 'w') as f:
            json.dump(log_history, f, indent=4)


        # === 更新 metrics_curve.png ===
        epochs = [x['epoch'] for x in log_history]
        accs = [x['acc'] for x in log_history]
        precs = [x['prec'] for x in log_history]
        recs = [x['rec'] for x in log_history]
        f1s = [x['f1'] for x in log_history]
        aucs = [x['auc'] for x in log_history]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, accs, label='Accuracy')
        plt.plot(epochs, precs, label='Precision')
        plt.plot(epochs, recs, label='Recall')
        plt.plot(epochs, f1s, label='F1 Score')
        plt.plot(epochs, aucs, label='AUROC')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Validation Metrics Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'metrics_curve.png'))
        plt.close()
        print('📊 metrics_curve.png 已更新！')

        # === Grad-CAM ===
        if cam is None:
            target_layer = model.layer4[-1] if 'resnet' in model_name else model.features[-1]
            cam = GradCAM(model=model, target_layers=[target_layer])

        sample_image, _ = val_dataset[0]
        input_tensor = sample_image.unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        rgb_image = sample_image.permute(1, 2, 0).numpy()
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        # plt.imsave(f'gradcam_epoch{epoch+1}.png', visualization)
        plt.imsave(os.path.join(args.output_dir, f'gradcam_epoch{epoch+1}.png'), visualization)

        print(f'🔥 Grad-CAM 已保存 gradcam_epoch{epoch+1}.png')

        # === 保存 best 模型 ===
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_model_path)
            print(f"🌟 New best F1: {f1:.4f} — Model saved to {save_model_path}!")

    print(f"\n🎉 Training finished. Best model saved as: {save_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary skin patch classifier with real-time logging and Grad-CAM.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save all outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save", type=str, default="best_model.pt")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet18", "resnet50", "efficientnet_b0"])
    args = parser.parse_args()
    main(args)
