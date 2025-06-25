## 🧠 Our Model: Acne Classification Using Cropped Patches from YOLOv5-ACNE04

### 📚 Dataset: ACNE04 (YOLOv5 version)

We use the **YOLOv5-compatible ACNE04 dataset**, where each facial image includes bounding box annotations for acne lesions.
The dataset is publicly available at:
🔗 [https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection/](https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection/)

### ✂️ Patch-based Strategy: Positive and Negative Sampling

Complete Positive and Negative Sampling code is in 
```
code/patches_detect.ipynb
```

To maximize annotation accuracy and train a robust classifier, we adopt a **positive-negative cropping strategy** on each image:

#### ✅ Positive Patches

* Directly cropped from acne bounding boxes.
* Provide highly accurate acne samples.

#### ❌ Negative Patches

* Sampled randomly from the **same image**, ensuring no overlap with acne boxes.
* The number of negative patches is **2\~3× more** than positives.
* Sampled only from the **central region** of the face:
  `(W * 0.2, H * 0.2)` to `(W * 0.8, H * 0.8)`
  This avoids selecting irrelevant background or out-of-face areas.

### ⚙️ Design Philosophy

This patch selection strategy ensures:

* ✅ Maximal use of reliable, annotated data.
* ✅ Positive vs. negative patches differ **only** by acne presence.
* ✅ All patches share lighting, skin tone, camera setting, and pose — reducing domain noise.
* ✅ Helps the model focus on **skin texture**, **color difference**, and **acne inflammation details**.



---

## 🎯 Enhancing Skin Awareness: Color & Texture

### (1) Color Contrast Awareness

Acne is often distinguished by **redness, discoloration, or pigmentation**.
We apply `ColorJitter` during data augmentation to make the model sensitive to these features.

```
python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
```

### (2) Texture Feature Enhancement

Acne lesions differ in surface texture (bumps, edges).
To better capture these features, we use a **deeper backbone network**:

```python
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
```

Compared to ResNet18, **ResNet50** offers stronger feature extraction for:

* Local skin textures
* Edges and contours
* Robustness in cross-domain scenarios

---

## ✅ Final Model Choice

We selected **ResNet50** as our final backbone due to:

* Its superior texture/structure learning capacity.
* Better performance in cross-domain generalization (e.g., between DermNet and ACNE04).
* Ability to distinguish subtle differences between acne-affected and healthy skin.

---

## 🧬 DermNet Dataset Overview

**DermNet** is a comprehensive dermatology image database covering **hundreds of skin conditions**, categorized by disease type. For our acne classification task, we selectively filtered categories to ensure relevance and reduce confusion.
<p align="center">
  <img src="https://github.com/user-attachments/assets/d5b2f7be-e098-4c89-bdc7-8ddac048a619" width="45%" />
  <img src="https://github.com/user-attachments/assets/ddf785ad-4feb-473e-8e68-6a48bef75aa3" width="45%" />
</p>

* One particularly challenging category is:
  **`Warts Molluscum and other Viral Infections`**
  These lesions often exhibit circular or raised features that **superficially resemble acne**, leading to possible misclassification.

* In contrast, most other DermNet categories feature:

  * **Irregular edges**
  * **Large, patchy or elliptical lesions**
  * **Locations on the torso or limbs**, which differ from typical acne distribution

---

## 💡 Our Thinking Behind Filtering and Model Design

To refine the classification and reduce confusion, we considered the following domain-specific observations:

1. **Facial Localization**

   * Acne lesions **primarily occur on the face**, unlike many other skin conditions.
   * We could leverage **facial region detection** or **facial landmark cues** to guide acne-focused cropping.

2. **Texture Differences**

   * Facial skin tends to be **smoother and more uniform** than skin on limbs or the torso.
   * This texture distinction can help models discriminate acne from unrelated conditions.

3. **Color Distribution**

   * Acne lesions are typically **reddish or pink**, due to inflammation.
   * We explicitly avoid categories like **melanoma** or **dark pigmented lesions** during data selection to reduce color-based confusion.

4. **Shape and Size**

   * Acne generally does **not appear in large patches**.
   * We filter out conditions with **broad, irregular, or spreading lesions**.

---

<details>
<summary>🧠 Binary Patch Classification — Version 1</summary>

This script trains a **binary classifier** to distinguish between acne (positive) and normal (negative) skin patches using image-level labels.

### 🧩 Dataset Design

- **Input Format**: The dataset is organized in two subfolders (`class0/`, `class1/`) under the specified `--data_dir`, and is loaded using `torchvision.datasets.ImageFolder`.
- **Positive Patches**: Cropped using YOLOv5's acne bounding boxes.
- **Negative Patches**: Sampled randomly from the same face image, ensuring:
  - The sampled region **does not overlap** with any acne bounding box.
  - Region lies **within the central area** of the image (between 20%–80% in both width and height) to ensure it stays on the face.
  - This approach helps control background, lighting, and skin texture, isolating acne-specific features.

---

### 🧪 Training Logic

- **Model Choices**:
  - `resnet18`, `resnet50`, and `efficientnet_b0` are supported (we mainly used **ResNet50**).
- **Augmentations**:
  - Includes resizing, random flipping, rotation, and **ColorJitter** to improve generalization and red-lesion sensitivity.
- **Train/Validation Split**:
  - 80/20 random split is applied automatically within the script.
- **Optimizer**: Adam with learning rate from `--lr` argument.
- **Loss**: CrossEntropyLoss for binary classification.

---

### 📤 Outputs (per epoch)

- 📝 `log.json`: Stores Accuracy, Precision, Recall, F1, and AUROC for each epoch.
- 📈 `metrics_curve.png`: Validation curve of metrics over time.
- 🔥 `gradcam_epoch{n}.png`: Visual explanation of the model's attention on validation data (via Grad-CAM).
- 💾 `best_model.pt`: Automatically saved when a new best F1 score is reached.

---

### 🚀 Training Command

    ```bash
    CUDA_VISIBLE_DEVICES=1 python /data_lg/keru/project/part2/code/train.py \
      --data_dir /data_lg/keru/project/part2/yolo_cutting \
      --model resnet50 \
      --epochs 50 \
      --batch_size 64 \
      --lr 3e-4 \
      --save best_resnet50.pt
      、、、

</details> 


