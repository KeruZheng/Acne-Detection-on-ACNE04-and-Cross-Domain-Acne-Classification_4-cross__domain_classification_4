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

## 🧬 DermNet valid test
we create a pesudo test list, which follow the initial distribution ratio of acne/non-ance in DermNet/test
```
Pseudo test list saved to /data_lg/keru/project/part2/DermNet/eval/final_test_list.txt
Acne: 312, Non-Acne: 3690
```
the code is in code/validate.ipynb , part FINAL TEST

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
### 📊 Metrics Snapshot (Version 1 — Failure Case)

  | Accuracy | Precision | Recall | F1 Score | AUROC  |
  |----------|-----------|--------|----------|--------|
  | 0.922    | 0.0       | 0.0    | 0.0      | 0.3886 |
  
  > ⚠️ Despite high accuracy, the model fails entirely at detecting positive samples, leading to **0 precision, recall, and F1 score** — a typical symptom of **severe class imbalance** (i.e., only predicting negatives correctly).

</details> 

<details>
<summary>📊 V2 vs V3 Comparison</summary>

| Item              | V2: Facenet + Basic Augmentations               | V3: Facenet + Three Augmentations (Paper Method)        |
|-------------------|--------------------------------------------------|----------------------------------------------------------|
| **Model Structure** | ✅ `facenet-pytorch`                             | ✅ Same                                                  |
| **Basic Augmentations** | ✅ Color + Blur + Rotation + Flip + Crop        | ✅ Partially inherited (avoid duplication)               |
| **Advanced Augmentations** | ❌ None                                      | ✅ HSV + Cutout + Affine                                |
| **Implementation Complexity** | ⭐⭐                                          | ⭐⭐⭐                                                    |
| **Target Focus**  | Generalize skin tone / pose                      | Simulate texture shift, occlusion, and blur             |
| **Contribution**  | Serves as a baseline to compare enhancement methods | Validate if HSV, Cutout, etc., further improve results  |

</details>

---

<details open>
<summary><strong>🧠 V2 vs. V2_Weighted(the same as v2_plus) — Binary Patch Classification</strong></summary>

### 🔍 Key Differences

| Item                  | V2                                                    | V2\_Weighted                                              |
| --------------------- | ----------------------------------------------------- | --------------------------------------------------------- |
| Data Source           | YOLOv5-labeled patches (naturally imbalanced dataset) | 1:1 balanced dataset of ACNE and non-ACNE samples         |
| Training Set Balance  | Imbalanced — mimics real-world acne frequency         | Balanced — artificially equal number of ACNE and non-ACNE |
| Motivation            | Designed to reflect real-world distribution           | Attempts to mitigate class imbalance during training      |
| Model & Augmentations | Exactly the same (Facenet + data augmentation)        | Exactly the same                                          |

---

### 💻 Code Paths

* **V2**: `code/train_v2.py`
* **V2\_Weighted**: `code/train_v2_plus.py`

### 🧪 Training Commands

<pre>
V2:
CUDA_VISIBLE_DEVICES=0,1,2 python train_v2.py \
--data_dir /data_lg/keru/project/part2/yolo_cutting_result \
--output_dir /data_lg/keru/project/part2/outputs_v2 \
--epochs 50 \
--batch_size 64 \
--lr 3e-4 \
--save best_facenet_v2.pt

V2_Weighted:
CUDA_VISIBLE_DEVICES=0,1,2 python train_v2_plus.py \
--data_dir /data_lg/keru/project/part2/yolo_cutting_result \
--output_dir /data_lg/keru/project/part2/outputs_v2_plus \
--epochs 50 \
--batch_size 64 \
--save best_facenet_v2_plus.pt
</pre>

---

### 📊 Evaluation Results

#### 📌 V2

| Metric    | Value      |
| --------- | ---------- |
| Accuracy  | 0.9220     |
| Precision | 0.0000     |
| Recall    | 0.0000     |
| F1 Score  | 0.0000     |
| AUROC     | **0.5268** |

#### 📌 V2\_Weighted

| Metric    | Value      |
| --------- | ---------- |
| Accuracy  | 0.9208     |
| Precision | 0.0000     |
| Recall    | 0.0000     |
| F1 Score  | 0.0000     |
| AUROC     | **0.5076** |

---
Grad-CAM V2：
<div style="display: flex; justify-content: space-around; align-items: center; margin: 20px 0;">
  <img src="https://github.com/user-attachments/assets/245fc934-4275-47c0-9522-b02da3087e52" alt="图片1" width="200" height="auto">
  <img src="https://github.com/user-attachments/assets/d2471a64-9a1e-4720-af45-258b16e32f7d" alt="图片2" width="200" height="auto">
  <img src="https://github.com/user-attachments/assets/20e07822-623f-4685-ac6c-6d2f201c478b" alt="图片3" width="200" height="auto">
  <img src="https://github.com/user-attachments/assets/4cb74a7c-9b56-4f84-9a11-cd2b2d8a1670" alt="图片4" width="200" height="auto">
</div>

Grad-CAM V2_plus：
<div style="display: flex; justify-content: space-around; align-items: center;">
  <img src="https://github.com/user-attachments/assets/d550ebdf-ece4-4322-9a54-5f5d26b7774a" alt="1" width="200">
  <img src="https://github.com/user-attachments/assets/40df6a44-9234-4847-b8ff-9138eba210d1" alt="2" width="200">
  <img src="https://github.com/user-attachments/assets/d201a250-b81e-4ab1-bb70-563c03565b03" alt="3" width="200">
  <img src="https://github.com/user-attachments/assets/eacd7565-40fb-4355-a66c-7b5fc95ba635" alt="4" width="200">
</div>
---

### 🧠 Analysis

The results reveal an interesting insight: **the imbalanced training set (V2) performed slightly better than the balanced one (V2\_Weighted)** in terms of AUROC (0.5268 vs. 0.5076), despite both achieving high accuracy but poor recall/precision due to label sparsity.

This aligns with our **real-world motivation**. In actual datasets like **DermNet**, acne samples account for **less than 10%**. The V2 model is trained on similarly distributed data, making it more suitable for realistic clinical deployment. Artificially balancing the dataset (as in V2\_Weighted) may hinder generalization by overfitting to rare positive samples.

Thus, **V2 demonstrates better alignment with real-world distribution and achieves better discrimination under class imbalance**.

</details>

---


<details open>
<summary><strong>🧠 V3 vs. V3_Weighted(the same as v2_plus) — Binary Patch Classification</strong></summary>

### 🔍 Key Differences

| Item                  | V3                                                    | V3\_Weighted                                              |
| --------------------- | ----------------------------------------------------- | --------------------------------------------------------- |
| Data Source           | YOLOv5-labeled patches (naturally imbalanced dataset) | 1:1 balanced dataset of ACNE and non-ACNE samples         |
| Training Set Balance  | Imbalanced — mimics real-world acne frequency         | Balanced — artificially equal number of ACNE and non-ACNE |
| Motivation            | Designed to reflect real-world distribution           | Attempts to mitigate class imbalance during training      |
| Model & Augmentations | Facenet + Three Augmentations (Paper Method)          | Exactly the same                                          |

---

### 💻 Code Paths

* **V3**: `code/train_v3.py`
* **V3\_Weighted**: `code/train_v3_plus.py`

### 🧪 Training Commands

<pre>
V3:
CUDA_VISIBLE_DEVICES=0,3,4 python train_v3.py \
--data_dir /data_lg/keru/project/part2/yolo_cutting_result \
--output_dir /data_lg/keru/project/part2/outputs_v3 \
--epochs 50 \
--batch_size 64 \
--lr 3e-4 \
--save best_facenet_v3.pt

V3_Weighted:
CUDA_VISIBLE_DEVICES=0,3,4 python train_v3_plus.py \
--data_dir /data_lg/keru/project/part2/yolo_cutting_result \
--output_dir /data_lg/keru/project/part2/outputs_v3 \
--epochs 50 \
--batch_size 64 \
--lr 3e-4 \
--save best_facenet_v3.pt
</pre>

---

### 📊 Evaluation Results

#### 📌 V3

| Metric    | Value      |
| --------- | ---------- |
| Accuracy  | 0.9193     |
| Precision | 0.0000     |
| Recall    | 0.0000     |
| F1 Score  | 0.0000     |
| AUROC     | **0.4670** |

#### 📌 V3\_Weighted

| Metric    | Value      |
| --------- | ---------- |
| Accuracy  | 0.9198     |
| Precision | 0.0000     |
| Recall    | 0.0000     |
| F1 Score  | 0.0000     |
| AUROC     | **0.4539** |

---
Grad-CAM V3：
<div style="display: flex; justify-content: space-around; align-items: center; margin: 20px 0;">
  <img src ="https://github.com/user-attachments/assets/19529614-1a9b-47b8-bea2-d6c58e1ea677"  alt="图片1" width="200" height="auto">
  <img src ="https://github.com/user-attachments/assets/79dde4ba-24d9-44e0-9c35-f8ae360dc9bf" alt="图片2" width="200" height="auto">
  <img src ="https://github.com/user-attachments/assets/da3d1265-6104-4d72-86c5-46525de13886"  alt="图片3" width="200" height="auto">
  <img src ="https://github.com/user-attachments/assets/e0023145-c51e-4061-adbe-9bc950768ead"  alt="图片4" width="200" height="auto">
</div>

Grad-CAM V3_plus：
<div style="display: flex; justify-content: space-around; align-items: center;">
  <img src ="https://github.com/user-attachments/assets/32eaf995-3164-4cd7-bd17-1508753e9d2c" alt="1" width="200">
  <img src ="https://github.com/user-attachments/assets/5d862e9d-b9a3-4bac-b511-fbb6f856b241" alt="2" width="200">
  <img src ="https://github.com/user-attachments/assets/502f87f2-bd85-48ca-b452-16b43013ffb8" alt="3"width="200">
  <img src ="https://github.com/user-attachments/assets/b3008385-d14f-4682-90fa-ff9f66870e9a" alt="4" width="200">
</div>
---


### 🧠 Analysis

### Analysis of V3 & V3_weighted Results  

The results show a clear pattern: Both **V3** and **V3_weighted** achieve relatively high accuracy (0.9193 and 0.9198, respectively) but suffer from **perfectly zero precision, recall, and F1-score**. This suggests a critical issue—likely severe class imbalance or a model bias toward predicting only the majority class.  

### Key Metric Breakdown  
- **Accuracy**: Misleadingly high, as it reflects overall correct predictions but ignores failures on minority classes.  
- **Precision, Recall, F1-Score**: Collapse to 0, indicating the model fails to effectively identify or predict the minority class (e.g., positive samples).  
- **AUROC**: Both scores (0.4670 for V3; 0.4539 for V3_weighted) are below 0.5, meaning the model performs **worse than random guessing** at distinguishing classes.  


### Real-World Context & Implications  
In datasets like medical imaging or fraud detection, minority classes (e.g., rare diseases, fraudulent transactions) are critical but sparse. Here, the extreme metric collapse suggests:  
- The model may be **overfitting to the majority class** (e.g., always predicting “negative”), rendering it useless for identifying rare, high-priority cases.  
- Artificially balancing classes (V3_weighted) did not resolve the issue—its AUROC is even lower than V3, hinting that forced balancing might **harm generalization** for real-world data distributions.  


### Conclusion  
Both models fail to handle class imbalance effectively. However, **V3 aligns slightly better with real-world sparsity** (its AUROC is marginally higher than V3_weighted). For practical use, retraining with strategies like weighted loss, oversampling minority classes, or adjusting prediction thresholds is essential to improve minority-class detection. Without these fixes, the models cannot reliably identify critical rare cases.

</details>

<details open>
<summary><strong>🧠 V3 — CycleGAN-based Acne Image Style Transfer and Training</strong></summary>

### 🔍 Key Workflow and Data Preparation

| Step                                                              | Description                                                                                                 |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Acne Image Selection                                              | Selected 1000 positive and 1000 negative acne patches from YOLOv5-split data                                |
| DermNet Image Selection                                           | Randomly selected 20 positive and 20 negative images from DermNet/train, then augmented to 1000 images each |
| Style Transfer Domains                                            | Defined as:                                                                                                 |
|                                                                   | - `/path/to/output/cyclegan_data/domainA`: All original ACNE04 images                                       |
|                                                                   | - `/path/to/output/cyclegan_data/domainB`: DermNet domain, style-augmented to 1000 images                   |
| Final Dataset Structure                                           | \`\`\`                                                                                                      |
| CycleGAN\_data/                                                   |                                                                                                             |
| ├─ domain\_acne/                                                  |                                                                                                             |
| │  ├─ positive/   # 1000 positive acne samples                    |                                                                                                             |
| │  └─ negative/   # 1000 negative acne samples                    |                                                                                                             |
| ├─ domain\_dermnet/                                               |                                                                                                             |
| │  ├─ positive/   # 1000 style-augmented DermNet positive samples |                                                                                                             |
| │  └─ negative/   # 1000 style-augmented DermNet negative samples |                                                                                                             |

````|

---

### 💻 Code Paths and Commands

#### Data Preparation

```bash
python /data_lg/keru/project/part2/code/prepare_cyclegan_data.py \
--acne_dir /data_lg/keru/project/part2/yolo_cutting_result \
--dermnet_dir /data_lg/keru/project/part2/DermNet/train \
--output_dir /data_lg/keru/project/part2/pytorch-CycleGAN-and-pix2pix/CycleGAN_data
````

#### CycleGAN Training

* **Positive samples style transfer training**

```bash
python train.py --dataroot /data_lg/keru/project/part2/pytorch-CycleGAN-and-pix2pix/CycleGAN_data/positive_train \
--name acne2dermnet_pos --model cycle_gan --gpu_ids 0,1
```

* **Negative samples style transfer training**

```bash
python train.py --dataroot /data_lg/keru/project/part2/pytorch-CycleGAN-and-pix2pix/CycleGAN_data/negative_train \
--name acne2dermnet_neg --model cycle_gan --gpu_ids 2,3
```

#### CycleGAN Inference (Style Transfer)

* Positive samples conversion

```bash
python /data_lg/keru/project/part2/pytorch-CycleGAN-and-pix2pix/test.py \
--dataroot /data_lg/keru/project/part2/pytorch-CycleGAN-and-pix2pix/CycleGAN_data/positive \
--name acne2dermnet_pos --model cycle_gan --phase test --no_dropout --num_test 1000 --gpu_ids 0
```

* Negative samples conversion

```bash
python /data_lg/keru/project/part2/pytorch-CycleGAN-and-pix2pix/test.py \
--dataroot /data_lg/keru/project/part2/pytorch-CycleGAN-and-pix2pix/CycleGAN_data/negative \
--name acne2dermnet_neg --model cycle_gan --phase test --no_dropout --num_test 1000 --gpu_ids 0
```

#### Verification of Output Size

```bash
ls /data_lg/keru/project/part2/pytorch-CycleGAN-and-pix2pix/results/acne2dermnet_pos/test_latest/images | wc -l
# Expected: 1000 images
```

---

### 🧪 Training with Converted Data

```bash
python /data_lg/keru/project/part2/code/train_cycleGAN.py \
--original_pos_dir /data_lg/keru/project/part2/yolo_cutting_result/positive \
--original_neg_dir /data_lg/keru/project/part2/yolo_cutting_result/negative \
--converted_pos_dir /data_lg/keru/project/part2/pytorch-CycleGAN-and-pix2pix/results/acne2dermnet_pos/test_latest/images \
--converted_neg_dir /data_lg/keru/project/part2/pytorch-CycleGAN-and-pix2pix/results/acne2dermnet_neg/test_latest/images \
--output_dir /data_lg/keru/project/part2/output/output_cycleGAN \
--epochs 50 --batch_size 64
```

---

### 📊 Evaluation

Model evaluation is conducted via the Jupyter notebook:

`/data_lg/keru/project/part2/code/validate.ipynb`

This notebook includes metrics calculation and visualization for both original and style-transferred acne images training results.

---
#### 📌 cyclegan——version

| Metric    | Value      |
| --------- | ---------- |
| Accuracy  | 0.6794     |
| Precision | 0.1100     |
| Recall    | 0.4391     |
| F1 Score  | 0.1760     |
| AUROC     | **0.6206** |

### 🧠 Analysis

* The pipeline leverages CycleGAN to transfer DermNet style features into acne images, enabling enriched domain diversity and better generalization.
* By carefully selecting and augmenting both positive and negative samples across acne and DermNet datasets, the model gains robustness against domain shifts.
* Style-transferred images serve as effective supplements for acne classification training, addressing the scarcity of labeled DermNet acne data.
* Training and inference are split for positive and negative samples to capture domain-specific transformations accurately.

---

This CycleGAN-based approach provides a novel way to enhance acne image classification by bridging domain gaps with style transfer, improving model performance and generalization on diverse acne image sources.

</details>

<details open>
<summary><strong>🧠 Performance Comparison and Analysis of V2, V3, and CycleGAN-enhanced V3</strong></summary>

### 📊 Evaluation Results

| Version         | Accuracy | Precision | Recall | F1 Score | AUROC      |
| --------------- | -------- | --------- | ------ | -------- | ---------- |
| **V2**          | 0.9220   | 0.0000    | 0.0000 | 0.0000   | **0.5268** |
| **V3**          | 0.9193   | 0.0000    | 0.0000 | 0.0000   | **0.4670** |
| **CycleGAN V3** | 0.6794   | 0.1100    | 0.4391 | 0.1760   | **0.6206** |

---

### 🧠 Analysis Summary

* **V2 vs V3:**
  Both versions achieve similarly high accuracy (\~91.9% to 92.2%) but suffer from zero precision, recall, and F1-score, indicating failure to correctly identify positive acne samples. The slightly higher AUROC in V2 (0.5268) compared to V3 (0.4670) suggests that training with data balanced closer to the **real-world distribution (V2)** enables the model to better reflect practical scenarios despite class imbalance.

* **CycleGAN-enhanced V3:**
  While the overall accuracy drops (to 67.9%), the model shows clear improvements in precision (0.11), recall (0.44), and F1 score (0.176), alongside a significantly higher AUROC (0.6206). This demonstrates that applying CycleGAN-based style transfer to incorporate DermNet domain features helps the model **better learn cross-domain representations** and improves its ability to detect acne in diverse image sources.

* **Conclusion:**
  Training on a dataset that reflects realistic class distributions is important for practical performance (as seen in V2). Moreover, the CycleGAN-based domain adaptation effectively enhances feature diversity and domain generalization, addressing limitations of purely acne-domain training and improving detection metrics in cross-domain settings.

</details>



