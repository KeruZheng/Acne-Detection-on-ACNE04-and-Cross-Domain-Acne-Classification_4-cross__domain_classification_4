import os
import random
import shutil
from PIL import Image, ImageEnhance, ImageOps
import argparse

def augment_image(img):
    # 简单扩增：随机裁剪、翻转、亮度/对比度调整
    # 输入PIL Image，返回增强后的PIL Image
    
    # 随机裁剪
    w, h = img.size
    crop_ratio = random.uniform(0.8, 1.0)
    new_w, new_h = int(w*crop_ratio), int(h*crop_ratio)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize((w, h))

    # 随机水平翻转
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    # 随机垂直翻转
    if random.random() < 0.5:
        img = ImageOps.flip(img)

    # 随机亮度调整
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    # 随机对比度调整
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    return img

def sample_and_augment(src_dirs, sample_num, augment_num, save_dir, resize_size):
    """
    src_dirs: list of folders to sample from
    sample_num: how many original images to sample
    augment_num: how many total images after augmentation
    save_dir: save path
    resize_size: (w,h)
    """

    os.makedirs(save_dir, exist_ok=True)

    # 收集所有图片路径
    all_imgs = []
    for d in src_dirs:
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_imgs.append(os.path.join(d, f))

    print(f"Found {len(all_imgs)} images in {src_dirs}")

    # 随机采样sample_num张
    sampled_imgs = random.sample(all_imgs, min(sample_num, len(all_imgs)))
    print(f"Sampled {len(sampled_imgs)} images from {src_dirs}")

    # 保存原始采样图
    for i, img_path in enumerate(sampled_imgs):
        img = Image.open(img_path).convert('RGB').resize(resize_size)
        img.save(os.path.join(save_dir, f"orig_{i}.jpg"))

    # 计算还需扩增的图片数
    augment_needed = augment_num - len(sampled_imgs)
    print(f"Need to augment {augment_needed} images to reach {augment_num}")

    # 轮流对原始图片做扩增直到达到augment_num
    idx = 0
    for i in range(augment_needed):
        img_path = sampled_imgs[idx % len(sampled_imgs)]
        img = Image.open(img_path).convert('RGB').resize(resize_size)
        img_aug = augment_image(img)
        img_aug.save(os.path.join(save_dir, f"aug_{len(sampled_imgs) + i}.jpg"))
        idx += 1

def main(args):
    random.seed(42)

    # ACNE路径假设正负样本文件夹结构：
    # args.acne_dir/positive/
    # args.acne_dir/negative/
    acne_pos_dir = os.path.join(args.acne_dir, 'positive')
    acne_neg_dir = os.path.join(args.acne_dir, 'negative')

    # DermNet正样本路径 (只用Acne and Rosacea Photos文件夹)
    dermnet_pos_dir = [os.path.join(args.dermnet_dir, 'Acne and Rosacea Photos')]

    # DermNet负样本路径 (train目录下除'Acne and Rosacea Photos'外其他所有子文件夹合并)
    dermnet_neg_dirs = []
    for d in os.listdir(args.dermnet_dir):
        if d != 'Acne and Rosacea Photos':
            dermnet_neg_dirs.append(os.path.join(args.dermnet_dir, d))

    # 输出目录结构
    domain_acne_pos = os.path.join(args.output_dir, 'domain_acne', 'positive')
    domain_acne_neg = os.path.join(args.output_dir, 'domain_acne', 'negative')
    domain_dermnet_pos = os.path.join(args.output_dir, 'domain_dermnet', 'positive')
    domain_dermnet_neg = os.path.join(args.output_dir, 'domain_dermnet', 'negative')

    # 1. 采样和保存ACNE正负样本各1000张（随机采样）
    sample_and_augment([acne_pos_dir], sample_num=1000, augment_num=1000, save_dir=domain_acne_pos, resize_size=(256,256))
    sample_and_augment([acne_neg_dir], sample_num=1000, augment_num=1000, save_dir=domain_acne_neg, resize_size=(256,256))

    # 2. 采样DermNet正负样本各10张，扩增到1000张
    sample_and_augment(dermnet_pos_dir, sample_num=10, augment_num=1000, save_dir=domain_dermnet_pos, resize_size=(256,256))
    sample_and_augment(dermnet_neg_dirs, sample_num=10, augment_num=1000, save_dir=domain_dermnet_neg, resize_size=(256,256))

    print("✅ Data preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acne_dir", type=str, required=True, help="Path to ACNE dataset folder with 'positive' and 'negative' subfolders")
    parser.add_argument("--dermnet_dir", type=str, required=True, help="Path to DermNet train folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for CycleGAN data preparation")
    args = parser.parse_args()
    main(args)
