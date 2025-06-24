import os
import sys
import sys
import os
sys.path.append('/data_lg/keru/project/part2/pytorch-CycleGAN-and-pix2pix')


import argparse
from PIL import Image
import torch
from torchvision import transforms
from models import create_model  # 需确保官方pytorch-CycleGAN-and-pix2pix代码结构在PYTHONPATH中
from util import util

def run_inference(input_dir, output_dir, checkpoint_dir, gpu_ids=[0]):
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置参数，仿照官方 options.py
    class Args:
        def __init__(self):
            self.dataroot = input_dir
            self.name = os.path.basename(checkpoint_dir)
            self.checkpoints_dir = os.path.dirname(checkpoint_dir)
            self.model = 'cycle_gan'
            self.gpu_ids = gpu_ids
            self.phase = 'test'
            self.eval = True
            self.batch_size = 1
            self.num_threads = 0
            self.preprocess = 'resize_and_crop'
            self.load_size = 286
            self.crop_size = 256
            self.direction = 'AtoB'  # 默认 A->B 方向，可根据模型训练时方向调整
            self.serial_batches = True
            self.no_flip = True
            self.verbose = False
            self.results_dir = 'results'
            self.isTrain = False 
            self.input_nc = 3          # 输入图像通道数，比如RGB是3
            self.output_nc = 3         # 输出图像通道数
            self.ngf = 64              # generator的滤波器数量，默认64
            self.netG = 'resnet_9blocks'  # generator类型
            self.norm = 'instance'     # 归一化类型
    
    opt = Args()
    
    # 创建模型
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()
    
    # 读取图片路径
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(root, f))
    image_paths.sort()
    
    # 处理每张图片
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert('RGB')
        
        # 预处理（官方预处理和训练时保持一致）
        transform_list = [
            transforms.Resize(opt.load_size),
            transforms.CenterCrop(opt.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform = transforms.Compose(transform_list)
        img_tensor = transform(img).unsqueeze(0)  # 增加 batch 维度
        
        # 放入模型
        model.set_input({'A': img_tensor.to(opt.gpu_ids[0])})  # 注意 key 'A' 或 'B' 依训练设置调整
        model.test()
        
        visuals = model.get_current_visuals()
        fake_img = visuals['fake_B'].cpu()  # 假设AtoB方向转换结果在 fake_B
        
        # 后处理保存图片
        save_img = util.tensor2im(fake_img)
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        Image.fromarray(save_img).save(out_path)
        
        if i % 50 == 0:
            print(f"[{i}/{len(image_paths)}] Saved converted image to {out_path}")
    print(f"All images converted and saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGAN inference script")
    parser.add_argument('--input_dir', type=str, required=True, help='Input images folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Folder to save converted images')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='CycleGAN checkpoint folder (checkpoints/<name>)')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='GPU ids to use')
    args = parser.parse_args()
    
    run_inference(args.input_dir, args.output_dir, args.checkpoint_dir, args.gpu_ids)
