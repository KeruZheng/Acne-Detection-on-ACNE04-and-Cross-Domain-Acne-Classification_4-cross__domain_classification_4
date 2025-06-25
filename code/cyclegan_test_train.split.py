import os
import random
import shutil
# 数据集根目录
root_dir = "/data_lg/keru/project/part2/pytorch-CycleGAN-and-pix2pix/CycleGAN_data"
# 遍历 positive 和 negative 主文件夹
for main_folder in ["negative", "positive"]:
    main_folder_path = os.path.join(root_dir, main_folder)
    # 遍历 trainA 和 trainB 子文件夹
    for sub_folder in ["trainA", "trainB"]:
        sub_folder_path = os.path.join(main_folder_path, sub_folder)
        test_folder_path = os.path.join(main_folder_path, f"test{sub_folder[-1]}")  # 构建 testA/testB 路径
        os.makedirs(test_folder_path, exist_ok=True)  # 创建测试集文件夹
        
        # 获取文件夹内所有文件
        files = [f for f in os.listdir(sub_folder_path) if os.path.isfile(os.path.join(sub_folder_path, f))]
        # 计算要划分的测试集文件数量（取整）
        test_count = int(len(files) * 0.2)  
        # 随机选择要移动的文件
        test_files = random.sample(files, test_count)  
        
        # 移动文件到测试集文件夹
        for file in test_files:
            src_path = os.path.join(sub_folder_path, file)
            dst_path = os.path.join(test_folder_path, file)
            shutil.move(src_path, dst_path)
            print(f"Moved {file} from {sub_folder_path} to {test_folder_path}")
