from datasets import load_from_disk
import pandas as pd

# 设置路径，指定包含所有 Arrow 文件的文件夹
dataset_path = "/home/liruizheng/opencompass/datasets/llm_dataset/eval/c4"

# 直接加载数据集
dataset = load_from_disk(dataset_path)

# 打印基本信息
print(f"Dataset features: {dataset.features}")
print(f"Number of rows: {len(dataset)}")

# 将前几行数据保存到指定文件夹下的文件
output_file = "/home/liruizheng/opencompass/opencompass/configs/eval/arrow_checker/c4_arrow.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    # 写入数据集基本信息
    f.write(f"Dataset features: {dataset.features}\n")
    f.write(f"Number of rows: {len(dataset)}\n")
    f.write("=" * 50 + "\n\n")
    
    for i in range(len(dataset)):
        f.write(f"Sample {i}:\n")
        f.write(f"{dataset[i]}\n")
        f.write("-" * 30 + "\n\n")

print(f"数据已保存到: {output_file}")

# 同时打印第一个样本到控制台
print("第一个样本:")
print(dataset[0])