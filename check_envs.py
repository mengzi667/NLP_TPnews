import sys
import os
import transformers
import torch
import pandas
import sklearn

# --- 打印环境诊断报告 ---

print("="*50)
print("              环境诊断报告 (Environment Forensic Report)")
print("="*50)

# 1. 打印当前运行此脚本的Python解释器的确切路径
print(f"\n[1] Python可执行文件路径 (Python Executable Path):")
print(f"    - {sys.executable}")
print("    => 这个路径【必须】包含你的Conda环境名(NLP_TPnews)，否则就是用错了Python。")


# 2. 打印关键库的版本和安装位置
print(f"\n[2] 核心库版本与位置 (Core Libraries Version & Location):")
print(f"    - transformers 版本: {transformers.__version__}")
print(f"    - transformers 位置: {os.path.dirname(transformers.__file__)}")
print("-" * 20)
print(f"    - torch 版本: {torch.__version__}")
print(f"    - torch 位置: {os.path.dirname(torch.__file__)}")
print("-" * 20)
print(f"    - pandas 版本: {pandas.__version__}")
print(f"    - scikit-learn 版本: {sklearn.__version__}")
print("    => `transformers`的版本【必须】是 4.30.0 或更高。")

# 3. 打印GPU相关信息
print(f"\n[3] GPU 与 CUDA 信息 (GPU & CUDA Info):")
cuda_available = torch.cuda.is_available()
print(f"    - PyTorch能否找到CUDA: {cuda_available}")
if cuda_available:
    print(f"    - CUDA 版本 (PyTorch自带): {torch.version.cuda}")
    print(f"    - 当前GPU设备: {torch.cuda.get_device_name(0)}")
else:
    print("    - 警告：PyTorch未能找到可用的CUDA设备，模型将在CPU上运行，速度会非常慢。")

print("\n" + "="*50)