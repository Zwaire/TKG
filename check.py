import torch

# 检查 CUDA 是否可用
print("CUDA 是否可用：", torch.cuda.is_available())

# 查看 GPU 数量
if torch.cuda.is_available():
    print("GPU 数量：", torch.cuda.device_count())
    print("当前 GPU 名称：", torch.cuda.get_device_name(0))
else:
    print("当前使用：CPU")