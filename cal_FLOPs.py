import torch
from torchsummary import summary
from thop import profile
from thop import clever_format
from model.L2SKNet.L2SKNet import L2SKNet_UNet as UNet
# from model.L2SKNet.o_modify_L2SKNet import L2SKNet_UNet as UNet


# 假设我们有一个预训练的模型
model = UNet(1, 1)
model.eval()

# 使用thop分析模型的运算量和参数量
input = torch.randn(1, 1, 256, 256)  # 随机生成一个输入张量，这个尺寸应该与模型输入的尺寸相匹配
MACs, params = profile(model, inputs=(input,))

# 将结果转换为更易于阅读的格式
MACs, params = clever_format([MACs, params], '%.3f')

print(f"参数量：{params}, 运算量：{MACs}")
