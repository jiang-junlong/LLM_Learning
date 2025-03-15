import torch
import torch.nn.functional as F
from sympy.abc import epsilon

def sigmoid(x:torch.Tensor):
    return 1/(1+torch.exp(-x))

# 二元交叉熵损失
def cross_entropy(predictions:torch.Tensor, targets:torch.Tensor):
    epsilon=1e-12
    predictions=torch.clamp(predictions,epsilon,1.-epsilon)
    return -torch.mean(targets*torch.log(predictions)+(1-targets)*torch.log(1-predictions))
outputs = torch.tensor([0.8,0.2,0.4],dtype=torch.float)
target = torch.tensor([1,0,1],dtype=torch.float)
outputs_sigmoid=sigmoid(outputs)
# 计算自定义交叉熵损失
print("Custom cross entropy:", cross_entropy(outputs_sigmoid, target))

# 与 PyTorch 内置函数对比
print("PyTorch BCE:", F.binary_cross_entropy(outputs_sigmoid, target))