import torch

def softmax(x:torch.Tensor, dim: int=-1) -> torch.Tensor:
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x,dim=dim,keepdim=True)
    return exp_x/sum_exp_x

# 测试代码
logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]], dtype=torch.float)
softmax_output = softmax(logits, dim=1)  # 按行计算 softmax
print("Handwritten softmax output:\n", softmax_output)
print("Sum of probabilities (should be 1):\n", torch.sum(softmax_output, dim=1))

# 与 PyTorch 内置函数对比
torch_softmax = torch.nn.functional.softmax(logits, dim=1)
print("\nPyTorch softmax output:\n", torch_softmax)