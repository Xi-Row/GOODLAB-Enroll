import torch
import custom_layernorm
import time
import argparse

# 定义一个PyTorch自定义操作
class CustomLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, beta):
        output = torch.empty_like(input)
        custom_layernorm.launch_layer_norm(output.data_ptr(), input.data_ptr(), gamma.data_ptr(), beta.data_ptr(), input.size(0), input.size(1))
        ctx.save_for_backward(input, gamma, beta, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma, beta, output = ctx.saved_tensors
        grad_input = torch.empty_like(input)
        grad_gamma = torch.empty_like(gamma)
        grad_beta = torch.empty_like(beta)
        # 这里需要实现反向传播逻辑（如果需要的话）
        return grad_input, grad_gamma, grad_beta

# 将自定义操作包装为PyTorch模块
class CustomLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape):
        super(CustomLayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return CustomLayerNormFunction.apply(x, self.gamma, self.beta)

# 测试代码
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LayerNorm Performance Test")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    batch_size = args.batch_size
    seq_len = args.seq_len

    x = torch.randn(batch_size, seq_len, device=device)
    custom_norm = CustomLayerNorm(seq_len).to(device)
    pytorch_norm = torch.nn.LayerNorm(seq_len).to(device)

    # 测试自定义CUDA内核
    torch.cuda.synchronize()
    start_time = time.time()
    custom_output = custom_norm(x)
    torch.cuda.synchronize()
    custom_time = time.time() - start_time

    # 测试PyTorch原生LayerNorm
    torch.cuda.synchronize()
    start_time = time.time()
    pytorch_output = pytorch_norm(x)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time

    # 验证结果
    assert torch.allclose(custom_output, pytorch_output), "Results do not match!"

    # 打印加速比
    speedup = pytorch_time / custom_time
    print(f"Batch Size: {batch_size}, Seq Length: {seq_len}, Speedup: {speedup:.2f}")