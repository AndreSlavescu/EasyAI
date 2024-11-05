import torch
import triton
import triton.language as tl
import math

@triton.jit
def fft_forward_kernel(data_ptr, output_ptr, N,
                       stride_in, stride_out,
                       BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    real = tl.load(data_ptr + offs * stride_in * 2, mask=mask, other=0.0)
    imag = tl.load(data_ptr + offs * stride_in * 2 + 1, mask=mask, other=0.0)

    num_bits = tl.static_log2(BLOCK_SIZE)
    rev = tl.bitcast(offs, tl.uint32)
    rev = tl.bitrev(rev, num_bits)
    real = tl.load(data_ptr + rev * stride_in * 2, mask=mask, other=0.0)
    imag = tl.load(data_ptr + rev * stride_in * 2 + 1, mask=mask, other=0.0)

    m = 2
    while m <= BLOCK_SIZE:
        k = offs % m
        group = (offs // m) * m
        pos = group + k
        idx1 = pos
        idx2 = pos + m // 2

        angle = -2 * math.pi * (k % (m // 2)) / m
        w_real = tl.cos(angle)
        w_imag = tl.sin(angle)

        u_real = tl.load(real + idx1, mask=mask, other=0.0)
        u_imag = tl.load(imag + idx1, mask=mask, other=0.0)

        v_real = tl.load(real + idx2, mask=mask, other=0.0)
        v_imag = tl.load(imag + idx2, mask=mask, other=0.0)

        t_real = w_real * v_real - w_imag * v_imag
        t_imag = w_real * v_imag + w_imag * v_real

        real_new1 = u_real + t_real
        imag_new1 = u_imag + t_imag
        real_new2 = u_real - t_real
        imag_new2 = u_imag - t_imag

        tl.store(real + idx1, real_new1, mask=mask)
        tl.store(imag + idx1, imag_new1, mask=mask)
        tl.store(real + idx2, real_new2, mask=mask)
        tl.store(imag + idx2, imag_new2, mask=mask)

        m *= 2

    tl.store(output_ptr + offs * stride_out * 2, real, mask=mask)
    tl.store(output_ptr + offs * stride_out * 2 + 1, imag, mask=mask)
    
@triton.jit
def fft_backward_kernel(grad_output_ptr, grad_input_ptr, N,
                        stride_grad_out, stride_grad_in,
                        BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    grad_real = tl.load(grad_output_ptr + offs * stride_grad_out * 2, mask=mask, other=0.0)
    grad_imag = tl.load(grad_output_ptr + offs * stride_grad_out * 2 + 1, mask=mask, other=0.0)

    num_bits = tl.static_log2(BLOCK_SIZE)
    rev = tl.bitcast(offs, tl.uint32)
    rev = tl.bitrev(rev, num_bits)
    grad_real = tl.load(grad_output_ptr + rev * stride_grad_out * 2, mask=mask, other=0.0)
    grad_imag = tl.load(grad_output_ptr + rev * stride_grad_out * 2 + 1, mask=mask, other=0.0)

    m = 2
    while m <= BLOCK_SIZE:
        k = offs % m
        group = (offs // m) * m
        pos = group + k
        idx1 = pos
        idx2 = pos + m // 2

        angle = 2 * math.pi * (k % (m // 2)) / m 
        w_real = tl.cos(angle)
        w_imag = tl.sin(angle)

        u_real = tl.load(grad_real + idx1, mask=mask, other=0.0)
        u_imag = tl.load(grad_imag + idx1, mask=mask, other=0.0)

        v_real = tl.load(grad_real + idx2, mask=mask, other=0.0)
        v_imag = tl.load(grad_imag + idx2, mask=mask, other=0.0)

        t_real = w_real * v_real - w_imag * v_imag
        t_imag = w_real * v_imag + w_imag * v_real

        grad_real_new1 = u_real + t_real
        grad_imag_new1 = u_imag + t_imag
        grad_real_new2 = u_real - t_real
        grad_imag_new2 = u_imag - t_imag

        tl.store(grad_real + idx1, grad_real_new1, mask=mask)
        tl.store(grad_imag + idx1, grad_imag_new1, mask=mask)
        tl.store(grad_real + idx2, grad_real_new2, mask=mask)
        tl.store(grad_imag + idx2, grad_imag_new2, mask=mask)

        m *= 2

    grad_real = grad_real / N
    grad_imag = grad_imag / N

    tl.store(grad_input_ptr + offs * stride_grad_in * 2, grad_real, mask=mask)
    tl.store(grad_input_ptr + offs * stride_grad_in * 2 + 1, grad_imag, mask=mask)

class FusedFFTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        N = input.shape[0]
        output = torch.empty_like(input)
        
        stride_in = input.stride(0)
        stride_out = output.stride(0)
        
        grid = lambda meta: (1,)
        fft_forward_kernel[grid](input.data_ptr(), output.data_ptr(), N,
                                 stride_in, stride_out, BLOCK_SIZE=N)
        
        ctx.N = N
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        N = ctx.N
        grad_input = torch.empty_like(grad_output)
        
        stride_grad_out = grad_output.stride(0)
        stride_grad_in = grad_input.stride(0)
        
        grid = lambda meta: (1,)
        fft_backward_kernel[grid](grad_output.data_ptr(), grad_input.data_ptr(), N,
                                  stride_grad_out, stride_grad_in, BLOCK_SIZE=N)
        return grad_input

class FastFourierTransform(torch.nn.Module):
    def __init__(self):
        super(FastFourierTransform, self).__init__()

    def forward(self, x):
        return FusedFFTFunction.apply(x)

if __name__ == "__main__":
    N = 8
    input_real = torch.randn(N, dtype=torch.float32, device='cuda')
    input_imag = torch.randn(N, dtype=torch.float32, device='cuda')
    input_tensor = torch.empty((N, 2), dtype=torch.float32, device='cuda')
    input_tensor[:, 0] = input_real
    input_tensor[:, 1] = input_imag
    input_tensor.requires_grad = True

    layer = FastFourierTransform()
    output = layer(input_tensor)

    input_complex = input_real + 1j * input_imag
    expected_output_complex = torch.fft.fft(input_complex).to('cuda')
    expected_output = torch.empty_like(input_tensor)
    expected_output[:, 0] = expected_output_complex.real
    expected_output[:, 1] = expected_output_complex.imag

    print("Custom FFT Output:\n", output)
    print("Expected FFT Output:\n", expected_output)
    print("Difference:\n", output - expected_output)

    if torch.allclose(output, expected_output, atol=1e-4):
        print("Forward FFT implementation is correct.")
    else:
        print("Forward FFT implementation is incorrect.")

    loss = output.sum()
    loss.backward()

    input_tensor_pt = input_tensor.clone().detach().requires_grad_(True)
    input_complex_pt = input_tensor_pt[:, 0] + 1j * input_tensor_pt[:, 1]
    output_pt = torch.fft.fft(input_complex_pt)
    loss_pt = output_pt.real.sum() + output_pt.imag.sum()
    loss_pt.backward()
    expected_grad = torch.empty_like(input_tensor)
    expected_grad[:, 0] = input_tensor_pt.grad.real
    expected_grad[:, 1] = input_tensor_pt.grad.imag

    print("Custom FFT Input Gradient:\n", input_tensor.grad)
    print("Expected Input Gradient:\n", expected_grad)
    print("Difference:\n", input_tensor.grad - expected_grad)

    if torch.allclose(input_tensor.grad, expected_grad, atol=1e-4):
        print("Backward FFT implementation is correct.")
    else:
        print("Backward FFT implementation is incorrect.")
