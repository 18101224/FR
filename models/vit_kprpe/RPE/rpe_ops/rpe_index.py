import torch
import rpe_index_cpp


EXPECTED_VERSION = "1.2.0"
assert rpe_index_cpp.version() == EXPECTED_VERSION, \
        f"""Unmatched `rpe_index_cpp` version: {rpe_index_cpp.version()}, expected version: {EXPECTED_VERSION}
Please re-build the package `rpe_ops`."""


class RPEIndexFunction(torch.autograd.Function):
    '''Y[b, h, i, j] = input[b, h, i, index[i, j]]'''
    @staticmethod
    def forward(ctx, input, index):
        '''
        Y[b, h, i, j] = input[b, h, i, index[i, j]]

        Parameters
        ----------
        input: torch.Tensor, float32 or bfloat16
            The shape is (B, H, L_query, num_buckets)
        index: torch.Tensor, int32
            The shape is (L_query, L_key)

        where B is the batch size, and H is the number of attention heads.

        Returns
        -------
        Y: torch.Tensor, same dtype as `input`
            The shape is (B, H, L_query, L_key)
        '''
        ctx.save_for_backward(index)
        ctx.input_shape = input.shape
        ctx.input_dtype = input.dtype
        # The native extension only dispatches float/half. Route BF16 through
        # FP32 internally, then cast back so the external contract stays stable.
        ctx.promote_bf16 = input.dtype == torch.bfloat16
        dispatch_input = input
        if ctx.promote_bf16:
            dispatch_input = input.float()
        forward_fn = rpe_index_cpp.forward_cpu if \
            input.device.type == 'cpu' else rpe_index_cpp.forward_gpu
        output = forward_fn(dispatch_input, index)
        if ctx.promote_bf16:
            output = output.to(dtype=ctx.input_dtype)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        '''
          - Inputs
              grad_output: float32/bfloat16 (B, H, L_query, L_key)
          - Outputs
              grad_input: same dtype as the original input
                  (B, H, L_query, num_buckets)
        '''
        index = ctx.saved_tensors[0]
        if ctx.needs_input_grad[0]:
            dispatch_grad_output = grad_output
            if ctx.promote_bf16:
                dispatch_grad_output = grad_output.float()
            grad_input = dispatch_grad_output.new_zeros(ctx.input_shape)
            backward_fn = rpe_index_cpp.backward_cpu if \
                grad_output.device.type == 'cpu' else rpe_index_cpp.backward_gpu
            backward_fn(grad_input, dispatch_grad_output, index)
            if ctx.promote_bf16:
                grad_input = grad_input.to(dtype=ctx.input_dtype)
            return grad_input, None
        return None, None


if __name__ == '__main__':
    import numpy as np
    import time
    B = 128
    H = 32
    L_query = 50
    L_key = L_query
    num_buckets = 50

    x = torch.randn(B, H, L_query, num_buckets)

    index = torch.randint(low=0, high=num_buckets, size=(L_query, L_key))
    index = index.to(torch.int)
    offset = torch.arange(0, L_query * num_buckets, num_buckets).view(-1, 1)

    def test(x, index, offset):
        tic = time.time()
        x1 = x.clone()
        x1.requires_grad = True
        x2 = x.clone()
        x2.requires_grad = True
        cmp_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        cmp_decimal = 2 if x.dtype in (torch.float16, torch.bfloat16) else 5

        y = RPEIndexFunction.apply(x1, index)
        gt_y = x2.flatten(2)[:, :, (index + offset).flatten()
                             ].view(B, H, L_query, L_key)

        np.testing.assert_almost_equal(
            gt_y.detach().to(cmp_dtype).cpu().numpy(),
            y.detach().to(cmp_dtype).cpu().numpy(),
            decimal=cmp_decimal)

        mask = torch.randn(gt_y.shape, device=x.device, dtype=cmp_dtype)
        (gt_y.to(cmp_dtype) * mask).sum().backward()
        (y.to(cmp_dtype) * mask).sum().backward()

        print("X1:", x1.grad.to(cmp_dtype).cpu().numpy().flatten().sum())
        print("X2:", x2.grad.to(cmp_dtype).cpu().numpy().flatten().sum())
        np.testing.assert_almost_equal(
            x1.grad.to(cmp_dtype).cpu().numpy(),
            x2.grad.to(cmp_dtype).cpu().numpy(),
            decimal=cmp_decimal)
        print("Test over", x.device)
        print("dtype:", x.dtype)
        print("Cost:", time.time() - tic)
    test(x, index, offset)
    test(x.to(torch.bfloat16), index, offset)
    if torch.cuda.is_available():
        test(x.cuda(), index.cuda(), offset.cuda())
        if torch.cuda.is_bf16_supported():
            test(x.cuda().to(torch.bfloat16), index.cuda(), offset.cuda())
