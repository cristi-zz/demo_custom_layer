import torch
from torch import nn
import numpy as np
from typing import List


class LinearActFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weights) -> torch.Tensor:
        """
        Chapter 2.3 from
        https://github.com/cristi-zz/demo_custom_layer/blob/master/pdf/custom_layer_math_derivation.pdf

        We implement the forward and backward functions for a simple, fully connected layer with a sigmoid activation.
        The purpose is to check the math and how the values are aggregated along the matrix dimensions.
        """
        z_output = input.matmul(weights.t())
        sigma_output = torch.sigmoid(z_output)
        ctx.save_for_backward(input, weights, sigma_output)
        return sigma_output

    @staticmethod
    def backward(ctx, grad_outputs) -> List[torch.Tensor]:
        """
        Backward overload.

        We get: Cost gradient wrt to layer output
        We give: Cost gradient wrt to weights, Cost gradient wrt to inputs
        """
        input, weights, sigma_output = ctx.saved_tensors
        sigma_deriv = sigma_output * (1 - sigma_output)  # Eq 19
        partial_grad = grad_outputs * sigma_deriv  # Eq 38, larger bracket. Will be reused for eq 43
        grad_weight = partial_grad.transpose(1,2).matmul(input) # The rest of the eq 38
        grad_weight = grad_weight.squeeze(0)  # To match expected output shape
        grad_input = partial_grad.matmul(weights)  # Equation 43

        return grad_input, grad_weight


class GaussianConvFunction_simple(torch.autograd.Function):

    @staticmethod
    def get_padding(n):
        nk = n - (1 - n %2)
        pad = int((nk - 1) / 2)  # compute padding so out signal len == in signal len
        return nk, pad

    @staticmethod
    def forward(ctx, input, sigma, mu) -> torch.Tensor:
        """
        Chapter 3 from
        https://github.com/cristi-zz/demo_custom_layer/blob/master/pdf/custom_layer_math_derivation.pdf

        Forward pass implements a Gaussian

        Input shape 1 x Time
        sigma, mu scalars

        Out: 1 x Time

        """
        nk, pad = GaussianConvFunction_simple.get_padding(input.shape[1])  # compute padding so out signal len == in signal len
        kernel_support = torch.arange(-nk/2, nk/2, 1, dtype=input.dtype, requires_grad=False)
        kernel_base = kernel_support - mu
        kernel_base = torch.exp(-1 * torch.square(kernel_base) / (2 * sigma * sigma))
        kernel = kernel_base / (sigma * np.sqrt(2 * np.pi))
        kernel = kernel.to(input.dtype)
        kernel = kernel.flip(0) # Because conv1d == freakin' cross correlation
        out_signal = nn.functional.conv1d(input.unsqueeze(0), kernel.view(1, 1, -1),padding=pad)
        ctx.save_for_backward(input, kernel_support, kernel_base, mu, sigma, out_signal)
        return out_signal.squeeze(0)

    @staticmethod
    def backward(ctx, grad_outputs) -> List[torch.Tensor]:
        input, kernel_support, kernel_base, mu, sigma, out_signal = ctx.saved_tensors
        grad_input = grad_sigma = grad_mu = None
        nk, pad = GaussianConvFunction_simple.get_padding(input.shape[1])  # compute padding so out signal len == in signal len
        sqrt2pi = np.sqrt(2 * np.pi)

        kernel_mu = (kernel_support - mu) / (torch.pow(sigma, 3) * sqrt2pi) * kernel_base
        kernel_mu = kernel_mu.flip(0)
        conv_with_mu = nn.functional.conv1d(input.unsqueeze(0), kernel_mu.view(1, 1, -1), padding=pad)
        grad_mu = grad_outputs.squeeze(0).dot(conv_with_mu.view(-1))
        grad_mu = grad_mu.unsqueeze(0)

        kernel_sigma = kernel_base / sqrt2pi * (torch.pow(kernel_support - mu, 2) / torch.pow(sigma, 4) - 1 / torch.pow(sigma, 2))
        kernel_sigma = kernel_sigma.flip(0)
        conv_with_sigma = nn.functional.conv1d(input.unsqueeze(0), kernel_sigma.view(1, 1, -1), padding=pad)
        grad_sigma = grad_outputs.squeeze(0).dot(conv_with_sigma.view(-1))
        grad_sigma = grad_sigma.unsqueeze(0)

        kernel_x = kernel_base / (sigma * sqrt2pi)
        grad_input = nn.functional.conv1d(grad_outputs.unsqueeze(0), kernel_x.view(1, 1, -1), padding=pad)

        return grad_input, grad_sigma, grad_mu


class GaussConvLayer_simple(nn.Module):
    """
    Apply the GaussianConvFunction_simple to some data

    The module stores the sigma and mu parameters
    """
    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        self.sigma.data[0] = 1
        self.mu = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

    def forward(self, input):
        out = GaussianConvFunction_simple.apply(input, self.sigma, self.mu)
        return out
