import torch
from torch import nn
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
