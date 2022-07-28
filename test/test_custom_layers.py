import numpy as np
import torch
from torch import nn

import custom_layers

def test_createLinearFunc():
    """
    We test if the math+implementation of a sigma(w x) layer is numerically identical with official implementation.

    From same weights, inputs and expected outputs we do a forward step and a backward step through the regular
    torch.nn.Linear layer and through our manually derived layer.

    There are two methods, one is manual, going step by step and the other uses gradcheck that numerically checks
    the backward implementation to match the gradients computed numerically from multiple forward steps.

    :return:
    """
    InFeats = 4
    OutFeats = 2

    in_sample = np.array([0.1, 0.8, 0.9, 0.1])
    out_true = np.array([0.9, 0.1])

    # We create the custom function and a classic linear layer.
    customLinearFunction = custom_layers.LinearActFunction.apply
    lin_layer = nn.Linear(InFeats, OutFeats, bias=False).cpu()   # 4 features in, 2 features out.

    # Setup the weights to the same values.
    in_data_layer = torch.from_numpy(in_sample)
    in_data_layer = in_data_layer.view(1, 1, InFeats) # Batch, 1st dimension, features.
    in_data_layer.requires_grad_(True)
    weight = torch.randn(OutFeats, InFeats, dtype=torch.double, requires_grad=True)     # out_feat x in_feat matrix
    in_data_custom = torch.tensor(in_data_layer, requires_grad=True)
    with torch.no_grad():
        lin_layer.weight.data = weight

    # Forward step. Note the separate activation for the classic linear layer
    out_layer_data = torch.sigmoid(lin_layer(in_data_layer))
    out_custom_data = customLinearFunction(in_data_custom, weight)
    assert torch.allclose(out_layer_data, out_custom_data), "The forward step gave mismatch values"

    # Apply some loss on regular layer. Propagate backwards and extract the gradients.
    loss_layer = torch.sum(torch.square(out_layer_data - torch.tensor(out_true)))
    loss_layer.backward()
    weight_layer_grads = lin_layer.weight.grad.detach().numpy()
    input_layer_grads = in_data_layer.grad.detach().numpy()

    # Apply loss and call the backward() on custom layer. Extract the gradients.
    loss_custom = torch.sum(torch.square(out_custom_data - torch.tensor(out_true)))
    assert torch.allclose(loss_custom, loss_layer), "The loss computed by the custom layer is wrong"
    loss_custom.backward()
    weight_custom_grads = weight.grad.detach().numpy()
    input_custom_grads = in_data_custom.grad.detach().numpy()

    # Check the gradients
    assert np.allclose(weight_layer_grads, weight_custom_grads), "The gradient wrt to weigths is wrong"
    assert np.allclose(input_layer_grads, input_custom_grads), "The gradient wrt to inputs is wrong"

    # Take some inputs and check the gradient propagations
    some_inputs = (torch.randn(1,1, InFeats, dtype=torch.double,requires_grad=True),
                   torch.randn(OutFeats, InFeats,dtype=torch.double,requires_grad=True))
    assert torch.autograd.gradcheck(customLinearFunction, some_inputs, eps=1e-6, atol=1e-4), "Automatic gradient check, failed"
