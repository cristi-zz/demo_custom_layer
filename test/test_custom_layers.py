import numpy as np
import pytest
import torch
from torch import nn
import custom_layers
from fastcore.foundation import L
from fastai.data.transforms import TfmdLists, DataLoaders, RandomSplitter, ToTensor
from fastai.losses import MSELossFlat
from fastai.learner import Learner
import fastai.callback.schedule   # Needed for fit_one_cycle

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


def generate_data_for_one_feature_gauss_test(N:int, sample_len:int, noise:bool=True):
    """
    Generate a bunch of sinus signals with various frequencies and a bit of noise.

    Returns an array of size N x sample_len
    """
    rnd_offsets = np.random.rand(N) * np.pi
    rnd_frequencies = np.random.randint(1, 3, N)
    signal = np.zeros((N, sample_len), np.float32)
    for k in range(N):
        signal[k, :] = np.linspace(0,  rnd_frequencies[k] * 2 * np.pi, sample_len) - rnd_offsets[k]
        signal[k, :] = np.sin(signal[k, :])
    if noise:
        signal *= np.random.rand(N, sample_len)
    return signal.astype(np.float32)


def gen_gauss(x_arr, sigma, miu):
    """
    Generates a Gaussian kernel with support x_arr and (miu, sigma) stats.
    """
    x_arr_shift = x_arr - miu
    gauss_kern = np.exp(-np.square(x_arr_shift) / (2*sigma*sigma))
    gauss_kern = gauss_kern * 1 / (sigma * np.sqrt(2*np.pi))

    return gauss_kern



def test_create_gauss_func_forward():
    """
    Tests the forward path of the custom activation
    """
    signal_len = 99
    np.random.seed(3345)
    asin = np.sin(np.linspace(0, 1 * 2 * np.pi, signal_len))
    signal = np.random.rand(signal_len) * asin

    gaussAct = custom_layers.GaussianConvFunction_simple.apply
    sigma = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    mu = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    input = torch.from_numpy(signal).to(torch.float32).view(1, -1)
    input.requires_grad = True
    sigma.data[0] = 4.5
    mu.data[0] = -19.5

    nk = signal_len - (1 - signal_len % 2)
    kernel_support = np.arange(-nk/2, nk/2, 1)
    kernel = gen_gauss(kernel_support, 4.5, -19.5)
    signal_convolved = np.convolve(signal, kernel, mode='same')

    out_vector = gaussAct(input, sigma, mu)
    out_vector_arr = out_vector.squeeze(0).detach().numpy()

    assert np.allclose(out_vector_arr, signal_convolved), "Forward output signal mistmatch."



def test_create_gauss_func_backward():
    """
    Tests the backward() of the custom function using pytorch numerical checker
    """
    signal_len = 100
    gaussAct = custom_layers.GaussianConvFunction_simple.apply
    sigma_c = torch.zeros((1,), dtype=torch.double, requires_grad=True)
    mu_c = torch.zeros((1,), dtype=torch.double, requires_grad=True)
    sigma_c.data[0] = 4.5
    mu_c.data[0] = -19.5
    input_c = torch.randn(1, signal_len, dtype=torch.double, requires_grad=True)
    some_inputs = (input_c, sigma_c, mu_c)

    assert torch.autograd.gradcheck(gaussAct, some_inputs, eps=1e-5, atol=1e-6)


def test_train_gaussLayer_with_simple_data():
    """
    Ultimate test, is a network with such a layer learns something?

    We take a bunch of signals, we shift them all, with an asymetric kernel
    We train a network to "restore" the shifted signals.
    Test shows that in few iterations the network learns the expected mu and sigma statistics.
    :return:
    """
    N = 500
    sample_len = 100
    target_sigma = 3
    target_mu = -11.5
    work_base = "dump"

    signal = generate_data_for_one_feature_gauss_test(N, sample_len, True)  # A bunch of signals

    nk = sample_len - (1 - sample_len % 2)
    kernel_support = np.arange(-nk/2, nk/2, 1)
    kernel = gen_gauss(kernel_support, target_sigma, target_mu)  # A Gauss kernel

    # We filter each signal, with the same asymetric kernel
    gt_signal = []
    for k in range(N):
        gt_sample = np.convolve(signal[k, :], kernel, mode='same')
        gt_signal.append(gt_sample)
    gt_signal = np.array(gt_signal).astype(np.float32)

    train_samples = zip(signal, gt_signal)
    train_samples = L(train_samples)

    splits = RandomSplitter(0.1)(train_samples)
    tls_train = TfmdLists(train_samples[splits[0]], [ToTensor()])
    dloaders = DataLoaders.from_dsets(tls_train, tls_train, bs=1, num_workers=1, device=torch.device('cpu'))
    model = custom_layers.GaussConvLayer_simple().cpu()

    learner = Learner(dloaders, model, loss_func=MSELossFlat(reduction="mean"), model_dir=work_base)

    learner.fit_one_cycle(2, 1e-2)
    print(f"Target mu: {target_mu}, actual mu: {model.mu.detach()}, target sigma: {target_sigma}, actual sigma: "
          f"{model.sigma.detach()}")
    learner.fit_one_cycle(2, 1e-2)
    print(
        f"Target mu: {target_mu}, actual mu: {model.mu.detach()}, target sigma: {target_sigma}, actual sigma: "
        f"{model.sigma.detach()}")
    learner.fit_one_cycle(5, 1e-2)
    print(
        f"Target mu: {target_mu}, actual mu: {model.mu.detach()}, target sigma: {target_sigma}, actual sigma: "
        f"{model.sigma.detach()}")

    assert model.mu.detach() == pytest.approx(target_mu, abs=0.5)
    assert model.sigma.detach() == pytest.approx(target_sigma, abs=0.5)