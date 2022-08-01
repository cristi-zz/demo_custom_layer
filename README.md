# Adding a custom layer in PyTorch
You have this brilliant idea, a way to transform input data to output data, that will inject so much inductive bias
into my network that the problem you are dealing with, will become trivial to learn, for a NN!

Excellent! How do we do that? How to derive a custom layer in PyTorch? We need the forward and backward code for it! 
But do we really need the backward step, too? Spoilers, usually no, but it's comforting to know that it can be done. 

## Setup

Use anaconda to create the env

    conda create -y --copy -c pytorch -n demo-layer python=3.9.2 numpy pytest matplotlib pillow jupyterlab ipywidgets ipympl pytorch=1.12.0 cudatoolkit=10.2

Activate the env, then run:

    conda install -c fastchan fastai

## Math

Check the [attached PDF](https://github.com/cristi-zz/demo_custom_layer/blob/master/pdf/custom_layer_math_derivation.pdf) for the explanations on how the data and gradients must flow in the network.

## Code

Go to ``src/`` folder and of course ``test/`` to see what's what. 

## Demo

In [``src/demo.ipynb``](https://github.com/cristi-zz/demo_custom_layer/blob/master/src/demo.ipynb) I show how
the network learns. 

![](https://github.com/cristi-zz/demo_custom_layer/blob/master/src/fig3_learning.gif "Animation of the learning process")

Enjoy!