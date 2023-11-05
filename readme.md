# SAGAN (Self-Attention Generative Adversarial Network)

## Introduction
SAGAN is a generative adversarial network (GAN) architecture that incorporates self-attention mechanisms to improve the quality of generated images. This README provides an overview of a SAGAN model implementation which generates 0s and 1s from MNIST images.

## Training Details
The SAGAN model is trained on randomly selected MNIST images consisting of 200 0s and 200 1s. The training process involves the following parameters:
- Number of training epochs: 100
- Batch size: 32
- Z dimension: 20
- Learning rate for the generator: 1e-4
- Learning rate for the discriminator: 4e-4
- Beta values for the optimizer: 0.0 and 0.9
- Seed: 103

## Model Architecture
The SAGAN model consists of a generator and a discriminator. The generator takes random noise as input and generates images, while the discriminator tries to distinguish between real and generated images. The self-attention mechanism is incorporated into both the generator and discriminator to capture long-range dependencies and improve the quality of generated images.

## Loss Function
The SAGAN model uses the hinge version of the adverserial loss to generate high-quality images. The loss function is optimized using gradient descent with the Adam optimizer.

## Results
The SAGAN model has been trained on the MNIST dataset and was able to generate MNIST-like 0s and 1s. The models gave best results around 100 training epochs and started to collapse if trained for higher epochs. Examples of the generated images can be found in the `figs` directory.
