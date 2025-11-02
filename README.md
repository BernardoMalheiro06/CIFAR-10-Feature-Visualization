# CIFAR-10-Feature-Visualization

- [About the Project](#about-the-project)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Visualization Techniques](#visualization-techniques)
- [Roadmap](#roadmap)
- [Results](#results)
- [Conclusion](#conclusion)

Pre-release now public — feedback welcome!

## About the Project
This project focuses on visualizing the features learned by convolutional neural networks (CNNs) trained on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal of this project is to understand how CNNs interpret and represent different features of the images through various visualization techniques.

## Requirements
The following libraries/tools were used to run the code in this project:
- Python 3.12 or above
- Pytorch 2.9.0
- torchvision 0.13.0
- matplotlib 3.5.1
- numpy 1.21.2


## Dataset
The CIFAR-10 dataset is available at [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html). It contains the following classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Visualization Techniques
My goal is using different techniques to visualize the features learned by the CNNs:
1. **Activation Maximization in the Input Space**: This technique involves generating images that maximize the activation of specific neurons in the network. It helps to understand what features a particular neuron is responsive to.
2. **Optimization in the Latent Space of a Decoder Model**: This method optimizes images in the latent space of a decoder pretrained on the CIFAR-10 dataset in a variational autoencoder (VAE).
3. **Optimization in a Decorrelated Latent Space**: This approach aims to optimize images in a latent space that has been decorrelated to better visualize the features learned by the CNN.

## Roadmap
1. `Classifier.ipynb`: A notebook that contains the training and evaluation of the CNN classifier trained on the CIFAR-10 dataset.
2. `VAE.ipynb`: A notebook that implements the variational autoencoder (VAE) used for latent space optimization.
3. `Visualization.ipynb`: A notebook that implements the various feature visualization techniques mentioned above.
- `utils.py`: A utility file containing helper functions for visualization.
- `models.py`: A file containing the definitions of the CNN and VAE models used in the project.

## Results
The results of the feature visualization techniques are documented in the `Visualization.ipynb` notebook. The visualizations provide insights into the features learned by the CNNs and how they correspond to different classes in the CIFAR-10 dataset.

## Conclusion
This project demonstrates the effectiveness of various feature visualization techniques in understanding the inner workings of convolutional neural networks trained on the CIFAR-10 dataset. The visualizations help to reveal the features that the networks focus on when making predictions, providing valuable insights into their decision-making processes.
