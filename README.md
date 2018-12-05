# An automatic nuclei segmentation method based on deep convolutional neural networks for histopathology images
Hwejin Jung, Bilal Lodhi, Bumsoo Kim, Junhyun Lee and Jaewoo Kang
Submitted in [Plos One](http://journals.plos.org/plosone/) 

## Introduction
Since nuclei segmentation in histopathology images can provide key information for identifying the presence or stage of a disease, the images need to be assessed carefully. However, color variation in histopathology images, and various structures of nuclei are two major obstacles in accurately segmenting and analyzing histopathology images. Several machine learning methods heavily rely on hand-crafted features which have limitations due to manual thresholding. To obtain robust results, deep learning based methods have been proposed. Deep convolutional neural networks used for automatically extracting features from raw image data have been proven to achieve great performance. Inspired by such achievements, we propose a nuclei segmentation method based on deep convolutional neural networks. To normalize the color of histopathology images, we use a deep convolutional Gaussian mixture color normalization model which is able to cluster pixels while considering the structures of nuclei. To segment nuclei, we use Mask R-CNN which achieves state-of-the-art object segmentation performance in the field of computer vision. In addition, we perform multiple inference as a post-processing step to boost segmentation performance. We evaluate our segmentation method on two different datasets. The first dataset consists of histopathology images of various organ while the other consists histopathology images of the same organ. Performance of our segmentation method is measured in various experimental setups at the object-level and the pixel-level. In addition, we compare the performance of our method with that of existing state-of-the-art methods. The experimental results show that our nuclei segmentation method outperforms the existing methods in terms of accuracy.

## Requirments

This code has been tested on Ubuntu 16.04 64-bit system.

### Prerequisites

#### color normalization
Python 2
TensorFlow
numpy

#### mask r-cnn
Python 3
Pytorch
torchvision
cython
numpy
matplotlib
opencv
pyyaml
packaging
pycocotools

## Contact

Hwejin Jung(hwejin23@gmail.com)



## Acknowledgements

color normalization code is inspired by the [Color-Normalization
](https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization).
mask r-cnnn code is inspired by the [Detectron](https://github.com/roytseng-tw/Detectron.pytorch).
