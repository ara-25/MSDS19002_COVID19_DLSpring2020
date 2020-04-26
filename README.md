# Covid-19 classification using Chest X-Ray Dataset
This repository contains code and results for COVID-19 classification assignment by Deep Learning Spring 2020 course offered at Information Technology University, Lahore, Pakistan. This assignment is only for learning purposes and is not intended to be used for clinical purposes.

## Dataset
Dataset is available [here](https://drive.google.com/file/d/1-HQQciKYfwAO3oH7ci6zhg45DduvkpnK/view).
This dataset contains chest X-Ray images classified into *infected* and *normal* categories.

## Description
Classification was done with transfer learning using the [VGG-16](https://arxiv.org/abs/1409.1556) and [ResNet-18](https://arxiv.org/abs/1512.03385) architectures and PyTorch.
Experiments were done using two approaches:
1. **Using pre-trained feature extraction layers:** Convolution layers pre-trained on ImageNet were used to extract features and custom classification layers were added and trained.
2. **Fine-tuning:** Models trained on ImageNet were finetuned.

## Results

### VGG-16

#### Pre-trained Feature extraction

Accuracy | F1-score
---------|---------
0.953 | 0.961

##### Confusion Matrices
- Train

  ![](confusion_matrices/vgg_train_conf_t1.png)

- Validation

  ![](confusion_matrices/vgg_valid_t1.png)
  
- Test

  ![](confusion_matrices/vgg_test_t1.png)
    
 
#### Fine-tuning

Accuracy | F1-score
---------|---------
0.973 | 0.978

##### Confusion Matrices
- Train

  ![](confusion_matrices/vgg_train_t2.png)

- Validation

  ![](confusion_matrices/vgg_valid_t2.png)
  
- Test

  ![](confusion_matrices/vgg_test_t2.png)

### ResNet-18

#### Pre-trained Feature extraction

Accuracy | F1-score
---------|---------
0.925 | 0.938

##### Confusion Matrices
- Train

  ![](confusion_matrices/resnet_train_t1.png)

- Validation

  ![](confusion_matrices/resnet_valid_t1.png)
  
- Test

  ![](confusion_matrices/resnet_test_t1.png)

#### Fine-tuning

Accuracy | F1-score
---------|---------
0.968 | 0.973

##### Confusion Matrices
- Train

  ![](confusion_matrices/resnet_train_t2.png)

- Validation

  ![](confusion_matrices/resnet_valid_t2.png)
  
- Test

  ![](confusion_matrices/resnet_test_t2.png)
