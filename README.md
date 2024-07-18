# S-Eye
Codebase of the BSc thesis by Linfeng Guo "Integrating Explainable AI and Semantic Segmentation For Advanced Failure Prediction in Autonomous Driving"



## Overview

Our tool S-Eye leverages the attention maps by the explainable AI algorithms SmoothGrad or Faster-ScoreCAM and the segmentation maps by a U-Net model. By 'Merging' both maps together road attentions and background attentions can be distinguished and turn them into a confidence score to automatically predict incoming failures of a DNN-based autonomous driving system (ADS). The figure below shows the attention maps of a confident ADS.


#incomming image and results 

<img src="images/nominal.gif" height="200" />


## Dependencies

**Software setup:** We adopted the PyCharm Professional 2023, a Python IDE by JetBrains.

First, you need [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) installed on your machine. Then, you can create and install all dependencies on a dedicated virtual environment, by running one of the following commands, depending on your platform.

```python
# macOS
conda env create -f environments.yml 

# Windows
conda env create -f windows.yml
```

Furthermore, for training the N-Net mode you additionally need to install Python 3.10.12 and PyTorch 2.2.1 with Torch-Vision 0.17.1. 


**Hardware setup:** Training the DNN models (self-driving cars and U-Net) on our datasets is computationally expensive. Therefore, we recommend using a machine with a GPU. In our setting, we ran our experiments on a machine equipped with a AMD Ryzen 7 5800H CPU processor, NVIDIA GeForce RTX 3060 Laptop GPU, and 8GB DDR4 RAM. 

 




## Replicate S-Eye experiments

### Datasets & Simulator

Driving datasets, self-driving car models, and simulator have a combined size of several GBs. All the Datasets can be found here: https://drive.google.com/drive/folders/18Bya6gTL_3H0UM0QG89TxCRRDfE9RiPf?usp=sharing

### Mutants

We use the following mutation operators:

* udacity_add_weights_regularisation_mutated0_MP_l1_3_1
* udacity_add_weights_regularisation_mutated0_MP_l1_l2_3_2
* udacity_add_weights_regularisation_mutated0_MP_l2_3_0
* udacity_change_activation_function_mutated0_MP_exponential_4_0
* udacity_change_activation_function_mutated0_MP_hard_sigmoid_4_0
* udacity_change_activation_function_mutated0_MP_relu_4_2
* udacity_change_activation_function_mutated0_MP_selu_4_0
* udacity_change_activation_function_mutated0_MP_sigmoid_4_3
* udacity_change_dropout_rate_mutated0_MP_0.125_0.125_6_2
* udacity_change_dropout_rate_mutated0_MP_1.0_1.0_6_1
* udacity_change_activation_function_mutated0_MP_softmax_4_4
* udacity_change_activation_function_mutated0_MP_softsign_4_5
* udacity_change_activation_function_mutated0_MP_tanh_4_2
* udacity_change_dropout_rate_mutated0_MP_0.25_0.25_6_7
* udacity_change_dropout_rate_mutated0_MP_0.75_0.75_6_0
* udacity_change_label_mutated0_MP_12.5_4
* udacity_change_label_mutated0_MP_25.0_1
* udacity_change_loss_function_mutated0_MP_mean_absolute_error_2

##Instruction
* For U_Net model, run 'segmentation/train'
* For XAI, run 'xai/heatmap'. Both SmoothGrad and Faster-ScoreCAM are included 
 
### Evaluation Scripts

For replicating the RQs, you can run:

* For S-Eye RAR/ARAR, run the file `scripts/evaluate`


```
