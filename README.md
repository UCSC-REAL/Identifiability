# Identifiability of Label Noise Transition Matrix
This code is a PyTorch implementation of our paper "[Identifiability of Label Noise Transition Matrix]" accepted by ICML 2023.
The code is run on the Tesla V-100.

## Prerequisites
Python 3.8.5

PyTorch 1.7.1

Torchvision 0.9.0


## Guideline
### Downloading dataset: 

Download the dataset from **http://www.cs.toronto.edu/~kriz/cifar.html** Put the dataset on **data/**


### Get pre-trained model:

We get the SimCLR and IPIRM models from **https://github.com/Wangt-CN/IP-IRM**. You can train the SimCLR and IPIRM models from that repository or directly download our trained models from  "[this url](https://drive.google.com/drive/folders/1SCxN8dY2ap7DK2E_jEWppYxlv89-e5FU?usp=sharing)" and put the model in **pretrained/**

### Estimate the transition matrix error:

Run command below to view the error of **weak model**, **simclr model** and **ipirm model** under instance 0.6 noise rate on CIFAR10.

```
python HOC_estimate_T.py --self_sup_type weak --noise_type instance --noise_rate 0.6
python HOC_estimate_T.py --self_sup_type simclr --noise_type instance --noise_rate 0.6
python HOC_estimate_T.py --self_sup_type ipirm --noise_type instance --noise_rate 0.6
```
### Forward loss correction with estimated transition matrix:
Following traditional pipeline of forward loss correction, we first select the model based on vanilla cross entropy loss, then finetune the model with estimated transition matrix using forward loss correction. 

First run command below to select the best model.

```
python CE_select_best.py --noise_type instance --noise_rate 0.6 
```
Then run command below to see the performance of forward loss correction.

```
python HOC_estimate_T_train.py --self_sup_type simclr --noise_type instance --noise_rate 0.6
python HOC_estimate_T_train.py --self_sup_type ipirm --noise_type instance --noise_rate 0.6
```

### Initializing DNN using disentangled features:

Run command below to see the effect of disentangled features on CIFAR100.
```
python CE_init.py --self_sup_type random --noise_type instance --noise_rate 0.6
python CE_init.py --self_sup_type simclr --noise_type instance --noise_rate 0.6
python CE_init.py --self_sup_type ipirm --noise_type instance --noise_rate 0.6
```


## References

**https://github.com/Wangt-CN/IP-IRM**

**https://github.com/UCSC-REAL/HOC**