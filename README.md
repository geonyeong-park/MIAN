# A Simple Unified Information Regularization Framework for Multi-Source Domain Adaptation
Pytorch implementation of MIAN: Multi-source Information-regularized Adaptation Network.
Provided as a supplementary code for NeurIPS 2020. 

## Dataset
- We support three multi-domain adaptation datasets: 
  - Digits-Five (Peng et al., 2019): MNIST-M, MNIST, SVHN, SYNTH, USPS
  - Office-31 (Saenko et al., 2010): Amazon, Webcam, DSLR
    - [Link](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code)
  - Office-Home (Venkateswara et al., 2017): Art, Clipart, Realworld, Product
    - [Link](http://hemanthdv.org/OfficeHome-Dataset/)
- Every datasets should be downloaded in data/ directory following the order below:
    - 'data/{office/office_home}/{domain}/{class}/{images}'
    - Digits-Five require .pkl files. 
      - 'data/digits/{domain}/{train/val}.pkl'
    - every words in path should be written in lower-case.
  - Due to memory issue, data and pretrained weights are not provided at this moment.

## Train examples


```
python train_gta2cityscapes_multi.py --snapshot-dir ./snapshots/GTA2Cityscapes_single_lsgan \
                                     --lambda-seg 0.0 \
                                     --lambda-adv-target1 0.0 --lambda-adv-target2 0.01 \
                                     --gan LS
```
