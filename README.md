# A Simple Unified Information Regularization Framework for Multi-Source Domain Adaptation
Pytorch implementation of MIAN: Multi-source Information-regularized Adaptation Network.
Provided as a supplementary code for ICML 2021. 
*Pytorch version: 1.4.0*

## Dataset
- We support three multi-domain adaptation datasets: 
  - Digits-Five (Peng et al., 2019): MNIST-M, MNIST, SVHN, SYNTH, USPS
  - Office-31 (Saenko et al., 2010): Amazon, Webcam, DSLR
    - [Link](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code)
  - Office-Home (Venkateswara et al., 2017): Art, Clipart, Realworld, Product
    - [Link](http://hemanthdv.org/OfficeHome-Dataset/)
- Every datasets should be downloaded in data/ directory following the order below:
    - `data/{office/office_home}/{domain}/{class}/{images}`
    - Digits-Five require .pkl files. 
      - `data/digits/{domain}/{train/val}.pkl`
    - every words in path should be written in lower-case.
  - Due to memory issue, data and pretrained weights are not provided at this moment.

## Train examples


```
python3 main.py --gpu 0 --task office --target Amazon --exp_name Amazon_test \ 
                        --advcoeff 0.1 --SVD_ld 0.0001 --no_MCD
```
- advcoeff: $\lambda_0$ in paper
- SVD_ld: $\mu_0$ in paper
  - Set to 0 for Vanila MIAN.
- no_MCD: Run main.py without Maximum Classifier Discrepancy (MCD, Saito et al., 2018)

```
python3 main.py --gpu 0 --task office --target Amazon --partial_domain Amazon DSLR --exp_name Amazon_test_partial_DSLR \ 
                        --advcoeff 0.1 --SVD_ld 0.0001 --no_MCD
```
- partial_domain: Specify domains to be utilized. (Includes target domain)
