# KIRBY 
An offitial implementation for "**Key Feature Replacement of In-Distribution Samples for out-of-Distribution Detection (AAAI 2023)**".

Authors: Jaeyoung Kim, Seo Taek Kong, Dongbin Na, Kyu-Hwan Jung

## Setup

* [Pytorch installation](https://pytorch.org/get-started/locally/)

```bash
pip install scikit-learn==1.0.2
pip install opencv-python==4.7.0.72
pip install torchcam==0.3.2
pip install tqdm
```

## Runs

#### STEP 1: construct OOD samples
```bash
# using pretrained wide-resnet trained with CIFAR10 
python generate_ood_data.py --dataset cifar10 --method layercam
```

#### STEP 2: training the rejection network
```bash
# CIFAR10 (ID) vs. SVHN (OOD)
python train_rejection_net.py
```


## Citation

```
@article{kim2022key,
  title={Key Feature Replacement of In-Distribution Samples for Out-of-Distribution Detection},
  author={Kim, Jaeyoung and Kong, Seo Taek and Na, Dongbin and Jung, Kyu-Hwan},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```