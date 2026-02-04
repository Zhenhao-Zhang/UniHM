# [ICLR 2026] UniHM: Unified Dexterous Hand Manipulation with Vision Language Model
This is the offical code repo for **ICLR 2026** paper **UniHM: Unified Dexterous Hand Manipulation with Vision Language Model**

[[Paper]](https://zhenhao-zhang.github.io/papers/UniHM/UniHM.pdf) [[Project Page]](https://zhenhao-zhang.github.io/UniHM_page/)

<div align="center">
    <img src="pipeline.png" height=500>
</div>


# Disclaimers
- Code Quality Level: Tired grad student, lots of hard code in my repo
- Training Enviroment: A800 80G GPUs
- Questions: please drop me an email, it is the fastest way to get feedback


# Plan

- [√] Paper Released.
- [√] Code.
- [√] Pretrained Weights.
- [√] Dataset.
- [√] Quick Satrt
- [√] Weights of UniHM

Any Question, feel free to contact zhangzhh2024@shanghaitech.edu.cn

# Enviroments SetUp
```
conda env create -f environment.yaml
```
Tips: You can install [dex-retargeting](https://github.com/dexsuite/dex-retargeting) first, then install [qwen](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B) and other packages.
# Pretrain Weights
[Qwen3-0.6b](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)

# DataSet
[DexYCB](https://dex-ycb.github.io/)

[OAKINK](https://github.com/oakink/OakInk)

# Quick Start
1. train VQVAE Encoder-Decoder
```
python train_vqvae.py
python train_vqvae_muti_encoder.py
```
2. Train DexHand VLM
```
python train_sft.py
```
# Weights of UniHM
[Weights](https://pan.baidu.com/s/1oYO_a6FOKloeHNCHiYnxVA?pwd=gcyb)

# Acknowledgement
Thanks for the excellent work [OpenHOI](https://github.com/Zhenhao-Zhang/OpenHOI),[DexGYS](https://github.com/iSEE-Laboratory/Grasp-as-You-Say),[AffordDP](https://github.com/SshiJwu/AffordDP),[CLIPort](https://github.com/huangwl18/ReKep),[MotionGPT](https://github.com/OpenMotionLab/MotionGPT)