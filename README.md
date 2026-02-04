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

# Real-World SetUP

1. The SDK of our Robots

[Franka](https://frankarobotics.github.io/docs/)

[Inspire Hand](https://wiki.pndbotics.com/dexterous_hands/inspire_dexterous_hands/)

[Xhand](https://github.com/yoimisan/xhand_control_python)

[Panda Gripper](https://github.com/frankarobotics/external_gripper_example)

2. Camera 

[CalibrationTools](https://github.com/littlebearsama/CalibrationTools)

3. VLM Planner

[Cliport](https://cliport.github.io/)

[Extend Cliport to 3D](https://github.com/google-research/ravens/blob/master/ravens/models/transport_6dof.py)

Also, if you donnot want to deploy CLIPort, we also suggest two **zero-shot**, more powerful and open-vocabulary VLM Planner

[RekeP](https://rekep-robot.github.io/)

[RoboBrain](https://superrobobrain.github.io/)

4. Point-SAM

4-1 2D Mask Segmentation

If you want more fast Segmentation, use

[GroundedSAM2](https://github.com/IDEA-Research/Grounded-SAM-2)

If you want to open-vocabulary Segmentation, try the following method

[Seg-R1](https://github.com/geshang777/Seg-R1)

4-2 3D PointCloud

[3D Point from 2D Mask](https://developer.supervisely.com/getting-started/python-sdk-tutorials/point-clouds/point-cloud-segmentation-with-2d-mask-guidance)

5. IK System

[Deoxys](https://github.com/UT-Austin-RPL/deoxys_control)

5. InspireHand Development(12DoF to 6 DoF)

[Maniptrans](https://maniptrans.github.io/)
# Acknowledgement
Thanks for the excellent work [OpenHOI](https://github.com/Zhenhao-Zhang/OpenHOI),[DexGYS](https://github.com/iSEE-Laboratory/Grasp-as-You-Say),[AffordDP](https://github.com/SshiJwu/AffordDP),[CLIPort](https://github.com/huangwl18/ReKep),[MotionGPT](https://github.com/OpenMotionLab/MotionGPT)