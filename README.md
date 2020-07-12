# CNNs for image classification
Train CNNs for image classification from scratch.

I post several pretrained weights below.

### Requirement
- `pytorch 1.4.0+`
- `torchvision`
- `tensorboard 1.14+`
- `numpy`
- `pyyaml`
- `tqdm`
- `pillow`

### Dataset
- `CIFAR-10`
- `CIFAR-100`
- `ImageNet 2012`

### Usage
- Add configuration file under `configs` folder as follows
  - ```
    cuda: "all"  # if not specified, use cpu. or specified as "0", "0,1"
    model:
        arch: resnet
        depth: 50
    data:
        dataset: imagenet
        train_dir: /PATH/TO/ILSVRC/train
        val_dir: /PATH/TO/ILSVRC/val
        workers: 16
    training:
        runid: xxxx  # recommended specified during validation and testing
        epochs: 100
        batch_size: 256
        loss:
            name: 'label_smooth'
            smoothing: 0.1
        optimizer:
            name: 'sgd'
            lr: 0.1
            weight_decay: 0.0001
            momentum: 0.9
        lr_schedule:
            name: 'multi_step'
            milestones: [30,60,90]
            gamma: 0.1
        save_interval: 1
        resume: save_model.pkl
        best_model: best_model.pkl
    ```
- run `train.py`, `validate.py` or `test.py` as follows
  - ```shell script
    python train.py --config configs/aaaa.yml
    ``` 
    
### Pretrained Model on ImageNet 2012

| Architecture | Top-1 error | Params | FLOPs | Pretrained weights |
| :----: | :----: | :----: |:----: | :----: |
| ResNet18 <br> (My Imp.)| 29.72 | 11.69M | 1.82G | [Google Drive](https://drive.google.com/open?id=1Sw9TUBtgRQDLNxpJupMnv6FTtOKgnlxI) <br>[Baidu Netdisk](https://pan.baidu.com/s/1fPsWBkb_Lh_bniYmt7DL-w) |
| ResNet18 <br> ([paper](https://arxiv.org/abs/1512.03385))| 30.43 | - | - | - |
| ResNet50 <br> (My Imp.)| 23.30 | 25.56M | 4.11G | [Google Drive](https://drive.google.com/open?id=1XreMz36IpUiEDsJtyU7t_QPKOs4JTB_C) <br>[Baidu Netdisk](https://pan.baidu.com/s/197FBBOgYPc1oxEsDkeo4Rg) |
| ResNet50 <br> ([paper](https://arxiv.org/abs/1512.03385))| 24.7 | - | - | - |
| ResNet101 <br> (My Imp.)| 22.18 | 44.55M | 7.84G | [Google Drive](https://drive.google.com/open?id=1Vrfl-Z590jGcFIn1-7Cz9lfgqD1sJ1hm) |
| ResNet101 <br> ([paper](https://arxiv.org/abs/1512.03385))| 22.44 | - | - | - |
||
| ResNeXt50 <br> (My Imp.) | 22.35 | 25.03M | 4.26G | [Google Drive](https://drive.google.com/open?id=1lI8Hi-XvJ42aBastq6FI3DhKU2sK92FH) <br>[Baidu Netdisk](https://pan.baidu.com/s/1t3gkJjPxfRFWWuE_C4U5rw) |
| ResNeXt50 <br> ([paper](https://arxiv.org/abs/1611.05431)) | 22.2 | - | - | - |
||
| SE ResNet50 <br> (My Imp.) | 22.64 | 28.09M | 4.12G | [Google Drive](https://drive.google.com/open?id=1Oyyhb43Y2kbGjT1EEgal-cm8fYQAemuj) <br>[Baidu Netdisk](https://pan.baidu.com/s/1tyfin8SqftpmzYhMvU2wxw) |
| SE ResNet50 <br> ([paper](https://arxiv.org/abs/1709.01507)) | 23.29 | - | - | - |
||
| CBAM ResNet50 <br> (My Imp.) | 22.40 | 28.07M | 4.13G | [Google Drive](https://drive.google.com/file/d/1SXxVdgBmnUTeXeHHXmTtHWquess9agiR) |
| CBAM ResNet50 <br> ([paper](https://arxiv.org/abs/1807.06521)) | 22.66 | - | - | - |
||
| SKNet50 <br> (My Imp.) | 21.26 | 27.49M | 4.50G | [Google Drive](https://drive.google.com/open?id=1h6NIwSemMrFDk4DWT7-Zdm9kolHljyZU) <br>[Baidu Netdisk](https://pan.baidu.com/s/1XTuMDqFuzljxmlfC2TKTyg) |
| SKNet50 <br> ([paper](https://arxiv.org/abs/1903.06586)) | 20.79 | - | - | - |
||
||
| MobileNet V2 0.5x <br> (My Imp.) | 35.62 | 1.97M | 138.46M | [Google Drive](https://drive.google.com/open?id=1Ve2EuZPOZIEPZulQaNXHQb0Xl6trcSby) |
| MobileNet V2 0.5x <br> ([paper](https://arxiv.org/abs/1801.04381)) | 35.6 | - | - | - |
| MobileNet V2 1x <br> (My Imp.) | 28.09 | 3.50M | 315.41M | [Google Drive](https://drive.google.com/open?id=18HMPfrhdFO2PRHVrm8PMZFDNJZ1QeMKB) <br>[Baidu Netdisk](https://pan.baidu.com/s/1yKD_2IEuEw8cZ9N4gkg6UA) |
| MobileNet V2 1x <br> ([paper](https://arxiv.org/abs/1801.04381)) | 28.0 | - | - | - |
||
| MobileNet V3 large <br> (My Imp.) | 26.79 | 5.48M | 230.05M | [Google Drive](https://drive.google.com/file/d/1-bPoxyg9FEczBXjoJZJZiPmYAdOm2iQs) |
| MobileNet V3 large <br> ([paper](https://arxiv.org/abs/1905.02244)) | 24.8 | - | - | - |
||

The hyperparameters and settings during my training for ResNet, ResNeXt, SENet, SKNet are the same as the paper, except I use **label smooth** loss.

And for MobileNet V2 and MobileNet V3, I follow the setup in [this project](https://github.com/d-li14/mobilenetv2.pytorch#training), and use **label smooth** loss too.


#### Pretrained weights usage
1. place the downloaded pretrained model in `runs/aaaa/xxxx` folder under this project, 
where `aaaa` is the name of configuration file and `xxxx` is **runid** in configuration file.
2. run `validate.py` or `test.py` as above.

