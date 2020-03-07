# CNNs for image classification
Back up of code I used to train CNNs for image classification from scratch.

### Requirement
- `pytorch 1.1.0+`
- `torchvision`
- `numpy`
- `pyyaml`
- `tqdm`
- `tensorboardX`
- `pillow`

### Dataset
- `CIFAR-10`
- `CIFAR-100`
- `ImageNet 2012`

### Usage
- Add configuration file under `configs` folder
  - `runid` needs to be configured in validate mode and test mode to obtain the model's parameters file
  - If `cuda` is not specified, model use cpu. `cuda` can be specified as `"all"` to use all GPUs, or a list of GPUs, such as `"0,1"`
- run `train.py`, `validate.py` or `test.py`