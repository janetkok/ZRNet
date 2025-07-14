# ZRNet: Physics-Informed Graph Neural Networks with Frequency-Aware Learning for Optical Aberration Correction


## Installation
This implementation is based on [BasicSR](https://github.com/xinntao/BasicSR) which is an open-source toolbox for image/video restoration tasks, [NAFNet](https://github.com/megvii-research/NAFNet), [Restormer](https://github.com/swz30/Restormer/tree/main/Denoising) and [Multi Output Deblur](https://github.com/Liu-SD/multi-output-deblur)


```
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```


## Data Preparation

Download CytoImageNet dataset from [Kaggle](https://www.kaggle.com/datasets/stanleyhua/cytoimagenet?resource=download).

Add full_path column to the metadata.csv file.
```
import pandas as pd
import os

input_filename = 'metadata.csv'
# The base directory to prepend to the path
base_dir = '/db'

df = pd.read_csv(input_filename)
df['full_path'] = base_dir  + df['path'] + '/' + df['filename']
df.to_csv(input_filename, index=False)

```
## Training

*Pretraining ZRNet_mlp*
```
torchrun --nproc_per_node=1 --master_port=4370 basicsr/train_infer.py -opt 'Options/zrnet_mlp.yml' --launcher pytorch
```
*Training ZRNet*
The pretrained ZRNet_mlp can be downloaded from from [here](https://zenodo.org/record/14865721/files/pretrainedmodels.zip?download=1)
```
torchrun --nproc_per_node=1 --master_port=4370 basicsr/train_infer.py -opt 'Options/zrnet_azi_train.yml' --launcher pytorch

```
## Testing
Download the trained [models](https://zenodo.org/record/14865721/files/pretrainedmodels.zip?download=1) and run the code
```
torchrun --nproc_per_node=1 --master_port=4370 basicsr/train_infer.py -opt 'Options/zrnet_azi_infer.yml' --launcher pytorch

```

## Notes
You could try training and testing ZRNet with different grouping strategies of Zernike Graphs.
- No grouping: Options/zrnet_nogrp_train.yml
- Aberration grouping: Options/zrnet_ab_train.yml
