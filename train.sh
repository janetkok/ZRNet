
torchrun --nproc_per_node=1 --master_port=4372 basicsr/train_infer.py -opt 'Options/zrnet_azi_infer.yml' --launcher pytorch
