#!/bin/bash

# Some housekeeping
export TRANSFORMERS_CACHE=$MY_STORAGE_FOLDER/hf_cache
export WANDB_CACHE_DIR=$MY_STORAGE_FOLDER/.wandb_cache
export WANDB_DATA_DIR=$MY_STORAGE_FOLDER/.wandb_staging

source .env/bin/activate

# main run
python training.py --batch_size=32 --devices=2 --max_epochs=1 --val_check_interval=50000 --limit_val_batches=5000 --learning_rate 0.0001 --save_dir=output_test/ --teacher=config-robbert/ --data=/cw/dtaidata/ml/2019-berdt/data/oscar-2022/output_thomas.txt --val_data=/cw/dtaidata/ml/2019-berdt/data/oscar-2022/output_thomas_val.txt --num_workers=40 --accumulate_grad_batches=64 --precision=bf16 --log_every_n_steps=1