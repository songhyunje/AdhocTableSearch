#!/bin/bash

python trainer.py --data_dir ./data/1 \
	--tabert_path ./model/tabert_base_k3/model.bin \
	--gpus 2 \
	--precision 16 \
	--max_epochs 20 \
	--do_train \
	--train_batch_size 4 \
	--valid_batch_size 1 \
	--seed 1235 \
	--output_dir output \
	--accumulate_grad_batches 10
