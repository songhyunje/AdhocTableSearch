#!/bin/bash

pushd output &> /dev/null
rm -rf checkpoint-epoch*
popd &> /dev/null

python trainer.py --data_dir ./data/1 \
                  --tabert_path ./model/tabert_base_k3/model.bin \
                  --gpus 2 \
                  --precision 16 \
                  --max_epochs 10 \
                  --do_train \
                  --train_batch_size 4 \
                  --valid_batch_size 1 \
                  --seed 1235 \
                  --output_dir output \
                  --accumulate_grad_batches 5

CKPT=`ls output | sort -k3 -t'=' | head -1`

echo ${CKPT}
python dense_retrieval.py --data_dir ./data/1/ \
                          --gpus 1 \
                          --ckpt_file ./output/${CKPT} \
                          --hnsw_index \

python dense_retrieval.py --data_dir ./data/1/ \
                          --gpus 1 \
                          --ckpt_file ./output/${CKPT} \

