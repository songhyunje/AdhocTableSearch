#!/bin/bash

for i in {1..5}
do
    DATA=./data/$i
    echo $DATA
    
    mkdir -p ${DATA}/processed
    
    pushd ${DATA}/processed &> /dev/null
    rm -rf test.table
    rm -rf train.pair
    popd &> /dev/null
    
    pushd output/${i} &> /dev/null
    rm -rf checkpoint-epoch*
    popd &> /dev/null
    
    python trainer.py --data_dir ${DATA} \
                      --tabert_path ./model/tabert_base_k3/model.bin \
                      --config_file ./model/tabert_base_k3/tb_config.json \
                      --gpus 1 \
                      --precision 16 \
                      --max_epochs 10 \
                      --lr 5e-5 \
                      --do_train \
                      --gradient_clip_val 1.0 \
                      --train_batch_size 4 \
                      --valid_batch_size 2 \
                      --output_dir output/${i} \
                      --accumulate_grad_batches 16
    
    CKPT=`ls output/${i} | sort -k3 -t'=' | head -1`
    
    echo ${CKPT}
    python dense_retrieval.py --data_dir ${DATA} \
                              --gpus 1 \
                              --ckpt_file ./output/${i}/${CKPT} \
                              --hnsw_index | tee ./result/${i}_hnsw.result
    
    python dense_retrieval.py --data_dir ${DATA} \
                              --gpus 1 \
                              --ckpt_file ./output/${i}/${CKPT} | tee ./result/${i}.result
done
