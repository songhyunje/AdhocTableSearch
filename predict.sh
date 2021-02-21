#!/bin/bash

for i in {1..5}
do
    DATA=./data/$i
    echo $DATA

    CKPT=`ls output/${i} | sort -k3 -t'=' | head -1`
    
    echo ${CKPT}
    python dense_retrieval.py --data_dir ${DATA} \
                              --gpus 1 \
			      --topk 500 \
                              --ckpt_file ./output/${i}/${CKPT} \
                              --hnsw_index | tee ./result/${i}_hnsw.result
    
    python dense_retrieval.py --data_dir ${DATA} \
                              --gpus 1 \
			      --topk 500 \
                              --ckpt_file ./output/${i}/${CKPT} | tee ./result/${i}.result
done

python predict.py
