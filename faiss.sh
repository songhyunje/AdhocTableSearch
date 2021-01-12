export CUDA_VISIBLE_DEVICES=0
ckpt_file=./0110_2_E5_20200401/checkpoint-epoch=04-val_loss=0.00.ckpt
SEED=20200401

python faiss_indexers.py --data_dir ./data/1/ \
  --bert_path bert-base-uncased \
  --tabert_path ./model/tabert_base_k3/model.bin \
  --gpus 1 \
  --ckpt_file ${ckpt_file} \
  --topk 5 \
