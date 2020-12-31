export CUDA_VISIBLE_DEVICES=1
ckpt_file=./1231_5_E2_20200401/checkpoint--epoch=01-val_loss=-0.00.ckpt
SEED=20200401

python predict.py --data_dir ./data \
  --bert_path bert-base-uncased \
  --tabert_path ./model/tabert_base_k3/model.bin \
  --gpus 1 \
  --test_batch_size 1 \
  --seed ${SEED} \
  --ckpt_file ${ckpt_file} \
  --output_file ${ckpt_file}.csv \
  --qrel_file ./data/test.jsonl.qrels \
  --result_file ./data/test.jsonl.result \
