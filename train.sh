export CUDA_VISIBLE_DEVICES=1
EPOCH=2
SEED=20200401
python trainer.py --data_dir ./data \
  --bert_path bert-base-uncased \
  --tabert_path ./model/tabert_base_k3/model.bin \
  --gpus 1 \
  --precision 32 \
  --max_epochs ${EPOCH} \
  --do_train \
  --do_predict \
  --train_batch_size 4 \
  --valid_batch_size 4 \
  --test_batch_size 1 \
  --seed ${SEED} \
  --output_dir ./1231_5_E${EPOCH}_${SEED}/ \
  --output_file ./data/Tresult1231_5_E${EPOCH}_${SEED}.csv \

