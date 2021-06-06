SEED=20200401
KFOLD=1

python trainer.py --data_dir data/${KFOLD} \
  --tabert_path ./model/tabert_base_k3/model.bin \
  --config_file ./model/tabert_base_k3/tb_config.json \
  --gpus 1 \
  --precision 16 \
  --max_epochs 5 \
  --do_train \
  --train_batch_size 2 \
  --valid_batch_size 1 \
  --min_row 30 \
  --seed ${SEED} \
  --output_dir output/FOLD${KFOLD}_E${EPOCH}_S${SEED}/ \
  --accumulate_grad_batches 16 \

