# Ad-hoc Table Search (In Progress)

#### 사용법 Summary
```
1. bash setup.sh
2. train 
3. predict
```

#### Train Step 
```shell script
1. vi train.sh  수정 
>>> 
export CUDA_VISIBLE_DEVICES=1
EPOCH=10
SEED=19940203
python trainer.py --data_dir ./data \
  --bert_path bert-base-uncased \
  --tabert_path ./model/tabert_base_k3/model.bin \
  --gpus 1 \
  --precision 32 \
  --max_epochs ${EPOCH} \
  --do_train \
  --train_batch_size 4 \ # 2080ti는 MAX 2, 3090은 4-5개
  --valid_batch_size 4 \ # 2080ti는 MAX 2, 3090은 4-5개
  --test_batch_size 1 \
  --seed ${SEED} \
  --output_dir ./1231_5_E${EPOCH}_${SEED}/ \ # ckpt 저장 dir
  --do_predict \ # Training 이 끝나고 predict를 할것인지 Flag 변수  
  --output_file ./1231_5_E${EPOCH}_${SEED}/result_E${EPOCH}_${SEED}.csv \ # predict 플래그가 있으면 결과물 저장 file이름

2. ./train.sh  
```

#### Predict Step 
```shell script
요약: 모델로드 -> inference (ouput: csv) -> csv to TREC format -> run TREC_script
1. vi predict.sh  수정 
>>> 
export CUDA_VISIBLE_DEVICES=1
ckpt_file=./1231_5_E2_20200401/checkpoint--epoch=01-val_loss=-0.00.ckpt
SEED=20200401

python predict.py --data_dir ./data \
  --bert_path bert-base-uncased \
  --tabert_path ./model/tabert_base_k3/model.bin \
  --gpus 1 \
  --test_batch_size 1 \
  --seed ${SEED} \
  --ckpt_file ${ckpt_file} \ # load하려는 ckpt파일 
  --output_file ${ckpt_file}.csv \ # 결과물 csv파일
  --qrel_file ./data/test.jsonl.qrels \ # K-fold qrels파일 
  --result_file ./data/test.jsonl.result \ # 저장원하는 result파일 이름 

2. ./predict.sh
...
Testing: 100%|█████████████████████| 615/615 [00:11<00:00, 51.53it/s]
>>> Json to TREC format ... Done...
>>> Run TREC Script ... Done...
{'ndcg_cut_5': 0.3694, 'ndcg_cut_10': 0.4626, 'ndcg_cut_15': 0.4995, 'ndcg_cut_20': 0.501}
```

#### TODO 

- [ ] **Dataloader에 배치안에 우선적으로 Postive를 있게끔 loader수정**
- [ ] Table Slicing 코드 추가 
- [ ] test_step에서 스코어 = 슬라이싱 테이블의 최대값으로 설정  
- [ ] Faiss 코드 정리 및 업로드
 