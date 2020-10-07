# Ad-hoc Table Search (In Progress)

#### 사용법 Summary
```
1. bash setup.sh

2. 다운로드 queries.txt, all.json후, data 디렉토리에 추가
```

#### TODO 
- 데이터셋 로딩 부분 수정
- DDP로 학습되지 않는 버그 수정 (단일 gpu로는 학습이 되는 것을 확인)
- 기존 pytorch_pretrained_bert이 아닌 transformers를 사용하도록 함. 
load_state_dict 할 때, Vertical_Attention_BERT 중 SpanBasedPrediction에서 loading이 되지 않음. 
추후 수정 예정
