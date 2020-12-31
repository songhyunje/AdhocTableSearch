import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

from table_bert import Table, Column, TableBertModel

class QueryTableDataset(Dataset):
    def __init__(self, data_dir: str = '.data', data_type: str = 'train',
                 query_tokenizer=None, table_tokenizer=None, max_query_length=15,
                 prepare=False):
        super().__init__()
        self.data_dir = data_dir
        self.query_file = data_type + '.query'
        self.table_file = data_type + '.table'
        self.ids_file = data_type + '.pair'
        self.data_type = data_type # test, train 구분하기위해
        self.prepare(data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length)
        # TODO: NLL loss에 최대한 Pos table이 들어가는 방향으로 수정을 해야함
        #self.pos_tables, self.neg_tables = torch.load(os.path.join(self.processed_folder, self.table_file))
        #self.query = torch.load(os.path.join(self.processed_folder, self.query_file))
        #self.tables = torch.load(os.path.join(self.processed_folder, self.table_file))
        self.pair_ids = torch.load(os.path.join(self.processed_folder, self.ids_file))

    def __len__(self):
        return len(self.pair_ids)

    def __getitem__(self, index):
        return self.pair_ids[index]

    def prepare(self, data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length):
        if self._check_exists():
            return
        processed_dir = Path(self.processed_folder)
        processed_dir.mkdir(exist_ok=True)
        if not (query_tokenizer and table_tokenizer):
            raise RuntimeError('Tokenizers are not found.' +
                               ' You must set query_tokenizer and table_tokenizer')
        print('Processing...')

        query_dict = defaultdict()
        pairs = []
        tables = defaultdict(list)
        path = Path(data_dir + '/' + data_type + '.jsonl')
        rel_to_score = {
            '0': 0.0,
            '1': 0.5,
            '2': 1.0,
        }

        with open(path) as f:
            for line in f.readlines():
                if not line.strip():
                    break
                # 테이블 기본 Meta data 파싱
                jsonStr = json.loads(line)
                tableId = jsonStr['docid']
                query = jsonStr['query']
                qid = jsonStr['qid']
                rel = jsonStr['rel']

                if qid not in query_dict:
                    query_tokenized = query_tokenizer.encode_plus(query,
                                                                  max_length=max_query_length,
                                                                  padding='max_length',
                                                                  truncation=True,
                                                                  return_tensors="pt"
                                                                  )
                    query_dict[qid] = query_tokenized ##BERT **input input_ids, seg_ids, mas_ids

                # Raw Json 파싱
                raw_json = json.loads(jsonStr['table']['raw_json'])
                title = raw_json['pgTitle']
                secTitle = raw_json['secondTitle']
                hRow = raw_json['numHeaderRows']
                row = raw_json['numDataRows']
                col = raw_json['numCols']
                caption = raw_json['caption']
                heading = raw_json['title']
                body = raw_json['data']

                if col == 0 or row == 0:
                    continue

                column_rep = Table(id=title,
                                   header=[Column(h.strip(), 'text') for h in heading],
                                   data=body
                                   ).tokenize(table_tokenizer)
                # TODO: caption을 다양하게 주는부분, 비교실험 해볼부분임
                caption = " ".join(heading) + " " + title + " " + secTitle + " " + caption
                caption_rep = table_tokenizer.tokenize(caption)

                # Save
                pairs.append([qid, query_tokenized, tableId, column_rep, caption_rep, float(rel_to_score[str(rel)])])

        with open(os.path.join(processed_dir, self.ids_file), 'wb') as f:
            torch.save(pairs, f)
        print('Done!')

    @property
    def processed_folder(self):
        return os.path.join(self.data_dir, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.ids_file)))


def query_table_collate_fn(batch):
    qid, query, tid, column, caption, rel = zip(*batch)
    input_ids, token_type_ids, attention_mask = [], [], []
    for q in query:
        input_ids.append(q["input_ids"].squeeze())
        token_type_ids.append(q["token_type_ids"].squeeze())
        attention_mask.append(q["attention_mask"].squeeze())

    query = {"input_ids": torch.stack(input_ids),
             "token_type_ids": torch.stack(token_type_ids),
             "attention_mask": torch.stack(attention_mask)}
    return query, column, caption, rel, qid, tid


if __name__ == "__main__":
    query_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    table_model = TableBertModel.from_pretrained('model/tabert_base_k3/model.bin')
    table_tokenizer = table_model.tokenizer

    dataset = QueryTableDataset(data_dir='data',
                                data_type='train',
                                query_tokenizer=query_tokenizer,
                                table_tokenizer=table_tokenizer,
                                prepare=True
                                )
    dataset = QueryTableDataset(data_dir='data',
                                data_type='test',
                                query_tokenizer=query_tokenizer,
                                table_tokenizer=table_tokenizer,
                                prepare=True
                                )
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            collate_fn=query_table_collate_fn)

    for _ in range(1):
        for d in dataloader:
            print(d)
            break
