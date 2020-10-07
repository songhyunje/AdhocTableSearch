import json
import pickle
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

from table_bert import Table, Column, TableBertModel


class QueryTableDataset(Dataset):
    def __init__(self, query_file, table_file, query_tokenizer, table_tokenizer,
                 max_query_length=15, prepare=True
                 ):
        super().__init__()

        if prepare:
            self.query = []
            with open(query_file, 'r') as f:
                for line in f.readlines():
                    id, query = line.strip().split("\t")
                    # TODO: 수정해야할 부분, 불필요하게 같은 질의에 대해 encode_plus를 수행함.
                    query_tokenized = query_tokenizer.encode_plus(query,
                                                                  max_length=max_query_length,
                                                                  padding='max_length',
                                                                  truncation=True,
                                                                  return_tensors="pt"
                                                                  )
                    self.query.append((id, query_tokenized))

            self.pos_tables, self.neg_tables = defaultdict(list), defaultdict(list)
            with open(table_file, 'r') as f:
                for line in f.readlines():
                    if not line.strip():
                        break

                    # 테이블 기본 Meta data 파싱
                    jsonStr = json.loads(line)
                    tableId = jsonStr['docid']
                    qid = jsonStr['qid']
                    rel = jsonStr['rel']

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
                    caption_rep = table_tokenizer.tokenize(caption)
                    if rel == '0':
                        self.neg_tables[qid].append((tableId, column_rep, caption_rep))
                    else:
                        self.pos_tables[qid].append((tableId, column_rep, caption_rep))

            # pickle.dump([query, pos_tables, neg_tables], open("data/data.pkl", "wb"))

    def __len__(self):
        return len(self.query)

    def __getitem__(self, index):
        qid = self.query[index][0]
        # TODO: 아래 코드는 임시로 동작하기 위해 작동해둔 코드임. 반드시 향후에 수정해야할 부분
        pos_tables = self.pos_tables.get(qid, self.neg_tables.get(qid))
        tid1, pos_column_rep, pos_caption_rep = random.choice(pos_tables)
        tid2, neg_column_rep, neg_caption_rep = random.choice(self.neg_tables[qid])
        return self.query[index][1], pos_column_rep, pos_caption_rep, neg_column_rep, neg_caption_rep


def query_table_collate_fn(batch):
    query, pos_column, pos_caption, neg_column, neg_caption = zip(*batch)
    input_ids, token_type_ids, attention_mask = [], [], []
    for q in query:
        input_ids.append(q["input_ids"].squeeze())
        token_type_ids.append(q["token_type_ids"].squeeze())
        attention_mask.append(q["attention_mask"].squeeze())

    query = {"input_ids": torch.stack(input_ids),
             "token_type_ids": torch.stack(token_type_ids),
             "attention_mask": torch.stack(attention_mask)}
    return query, pos_column, pos_caption, neg_column, neg_caption


if __name__ == "__main__":
    query_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    table_model = TableBertModel.from_pretrained('model/tabert_base_k3/model.bin')
    table_tokenizer = table_model.tokenizer

    dataset = QueryTableDataset(query_file='data/sample_queries.txt',
                                table_file='data/sample.json',
                                query_tokenizer=query_tokenizer,
                                table_tokenizer=table_tokenizer
                                )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=query_table_collate_fn)
    for d in dataloader:
        print(d)
        break
