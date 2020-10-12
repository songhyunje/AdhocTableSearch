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

        if prepare:
            self.prepare(data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length)

        self.query = torch.load(os.path.join(self.processed_folder, self.query_file))
        self.pos_tables, self.neg_tables = torch.load(os.path.join(self.processed_folder, self.table_file))

    def __len__(self):
        return len(self.query)

    def __getitem__(self, index):
        qid = self.query[index][0]
        # TODO: 아래 코드는 임시로 동작하기 위해 작동해둔 코드임. 반드시 향후에 수정해야할 부분
        pos_tables = self.pos_tables.get(qid, self.neg_tables.get(qid))
        tid1, pos_column_rep, pos_caption_rep = random.choice(pos_tables)
        tid2, neg_column_rep, neg_caption_rep = random.choice(self.neg_tables[qid])
        return self.query[index][1], pos_column_rep, pos_caption_rep, neg_column_rep, neg_caption_rep

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
        pos_tables, neg_tables = defaultdict(list), defaultdict(list)

        path = Path(data_dir + '/' + data_type + '.jsonl')
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
                    query_dict[qid] = query_tokenized

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
                    neg_tables[qid].append((tableId, column_rep, caption_rep))
                else:
                    pos_tables[qid].append((tableId, column_rep, caption_rep))

        queries = [(k, v) for k, v in query_dict.items()]
        tables = (pos_tables, neg_tables)
        with open(os.path.join(processed_dir, self.query_file), 'wb') as f:
            torch.save(queries, f)
        with open(os.path.join(processed_dir, self.table_file), 'wb') as f:
            torch.save(tables, f)

        print('Done!')

    @property
    def processed_folder(self):
        return os.path.join(self.data_dir, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.query_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.table_file)))


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
    # query_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # bert_model = BertModel.from_pretrained('bert-base-uncased')
    # table_model = TableBertModel.from_pretrained('model/tabert_base_k3/model.bin')
    # table_tokenizer = table_model.tokenizer
    #
    # dataset = QueryTableDataset(data_dir='data',
    #                             data_type='train',
    #                             query_tokenizer=query_tokenizer,
    #                             table_tokenizer=table_tokenizer,
    #                             prepare=True
    #                             )
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=query_table_collate_fn)
    # for d in dataloader:
    #     print(d)
    #     break

    # Prepare=False
    dataset = QueryTableDataset(data_dir='data',
                                data_type='train',
                                prepare=False
                                )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=query_table_collate_fn)
    for d in dataloader:
        print(d)
        break