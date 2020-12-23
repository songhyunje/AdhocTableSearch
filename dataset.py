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
        self.data_type = data_type # test, train 구분하기위해

        if prepare and self.data_type.strip() == 'test':
            self.prepare(data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length)
        elif prepare:
            self.prepare(data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length)

        self.query = torch.load(os.path.join(self.processed_folder, self.query_file))
        #self.pos_tables, self.neg_tables = torch.load(os.path.join(self.processed_folder, self.table_file))
        self.tables = torch.load(os.path.join(self.processed_folder, self.table_file))

    def __len__(self):
        return len(self.query)

    def __getitem__(self, index):
        # TODO: 아래 코드는 임시로 동작하기 위해 작동해둔 코드임. 반드시 향후에 수정해야할 부분
        qid = self.query[index][0]
        tables = self.tables.get(qid) #table inform, rel
        tid, col_rep, cap_rep, rel_score = random.choice(tables)

        '''
        if self.data_type == "test":
            # if Test:
            # return query, table_column, table_caption, qid, tid
            tid1, pos_column_rep, pos_caption_rep = [t[0] for t in pos_tables], \
                                                    [t[1] for t in pos_tables], \
                                                    [t[2] for t in pos_tables]
            neg_column_rep = qid
            neg_caption_rep = tid1
        else:
            tid1, pos_column_rep, pos_caption_rep, rel1 = random.choice(pos_tables)
            tid2, neg_column_rep, neg_caption_rep, rel2 = random.choice(self.neg_tables[qid])
        '''

        #return self.query[index][1], pos_column_rep, pos_caption_rep, neg_column_rep, neg_caption_rep, [rel1, rel2]
        return self.query[index][1], col_rep, cap_rep, rel_score

    def prepare_test(self, data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length):
        # if Test:
        # 따로 안 나누고 postive쪽에 다 넣어놔야함
        if self._check_exists():
            return

        processed_dir = Path(self.processed_folder)
        processed_dir.mkdir(exist_ok=True)

        if not (query_tokenizer and table_tokenizer):
            raise RuntimeError('Tokenizers are not found.' +
                               ' You must set query_tokenizer and table_tokenizer')

        print('Test Processing...')
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
                # Caption 부븐을 변화를 준다.
                # {'ndcg_cut_5': 0.3525, 'ndcg_cut_10': 0.4645, 'ndcg_cut_15': 0.4795, 'ndcg_cut_20': 0.4813}
                caption = " ".join(heading) + " " + title + " " + secTitle + " " + caption
                #caption = title + " " + secTitle + " " + caption

                caption_rep = table_tokenizer.tokenize(caption)
                pos_tables[qid].append((tableId, column_rep, caption_rep))

        queries = [(k, v) for k, v in query_dict.items()]
        tables = (pos_tables, neg_tables)
        with open(os.path.join(processed_dir, self.query_file), 'wb') as f:
            torch.save(queries, f)
        with open(os.path.join(processed_dir, self.table_file), 'wb') as f:
            torch.save(tables, f)

        print('Done!')

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
        tables = defaultdict(list)

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
                "안녕 [SEP] 안녕2 ?? 안녕 ' ' 안녕 "
                caption = " ".join(heading) + " " + title + " " + secTitle + " " + caption
                caption_rep = table_tokenizer.tokenize(caption)
                #if rel == '0':
                #    neg_tables[qid].append((tableId, column_rep, caption_rep, rel))
                #else:
                #    pos_tables[qid].append((tableId, column_rep, caption_rep, rel))
                tables[qid].append(( tableId, column_rep, caption_rep, torch.tensor(int(rel), dtype=torch.long)) )

        queries = [(k, v) for k, v in query_dict.items()]
        #tables = (pos_tables, neg_tables)
        tables = tables

        with open(os.path.join(processed_dir, self.query_file), 'wb') as f:
            torch.save(queries, f) #query
        with open(os.path.join(processed_dir, self.table_file), 'wb') as f:
            torch.save(tables, f) #tabke information, rel_score

        print('Done!')

    @property
    def processed_folder(self):
        return os.path.join(self.data_dir, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.query_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.table_file)))


def query_table_collate_fn(batch):
    query, column, caption, rel = zip(*batch)
    input_ids, token_type_ids, attention_mask = [], [], []
    for q in query:
        input_ids.append(q["input_ids"].squeeze())
        token_type_ids.append(q["token_type_ids"].squeeze())
        attention_mask.append(q["attention_mask"].squeeze())

    query = {"input_ids": torch.stack(input_ids),
             "token_type_ids": torch.stack(token_type_ids),
             "attention_mask": torch.stack(attention_mask)}
    return query, column, caption, rel


if __name__ == "__main__":
    # cased인지 uncased인지 확인
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
    sum = 0
    for _ in range(1):
        for d in dataloader:
            q, tcol, tcontex, qid, tid = d
            print(qid, tid, len(tid[0]))
            sum += len(tid[0])
        print(sum)
        print("_"*50)


    # Prepare=False
    # dataset = QueryTableDataset(data_dir='data',
    #                             data_type='test',
    #                             prepare=True
    #                             )
    #dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=query_table_collate_fn)
    # dataloader = DataLoader(dataset,
    #                         batch_size=1,
    #                         collate_fn=query_table_collate_fn
    #                         )
    # i = 1
    # for d in dataloader:
    #     i+=1