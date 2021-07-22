import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from table_bert import Table, Column, TableBertModel


def encode_tables(table_json, is_slice, table_tokenizer, min_row):
    rel = table_json['rel']
    tid = table_json['table']['tid']

    raw_json = json.loads(table_json['table']['raw_json'])
    textBeforeTable = raw_json['textBeforeTable']  # 추후
    textAfterTable = raw_json['textAfterTable']    # 추후

    title = raw_json['pageTitle']
    caption = raw_json['title'].strip()  # Caption 역할
    tableOrientation = raw_json['tableOrientation'].strip()  # [HORIZONTAL, VERTICAL]

    headerPosition = raw_json['headerPosition']  # ['FIRST_ROW', 'MIXED', 'FIRST_COLUMN', 'NONE’]
    hasHeader = raw_json['hasHeader']            # [true, false]
    keyColumnIndex = raw_json['keyColumnIndex']
    headerRowIndex = raw_json['headerRowIndex']  # 0 == 첫줄, -1 == 없음
    entities = raw_json['entities']

    body = raw_json['relation']
    if tableOrientation == "HORIZONTAL":
        body = list(map(list, zip(*body)))  # transpose

    header = body[headerRowIndex] if hasHeader else [''] * len(body[0])

    # caption = caption if caption else title
    context = f'{title} {caption}'.strip()
    context_rep = table_tokenizer.tokenize(context)

    if is_slice:
        table_reps = slice_table(tid, header, body, table_tokenizer, min_row)
    else:
        table_reps = [Table(id=tid,
                            header=[Column(h.strip(), infer_column_type(h)) for h in header],
                            data=body
                           ).tokenize(table_tokenizer)]

    # memory issues!
    if len(table_reps) > 3:
        table_reps = table_reps[:3]

    return table_reps, context_rep


def slice_table(tid, heading, data, table_tokenizer, min_row):
    table_reps = []

    for i in range(0, len(data), min_row):
        rows = data[i:i+min_row]
        table_rep = Table(id=f'{tid}_{i}',
                           header=[Column(h.strip(), infer_column_type(h)) for h in heading],
                           data=rows
                           ).tokenize(table_tokenizer)
        table_reps.append(table_rep)

    return table_reps


class QueryTableDataset(Dataset):
    def __init__(self, data_dir: str = '.data', data_type: str = 'train',
                 query_tokenizer=None, table_tokenizer=None, max_query_length=7,
                 min_row=30, prepare=False, is_slice=True):
        self.data_dir = data_dir
        self.query_file = f'{data_type}.query'
        self.table_file = f'{data_type}.table'
        self.pos_rel_file = f'{data_type}.rel.pos'
        self.neg_rel_file = f'{data_type}.rel.neg'
        self.data_type = data_type
        self.hard_num = 1
        self.is_slice = is_slice
        if prepare:
            self.prepare(data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length, min_row=min_row)

        self.query_dict = torch.load(os.path.join(self.processed_folder, self.query_file))
        self.table_dict = torch.load(os.path.join(self.processed_folder, self.table_file))
        self.pos_rel = torch.load(os.path.join(self.processed_folder, self.pos_rel_file))

        # hard static negative
        self.neg_rel = torch.load(os.path.join(self.processed_folder, self.neg_rel_file))

        # # of unique query 
        self.qids = [(qid, tid) for qid, tids in self.pos_rel.items() for tid in tids]

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, index):
        qid, tid = self.qids[index]
        query = self.query_dict[qid]
        table = self.table_dict[tid]

        # hard static negative
        # hard_tids = random.sample(self.neg_rel[qid], self.hard_num)
        # hard_tables = [self.table_dict[hard_tid] for hard_tid in hard_tids]
        # return qid, query, tid, table, hard_tids, hard_tables

        hard_tid = random.choice(self.neg_rel[qid])
        hard_table = self.table_dict[hard_tid]
        return qid, query, tid, table, hard_tid, hard_table

    def prepare(self, data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length, min_row):
        if self._check_exists():
            return

        processed_dir = Path(self.processed_folder)
        processed_dir.mkdir(exist_ok=True)
        if not (query_tokenizer and table_tokenizer):
            raise RuntimeError('Tokenizers are not found.' +
                               ' You must set query_tokenizer and table_tokenizer')
        print('Processing...')

        query_dict = defaultdict()
        table_dict = defaultdict()
        pos_rel_dict = defaultdict(list)
        neg_rel_dict = defaultdict(list)
        path = Path(data_dir + '/' + data_type + '.jsonl')

        with open(path) as f:
            for line in f.readlines():
                if not line.strip():
                    break

                jsonStr = json.loads(line)
                query = jsonStr['query']
                qid = jsonStr['qid']
                tid = jsonStr['docid'] # docid == talbe[tid]
                rel = jsonStr['rel']

                if qid not in query_dict:
                    query_tokenized = query_tokenizer.encode_plus(query,
                                                                  max_length=max_query_length,
                                                                  padding='max_length',
                                                                  truncation=True,
                                                                  return_tensors="pt"
                                                                  )
                    query_dict[qid] = query_tokenized

                if tid not in table_dict:
                    table_reps, caption_rep = encode_tables(jsonStr, self.is_slice, table_tokenizer, min_row)
                    table_dict[tid] = (table_reps, [caption_rep] * len(table_reps))
               
                if rel > 0:
                    pos_rel_dict[qid].append(tid)
                else:
                    neg_rel_dict[qid].append(tid)
                # data[qid].append((query_dict[qid], table_reps, [caption_rep] * len(table_reps), rel))

        # sort by rel
        for qid in pos_rel_dict:
            pos_rel_dict[qid].sort(key=lambda tup: tup[1], reverse=True)

        for qid in neg_rel_dict:
            neg_rel_dict[qid].sort(key=lambda tup: tup[1], reverse=True)

        # Save 
        with open(os.path.join(processed_dir, self.query_file), 'wb') as f:
            torch.save(query_dict, f)

        with open(os.path.join(processed_dir, self.table_file), 'wb') as f:
            torch.save(table_dict, f)

        with open(os.path.join(processed_dir, self.pos_rel_file), 'wb') as f:
            torch.save(pos_rel_dict, f)

        with open(os.path.join(processed_dir, self.neg_rel_file), 'wb') as f:
            torch.save(neg_rel_dict, f)
        print('Done!')


    @property
    def processed_folder(self):
        return os.path.join(self.data_dir, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.table_file))


def triple_collate_function(pos_rel_dict):
    def collate_function(batch):
        query_ids, queries, table_ids, tables, hard_table_ids, hard_tables = zip(*batch)
        input_ids, token_type_ids, attention_mask = [], [], []
        for q in queries:
            input_ids.append(q["input_ids"].squeeze())
            token_type_ids.append(q["token_type_ids"].squeeze())
            attention_mask.append(q["attention_mask"].squeeze())

        query = {"input_ids": torch.stack(input_ids),
                 "token_type_ids": torch.stack(token_type_ids),
                 "attention_mask": torch.stack(attention_mask)}

        tables = list(zip(*tables))
        hard_tables = list(zip(*hard_tables))

        # rearranged_tables = []
        # for items in zip(*tables):
        #     rearranged_tables.append([item[0] for item in items])

        # rearranged_hard_tables = []
        # for items in zip(*hard_tables):
        #     rearranged_hard_tables.append([item[0] for item in items])

        rel_pair_mask = [[1 if tid not in pos_rel_dict[qid] else 0 for tid in table_ids] for qid in query_ids]
        hard_pair_mask = [[1 if tid not in pos_rel_dict[qid] else 0 for tid in hard_table_ids] for qid in query_ids]
        return query, tables, hard_tables, torch.FloatTensor(rel_pair_mask), torch.FloatTensor(hard_pair_mask)
    return collate_function 


def query_table_collate_fn(batch):
    query, tables, caption, rel = zip(*batch)
    input_ids, token_type_ids, attention_mask = [], [], []
    for q in query:
        input_ids.append(q["input_ids"].squeeze())
        token_type_ids.append(q["token_type_ids"].squeeze())
        attention_mask.append(q["attention_mask"].squeeze())

    query = {"input_ids": torch.stack(input_ids),
             "token_type_ids": torch.stack(token_type_ids),
             "attention_mask": torch.stack(attention_mask)}

    return query, tables, caption, torch.Tensor(rel)


def infer_column_type(value):
    if not value:
        return ''
    elif value.replace('.','').replace(',','').replace('-','').isdigit():
        return 'real'
    return 'text'


class TableDataset(Dataset):
    def __init__(self, data_dir: str = '.data', data_type: str = 'test', table_tokenizer=None, 
                 min_row=10, prepare=False, is_slice=True):
        self.data_dir = data_dir
        self.table_file = f'{data_type}_{min_row}.table'
        self.is_slice = is_slice

        if prepare:
            self.prepare(data_type, table_tokenizer, min_row)

        self.tables = torch.load(os.path.join(self.processed_folder, self.table_file))

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, index):
        return self.tables[index]

    def prepare(self, data_type, table_tokenizer, min_row):
        if self._check_exists():
            return

        processed_dir = Path(self.processed_folder)
        processed_dir.mkdir(exist_ok=True)
        if not table_tokenizer:
            raise RuntimeError('Tokenizers are not found.' +
                               ' You must set table_tokenizer')
        # print('Processing...')

        tables = []
        path = Path(self.data_dir + '/' + data_type + '.jsonl')

        with open(path) as f:
            for line in f.readlines():
                if not line.strip():
                    break

                # 테이블 기본 Meta data 파싱
                jsonStr = json.loads(line)
                tableId = jsonStr['docid']    # tableId -> tid
                query = jsonStr['query']
                qid = jsonStr['qid']
                rel = jsonStr['rel']

                # Table Encode
                table_reps, caption_rep = encode_tables(jsonStr, self.is_slice, table_tokenizer, min_row)
                tables.append((f"{tableId}", table_reps, [caption_rep] * len(table_reps)))

        # Save
        with open(os.path.join(processed_dir, self.table_file), 'wb') as f:
            torch.save(tables, f)

    @property
    def processed_folder(self):
        return os.path.join(self.data_dir, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.table_file))


def table_collate_fn(batch):
    tid, tables, caption = zip(*batch)
    return tid, tables, caption


class QueryDataset(Dataset):
    def __init__(self, data_dir: str = '.data', data_type: str = 'test', query_tokenizer=None, 
                 max_query_length=7, prepare=False):
        self.data_dir = data_dir
        self.query_file = data_type + '.query'

        if prepare:
            self.prepare(data_type, query_tokenizer, max_query_length)

        self.queries = torch.load(os.path.join(self.processed_folder, self.query_file))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        return self.queries[index]

    def prepare(self, data_type, query_tokenizer, max_query_length):
        if self._check_exists():
            return

        processed_dir = Path(self.processed_folder)
        processed_dir.mkdir(exist_ok=True)
        if not query_tokenizer:
            raise RuntimeError('Tokenizers are not found.' +
                               ' You must set query_tokenizer')
        print('Processing...')
        query_dict = dict()
        path = Path(self.data_dir + '/' + data_type + '.jsonl')

        with open(path) as f:
            for line in f.readlines():
                if not line.strip():
                    break

                # 테이블 기본 Meta data 파싱
                jsonStr = json.loads(line)
                query = jsonStr['query']
                qid = jsonStr['qid']

                if qid not in query_dict:
                    query_tokenized = query_tokenizer.encode_plus(query,
                                                                  max_length=max_query_length,
                                                                  padding='max_length',
                                                                  truncation=True,
                                                                  return_tensors="pt"
                                                                  )
                    query_dict[qid] = query_tokenized  # BERT **input input_ids, seg_ids, mas_ids

        queries = list(query_dict.items())
        # Save
        with open(os.path.join(processed_dir, self.query_file), 'wb') as f:
            torch.save(queries, f)
        print('Done!')

    @property
    def processed_folder(self):
        return os.path.join(self.data_dir, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.query_file))


def query_collate_fn(batch):
    qid, query = zip(*batch)

    input_ids, token_type_ids, attention_mask = [], [], []
    for q in query:
        input_ids.append(q["input_ids"].squeeze())
        token_type_ids.append(q["token_type_ids"].squeeze())
        attention_mask.append(q["attention_mask"].squeeze())

    query = {"input_ids": torch.stack(input_ids),
             "token_type_ids": torch.stack(token_type_ids),
             "attention_mask": torch.stack(attention_mask)}

    return qid, query 


if __name__ == "__main__":
    table_model = TableBertModel.from_pretrained('model/tabert_base_k3/model.bin')
    query_tokenizer = table_model.tokenizer
    table_tokenizer = table_model.tokenizer

    dataset = QueryTableDataset(data_dir='data/1',
                                data_type='train',
                                query_tokenizer=query_tokenizer,
                                table_tokenizer=table_tokenizer,
                                prepare=True,
                                )
    data_collator = triple_collate_function(dataset.pos_rel)
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            collate_fn=data_collator)

    for d in dataloader:
        print(d)
        break
