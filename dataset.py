import itertools
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

from table_bert import Table, Column, TableBertModel


class Sample(object):
    def __init__(self, query, positive_tables, negative_tables):
        self.query = query
        self.positive_tables = positive_tables
        self.negative_tables = negative_tables


class QueryTableDataset(Dataset):
    def __init__(self, data_dir: str = '.data', data_type: str = 'train',
                 query_tokenizer=None, table_tokenizer=None, max_query_length=15,
                 prepare=False, is_slice=True):
        self.data_dir = data_dir
        self.query_file = data_type + '.query'
        self.table_file = data_type + '.table'
        self.ids_file = data_type + '.pair'
        self.data_type = data_type  # test, train 구분하기위해
        self.is_slice = is_slice
        if prepare:
            self.prepare(data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length)

        self.data = torch.load(os.path.join(self.processed_folder, self.ids_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def slice_table(self, title, heading, datas, table_tokenizer):
        table_rep_list = []
        row_n = 5 # 평균행이 약 13개

        # n-Row slice
        if len(datas) > row_n:
            slice_row_data = [datas[i * row_n:(i + 1) * row_n] for i in range((len(datas) + row_n - 1) // row_n)]
            for row in slice_row_data:
                column_rep = Table(id=title,
                                   header=[Column(h.strip(), 'text') for h in heading],
                                   data=row
                                   ).tokenize(table_tokenizer)
                table_rep_list.append(column_rep)

        else:
            column_rep = Table(id=title,
                               header=[Column(h.strip(), 'text') for h in heading],
                               data=datas
                               ).tokenize(table_tokenizer)
            table_rep_list.append(column_rep)

        #1-Col Slice
        for idx, col in enumerate(heading):
            trans_heading = [col] * len(datas) # 행 개수 만큼 Heading을 [col, col, ... 이렇게 펴줌 ]
            trans_row = [ [row[idx] for row in datas] ] # 각 열의 해당하는 row data를 가로로
            column_rep = Table(id=title,
                               header=[Column(h.strip(), 'text') for h in trans_heading],
                               data=trans_row
                               ).tokenize(table_tokenizer)
            table_rep_list.append(column_rep)

        # TODO: 만약 Col-1개 짜른것의 성능이 너무 안 좋으면 데이터를 보고 Multi-col 답변이 많은지를 분석후 코드 추가
        # multi-Col Slice
        # row_c = 2  # 평균열이 약 5개
        # if len(heading) > 2:
        #     slice_col_data = [heading[i * row_c:(i + 1) * row_c] for i in range((len(heading) + row_c - 1) // row_c)]
        #     col_idx = 0
        #     for col in slice_col_data:
        #         row_data = [row[col_idx:len(col)] for row in datas]
        #         col_idx += len(col)
        #         try:
        #             column_rep = Table(id=title,
        #                                header=[Column(h.strip(), 'text') for h in col],
        #                                data=row_data
        #                                ).tokenize(table_tokenizer)
        #         except:
        #             # Col의 모든 Row가 빈값이면 에러남
        #             continue
        #         table_rep_list.append(column_rep)
        # else:
        #     column_rep = Table(id=title,
        #                        header=[Column(h.strip(), 'text') for h in heading],
        #                        data=datas
        #                        ).tokenize(table_tokenizer)
        #     table_rep_list.append(column_rep)
        return table_rep_list

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
        data = []
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
                    query_dict[qid] = query_tokenized  # BERT **input input_ids, seg_ids, mas_ids

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
                if self.is_slice:
                    column_reps = self.slice_table(title, heading, body, table_tokenizer)
                    #print(f"row : {row}, col : {col}, slcie : {len(slice_tables)}")
                else:
                    column_reps = [Table(id=title,
                                       header=[Column(h.strip(), 'text') for h in heading],
                                       data=body
                                       ).tokenize(table_tokenizer)]
                # TODO: caption을 다양하게 주는부분, 비교실험 해볼부분임
                caption = " ".join(heading) + " " + title + " " + secTitle + " " + caption
                caption_rep = table_tokenizer.tokenize(caption)

                if rel == '0':
                    for column_rep in column_reps:
                        neg_tables[qid].append((column_rep, caption_rep))
                else:
                    for column_rep in column_reps:
                        pos_tables[qid].append((column_rep, caption_rep))

        for qid in query_dict:
            if not pos_tables[qid]:
                continue

            for t in itertools.product(pos_tables[qid], neg_tables[qid]):
                data.append([query_dict[qid]] + list(itertools.chain.from_iterable(t)))

        # Save
        with open(os.path.join(processed_dir, self.ids_file), 'wb') as f:
            torch.save(data, f)
        print('Done!')

    @property
    def processed_folder(self):
        return os.path.join(self.data_dir, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.ids_file))


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


class QueryTablePredictionDataset(Dataset):
    def __init__(self, data_dir: str = '.data', data_type: str = 'test',
                 query_tokenizer=None, table_tokenizer=None, max_query_length=15,
                 prepare=False, is_slice=True):
        self.data_dir = data_dir
        self.query_file = data_type + '.query'
        self.table_file = data_type + '.table'
        self.ids_file = data_type + '.pair'
        self.is_slice = is_slice

        if prepare:
            self.prepare(data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length)

        self.pair_ids = torch.load(os.path.join(self.processed_folder, self.ids_file))

    def __len__(self):
        return len(self.pair_ids)

    def __getitem__(self, index):
        return self.pair_ids[index]

    def slice_table(self, title, heading, datas, table_tokenizer):
        table_rep_list = []
        row_n = 5  # 평균행이 약 13개

        # n-Row slice
        if len(datas) > row_n:
            slice_row_data = [datas[i * row_n:(i + 1) * row_n] for i in range((len(datas) + row_n - 1) // row_n)]
            for row in slice_row_data:
                column_rep = Table(id=title,
                                   header=[Column(h.strip(), 'text') for h in heading],
                                   data=row
                                   ).tokenize(table_tokenizer)
                table_rep_list.append(column_rep)

        else:
            column_rep = Table(id=title,
                               header=[Column(h.strip(), 'text') for h in heading],
                               data=datas
                               ).tokenize(table_tokenizer)
            table_rep_list.append(column_rep)

        # 1-Col Slice
        for idx, col in enumerate(heading):
            trans_heading = [col] * len(datas)  # 행 개수 만큼 Heading을 [col, col, ... 이렇게 펴줌 ]
            trans_row = [[row[idx] for row in datas]]  # 각 열의 해당하는 row data를 가로로
            column_rep = Table(id=title,
                               header=[Column(h.strip(), 'text') for h in trans_heading],
                               data=trans_row
                               ).tokenize(table_tokenizer)
            table_rep_list.append(column_rep)

        # TODO: 만약 Col-1개 짜른것의 성능이 너무 안 좋으면 데이터를 보고 Multi-col 답변이 많은지를 분석후 코드 추가
        # multi-Col Slice
        # row_c = 2  # 평균열이 약 5개
        # if len(heading) > 2:
        #     slice_col_data = [heading[i * row_c:(i + 1) * row_c] for i in range((len(heading) + row_c - 1) // row_c)]
        #     col_idx = 0
        #     for col in slice_col_data:
        #         row_data = [row[col_idx:len(col)] for row in datas]
        #         col_idx += len(col)
        #         try:
        #             column_rep = Table(id=title,
        #                                header=[Column(h.strip(), 'text') for h in col],
        #                                data=row_data
        #                                ).tokenize(table_tokenizer)
        #         except:
        #             # Col의 모든 Row가 빈값이면 에러남
        #             continue
        #         table_rep_list.append(column_rep)
        # else:
        #     column_rep = Table(id=title,
        #                        header=[Column(h.strip(), 'text') for h in heading],
        #                        data=datas
        #                        ).tokenize(table_tokenizer)
        #     table_rep_list.append(column_rep)
        return table_rep_list



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
                    query_dict[qid] = query_tokenized  # BERT **input input_ids, seg_ids, mas_ids

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

                if self.is_slice:
                    column_reps = self.slice_table(title, heading, body, table_tokenizer)
                    #print(f"row : {row}, col : {col}, slcie : {len(slice_tables)}")
                else:
                    column_reps = [Table(id=title,
                                       header=[Column(h.strip(), 'text') for h in heading],
                                       data=body
                                       ).tokenize(table_tokenizer)]

                caption = " ".join(heading) + " " + title + " " + secTitle + " " + caption
                caption_rep = table_tokenizer.tokenize(caption)
                for column_rep in column_reps:
                    pairs.append([qid, query_dict[qid], tableId, column_rep, caption_rep, rel])

        # Save
        with open(os.path.join(processed_dir, self.ids_file), 'wb') as f:
            torch.save(pairs, f)
        print('Done!')

    @property
    def processed_folder(self):
        return os.path.join(self.data_dir, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.ids_file))


def query_table_prediction_collate_fn(batch):
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

    dataset = QueryTableDataset(data_dir='data/1',
                                data_type='train',
                                query_tokenizer=query_tokenizer,
                                table_tokenizer=table_tokenizer,
                                prepare=True,
                                )
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            collate_fn=query_table_collate_fn)

    for _ in range(1):
        for d in dataloader:
            print(d)
            break
