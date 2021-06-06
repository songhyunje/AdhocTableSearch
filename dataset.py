import itertools
import json
import os
import random
from math import ceil
from collections import defaultdict
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader, Dataset

from table_bert import Table, Column, TableBertModel


# html_pattern = re.compile(r'<\w+ [^>]*>([^<]+)</\w+>')
# tag_pattern = re.compile(r'<.*?>')
# link_pattern = re.compile(r'\[.*?\|.*?\]')


# def get_negative_rank(path=Path('.data/bench/1/rel0_rank'), threshold=0.0811):
#     rank_dict = {}
#     with open(path, 'r') as f:
#         lines = f.readlines()[:3200]
#         for line in lines:
#             qid, rel, tid, overlap_ratio = line.split('\t')
#             if float(overlap_ratio) >= threshold:
#                 try:
#                     rank_dict[qid].append(tid)
#                 except:
#                     rank_dict[qid] = [tid]
#     return rank_dict


def encode_tables(table_json, is_slice, query, table_tokenizer, min_row):
    rel = table_json['rel']
    raw_json = json.loads(table_json['table']['raw_json'])

    textBeforeTable = raw_json['textBeforeTable']  # 추후
    textAfterTable = raw_json['textAfterTable']    # 추후

    title = raw_json['pageTitle']
    # caption = re.sub(r'[^a-zA-Z0-9]', ' ', raw_json['title']).strip()  # Caption 역할
    caption = raw_json['title'].strip()  # Caption 역할
    tableOrientation = raw_json['tableOrientation'].strip()  # [HORIZONTAL, VERTICAL]

    headerPosition = raw_json['headerPosition']  # ['FIRST_ROW', 'MIXED', 'FIRST_COLUMN', 'NONE’]
    hasHeader = raw_json['hasHeader']            # [true, false]
    keyColumnIndex = raw_json['keyColumnIndex']
    headerRowIndex = raw_json['headerRowIndex']  # 0 == 첫줄, -1 == 없음
    entities = raw_json['entities']

    body = raw_json['relation']
    if tableOrientation == "HORIZONTAL":
        # col_cnt = len(table_data)
        # row_cnt = len(table_data[0])

        # for row in range(row_cnt):
        #     tmp_row_data = []
        #     for col in range(col_cnt):
        #         tmp_row_data.append(table_data[col][row])
        #     body.append(tmp_row_data)
        body = list(map(list, zip(*body)))  # transpose

        # Header
        # for table_col in table_data:
        #     heading.append(table_col[0])

    # else:  # tableOrientation == "VERTICAL":
    #     body = table_data
        # col_cnt = len(table_data)
        # row_cnt = len(table_data[0])

        # for row in range(row_cnt):
        #     tmp_row_data = []
        #     for col in range(col_cnt):
        #         tmp_row_data.append(table_data[col][row])
        #     body.append(tmp_row_data)

        # Header
        # for table_col in table_data:
        #     heading.append(table_col[0])

    header = body[headerRowIndex] if hasHeader else [''] * len(body[0])

    # Heading preprocessing + link remove
    # heading_str = ' '.join(heading)
    # if html_pattern.search(heading_str):
    #     if link_pattern.search(heading_str):  # 같이 있는 경우
    #         heading = [re.sub(tag_pattern, '', column).strip() for column in heading]
    #         for idx, column in enumerate(heading):
    #             if link_pattern.search(column):
    #                 real_text = link_pattern.search(column).group().split('|')[-1][:-1].strip()
    #                 heading[idx] = real_text
    #     else:
    #         heading = [re.sub(html_pattern, '', column).strip() for column in heading]

    # Row preporcessing + link remove
    # cell_sum_str = ''
    # for rows in body:
    #     cell_sum_str += ' '.join(rows)
    # cell_sum_str = ' '.join([row for rows in body])

    # if html_pattern.search(cell_sum_str):
    #     if link_pattern.search(cell_sum_str):  # 같이 있으면
    #         for i, rows in enumerate(body):
    #             for j, cell in enumerate(rows):
    #                 if link_pattern.search(cell):
    #                     cell = re.sub(tag_pattern, '', cell).strip()
    #                     real_text = link_pattern.search(cell).group().split('|')[-1][:-1]
    #                     body[i][j] = real_text
    #                 else:
    #                     cell = re.sub(html_pattern, '', cell).strip()
    #                     body[i][j] = cell

    #     else:
    #         row_list = []
    #         for rows in body:
    #             row_list.append([re.sub(html_pattern, '', row).strip() for row in rows])
    #         body = row_list

    caption = caption if caption else title
    caption_rep = table_tokenizer.tokenize(caption)

    if is_slice:
        table_reps = slice_table(title, header, body, caption, table_tokenizer, min_row)
    else:
        table_reps = [Table(id=caption,
                            header=[Column(h.strip(), infer_column_type(h)) for h in header],
                            data=body
                           ).tokenize(table_tokenizer)]

    # memory issues!
    if len(table_reps) > 3:# 
        table_reps = table_reps[:3]

    return caption_rep, table_reps


def slice_table(title, heading, data, caption, table_tokenizer, min_row):
    table_reps = []

    # min_row = 20         # 최소 5개의 행은 있어야 함
    # max_table_nums = 10  # 테이블은 최대 10개로 나뉘어짐
    for i in range(0, len(data), min_row):
        rows = data[i:i+min_row]
        table_rep = Table(id=title,
                           header=[Column(h.strip(), infer_column_type(h)) for h in heading],
                           data=rows
                           ).tokenize(table_tokenizer)
        table_reps.append(table_rep)

    # if len(data) <= min_row:  # 테이블이 최소행 보다 작은 경우
    #     column_rep = Table(id=title,
    #                        header=[Column(h.strip(), infer_column_type(h)) for h in heading],
    #                        data=data
    #                        ).tokenize(table_tokenizer)
    #     table_rep_list.append(column_rep)
    # else:
    #     row_n = max(min_row, ceil(len(data) / max_table))
    #     slice_row_data = [data[i * row_n:(i + 1) * row_n] for i in range((len(data) + row_n - 1) // row_n)]
    #     for rows in slice_row_data:
    #         column_rep = Table(id=title,
    #                            header=[Column(h.strip(), infer_column_type(h)) for h in heading],
    #                            data=rows
    #                            ).tokenize(table_tokenizer)
    #         table_rep_list.append(column_rep)

    # if len(data) <= min_row:  # 테이블이 최소행 보다 작은 경우
    #     column_rep = Table(id=title,
    #                        header=[Column(h.strip(), infer_column_type(h)) for h in heading],
    #                        data=data
    #                        ).tokenize(table_tokenizer)
    #     table_rep_list.append(column_rep)
    # else:
    #     row_n = max(min_row, ceil(len(data) / max_table))
    #     slice_row_data = [data[i * row_n:(i + 1) * row_n] for i in range((len(data) + row_n - 1) // row_n)]
    #     for rows in slice_row_data:
    #         column_rep = Table(id=title,
    #                            header=[Column(h.strip(), infer_column_type(h)) for h in heading],
    #                            data=rows
    #                            ).tokenize(table_tokenizer)
    #         table_rep_list.append(column_rep)

        # row_n = max(min_row, ceil(len(data) / max_table_nums))
        # slice_row_data = [data[i * row_n:(i + 1) * row_n] for i in range((len(data) + row_n - 1) // row_n)]
        # if rel == 0:  # Negative
        #     for rows in slice_row_data:
        #         column_rep = Table(id=title,
        #                            header=[Column(h.strip(), infer_column_type(h)) for h in heading],
        #                            data=rows
        #                            ).tokenize(table_tokenizer)
        #         table_rep_list.append((rel, column_rep))

        # else:  # Positive
        #     query_tokens = [token.strip() for token in query.split(' ')]
        #     is_always_postive = False
        #     for token in query_tokens:
        #         if token in caption:
        #             is_always_postive = True
        #             break

        #     if is_always_postive:  # caption에 포함되어있는 경우
        #         for rows in slice_row_data:
        #             column_rep = Table(id=title,
        #                                header=[Column(h.strip(), 'text') for h in heading],
        #                                data=rows
        #                                ).tokenize(table_tokenizer)
        #             table_rep_list.append((rel, column_rep))
        #     else:
        #         for rows in slice_row_data:
        #             column_rep = Table(id=title,
        #                                header=[Column(h.strip(), 'text') for h in heading],
        #                                data=rows
        #                                ).tokenize(table_tokenizer)
        #             modify_rel = '0'
        #             # Row data를 하나의 string으로
        #             cell_string_sum = ''
        #             for row in rows:
        #                 cell_string_sum += ' '.join(row)
        #             # Query tokens과 overlap
        #             for token in query_tokens:
        #                 if token in cell_string_sum:
        #                     modify_rel = '1'
        #                     break
        #             table_rep_list.append((modify_rel, column_rep))

    return table_reps


class QueryTableDataset(Dataset):
    def __init__(self, data_dir: str = '.data', data_type: str = 'train',
                 query_tokenizer=None, table_tokenizer=None, max_query_length=7,
                 min_row=256, max_tables=2,
                 prepare=False, is_slice=True):
        self.data_dir = data_dir
        self.ids_file = f'{data_type}_{min_row}_{max_tables}.pair'
        self.data_type = data_type
        self.is_slice = is_slice
        if prepare:
            self.prepare(data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length,
                         min_row=min_row, max_tables=max_tables)

        self.data = torch.load(os.path.join(self.processed_folder, self.ids_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def prepare(self, data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length, min_row, max_tables):
        if self._check_exists():
            return

        processed_dir = Path(self.processed_folder)
        processed_dir.mkdir(exist_ok=True)
        if not (query_tokenizer and table_tokenizer):
            raise RuntimeError('Tokenizers are not found.' +
                               ' You must set query_tokenizer and table_tokenizer')
        print('Processing...')

        query_dict = defaultdict()
        data = []
        path = Path(data_dir + '/' + data_type + '.jsonl')

        with open(path) as f:
            for line in f.readlines():
                if not line.strip():
                    break

                # 테이블 기본 Meta data 파싱
                jsonStr = json.loads(line)
                tableId = jsonStr['docid'] # tableId -> tid
                query = jsonStr['query']
                qid = jsonStr['qid']
                tid = jsonStr['docid']
                rel = jsonStr['rel']

                if qid not in query_dict:
                    query_tokenized = query_tokenizer.encode_plus(query,
                                                                  max_length=max_query_length,
                                                                  padding='max_length',
                                                                  truncation=True,
                                                                  return_tensors="pt"
                                                                  )
                    query_dict[qid] = query_tokenized

                caption_rep, column_reps = encode_tables(jsonStr, self.is_slice, query, table_tokenizer, min_row)
                # ret: 0, 1, 2
                rel = 1 if rel > 0 else 0
                data.append((query_dict[qid], column_reps, [caption_rep] * len(column_reps), rel))
                
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
    query, columns, caption, rel = zip(*batch)
    input_ids, token_type_ids, attention_mask = [], [], []
    for q in query:
        input_ids.append(q["input_ids"].squeeze())
        token_type_ids.append(q["token_type_ids"].squeeze())
        attention_mask.append(q["attention_mask"].squeeze())

    query = {"input_ids": torch.stack(input_ids),
             "token_type_ids": torch.stack(token_type_ids),
             "attention_mask": torch.stack(attention_mask)}

    return query, columns, caption, torch.Tensor(rel)


def infer_column_type(value):
    if not value:
        return ''
    elif value.replace('.','').replace(',','').replace('-','').isdigit():
        return 'real'
    return 'text'


class TableDataset(Dataset):
    def __init__(self, data_dir: str = '.data', data_type: str = 'test', table_tokenizer=None, 
                 min_row=10, max_tables=10, prepare=False, is_slice=True):
        self.data_dir = data_dir
        self.table_file = f'{data_type}_{min_row}_{max_tables}.table'
        self.is_slice = is_slice

        if prepare:
            self.prepare(data_type, table_tokenizer, min_row, max_tables)

        self.tables = torch.load(os.path.join(self.processed_folder, self.table_file))

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, index):
        return self.tables[index]

    def prepare(self, data_type, table_tokenizer, min_row, max_tables):
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
                tid = jsonStr['docid']
                rel = jsonStr['rel']

                # Table Encode
                caption_rep, column_reps = encode_tables(jsonStr, self.is_slice, query, table_tokenizer, min_row)
                tables.append((f"{tableId}", column_reps, [caption_rep] * len(column_reps)))
                # tables.append([f"{tableId}", column_reps,  p])

                # for i, column_rep in enumerate(column_reps, 1):
                #     tables.append([f"{tableId}-{i}", column_rep, caption_rep])

        # Save
        with open(os.path.join(processed_dir, self.table_file), 'wb') as f:
            torch.save(tables, f)

    @property
    def processed_folder(self):
        return os.path.join(self.data_dir, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.table_file))


def table_collate_fn(batch):
    tid, column, caption = zip(*batch)
    return tid, column, caption


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
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            collate_fn=query_table_collate_fn)

    for _ in range(1):
        for d in dataloader:
            print(d)
            break
