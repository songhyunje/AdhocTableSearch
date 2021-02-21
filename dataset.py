import itertools
import json
import os
from math import ceil
from collections import defaultdict
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

from table_bert import Table, Column, TableBertModel


html_pattern = re.compile(r'<\w+ [^>]*>([^<]+)</\w+>')
tag_pattern = re.compile(r'<.*?>')
link_pattern = re.compile(r'\[.*?\|.*?\]')


class QueryTableDataset(Dataset):
    def __init__(self, data_dir: str = '.data', data_type: str = 'train',
                 query_tokenizer=None, table_tokenizer=None, max_query_length=15,
                 min_rows=10, max_tables=10,
                 prepare=False, is_slice=True):
        self.data_dir = data_dir
        self.ids_file = f'{data_type}_{min_rows}_{max_tables}.pair'
        self.data_type = data_type  # test, train 구분하기위해
        self.is_slice = is_slice
        if prepare:
            self.prepare(data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length,
                         min_rows=min_rows, max_tables=max_tables)

        self.data = torch.load(os.path.join(self.processed_folder, self.ids_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def prepare(self, data_dir, data_type, query_tokenizer, table_tokenizer, max_query_length, min_rows, max_tables):
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
                numericIdx = raw_json['numericColumns']

                # Heading preprocessing + link remove
                heading_str = ' '.join(heading)
                if html_pattern.search(heading_str):
                    if link_pattern.search(heading_str): # 같이 있는 경우 
                        heading = [re.sub(tag_pattern, '', column).strip() for column in heading]
                        for idx, column in enumerate(heading):
                            if link_pattern.search(column):
                                real_text = link_pattern.search(column).group().split('|')[-1][:-1].strip()
                                heading[idx] = real_text
                    else:
                        heading = [re.sub(html_pattern, '', column).strip() for column in heading]
                
                # Row preporcessing + link remove 

                cell_sum_str = ''
                for rows in body:
                    cell_sum_str += ' '.join(rows)

                if html_pattern.search(cell_sum_str):
                    if link_pattern.search(cell_sum_str): # 같이 있으면                    
                        for i, rows in enumerate(body):
                            for j, cell in enumerate(rows):
                                if link_pattern.search(cell):
                                    cell = re.sub(tag_pattern, '', cell).strip()
                                    real_text = link_pattern.search(cell).group().split('|')[-1][:-1]
                                    body[i][j] = real_text
                                else:
                                    cell = re.sub(html_pattern, '', cell).strip()
                                    body[i][j] = cell

                    else:
                        row_list = []
                        for rows in body:
                            row_list.append([re.sub(html_pattern, '', row).strip() for row in rows])
                        body = row_list

                if col == 0 or row == 0:
                    continue

                # Infer column type
                heading_type = infer_column_type_from_row_values(numericIdx, heading, body)

                # TODO: caption을 다양하게 주는부분, 비교실험 해볼부분임
                caption = " ".join(heading) + " " + title + " " + secTitle + " " + caption
                caption_rep = table_tokenizer.tokenize(caption)

                if self.is_slice:
                    column_reps = slice_table(title, heading, body, table_tokenizer, heading_type, min_rows, max_tables)
                else:
                    column_reps = [Table(id=title,
                                       header=[Column(h.strip(), heading_type.get(h, 'text')) for h in heading],
                                       data=body
                                       ).tokenize(table_tokenizer)]
                rel = 1 if int(rel) > 0 else 0
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


def slice_table(title, heading, datas, table_tokenizer, heading_type, min_rows=10, max_table_nums=10):
    """
    min_rows: 최소 행 개수
    max_table_nums: 최대 테이블 개수

    시나리오, 최소행 = 5, 최대테이블 = 10 이라고 할 때 
    30행 테이블은 => 5행테이블 x 6개로 쪼개져야하고 
    300행은 => 5행 x 60개가 아닌, 30행 x 10개로 쪼개져야함 
    """

    table_rep_list = []
    if len(datas) <= min_rows:  # 애초에 테이블이 최소행 보다 작은 경우
        column_rep = Table(id=title,
                           header=[Column(h.strip(), heading_type.get(h, 'text')) for h in heading],
                           data=datas
                           ).tokenize(table_tokenizer)
        table_rep_list.append(column_rep)
    else:
        row_n = max(min_rows, ceil(len(datas) / max_table_nums))
        slice_row_data = [datas[i * row_n:(i + 1) * row_n] for i in range((len(datas) + row_n - 1) // row_n)]
        for rows in slice_row_data:
            column_rep = Table(id=title,
                               header=[Column(h.strip(), heading_type.get(h, 'text')) for h in heading],
                               data=rows
                               ).tokenize(table_tokenizer)
            table_rep_list.append(column_rep)

    return table_rep_list


def infer_column_type_from_row_values(numeric_idx_list, heading, body):
    heading_type = {k : 'text' for k in heading}
    for n_idx in numeric_idx_list:
        heading_type[heading[n_idx]] = 'real'
        for i, rows in enumerate(body):
            try:
                float(rows[n_idx].strip().replace('−','-').replace(',','').replace('–','-'))
            except:
                heading_type[heading[n_idx]] = 'text'
                break
    return heading_type


class TableDataset(Dataset):
    def __init__(self, data_dir: str = '.data', data_type: str = 'test', table_tokenizer=None, 
                 min_rows=10, max_tables=10,
                 prepare=False, is_slice=True):
        self.data_dir = data_dir
        self.table_file = f'{data_type}_{min_rows}_{max_tables}.table'
        self.is_slice = is_slice

        if prepare:
            self.prepare(data_type, table_tokenizer, min_rows, max_tables)

        self.tables = torch.load(os.path.join(self.processed_folder, self.table_file))

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, index):
        return self.tables[index]

    def prepare(self, data_type, table_tokenizer, min_rows, max_tables):
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
                tableId = jsonStr['docid']

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
                numericIdx = raw_json['numericColumns']

                # Heading preprocessing + link remove
                heading_str = ' '.join(heading)
                if html_pattern.search(heading_str):
                    if link_pattern.search(heading_str): # 같이 있는 경우 
                        heading = [re.sub(tag_pattern, '', column).strip() for column in heading]
                        for idx, column in enumerate(heading):
                            if link_pattern.search(column):
                                real_text = link_pattern.search(column).group().split('|')[-1][:-1].strip()
                                heading[idx] = real_text
                    else:
                        heading = [re.sub(html_pattern, '', column).strip() for column in heading]
                
                # Row preporcessing + link remove 

                cell_sum_str = ''
                for rows in body:
                    cell_sum_str += ' '.join(rows)

                if html_pattern.search(cell_sum_str):
                    if link_pattern.search(cell_sum_str): # 같이 있으면                    
                        for i, rows in enumerate(body):
                            for j, cell in enumerate(rows):
                                if link_pattern.search(cell):
                                    cell = re.sub(tag_pattern, '', cell).strip()
                                    real_text = link_pattern.search(cell).group().split('|')[-1][:-1]
                                    body[i][j] = real_text
                                else:
                                    cell = re.sub(html_pattern, '', cell).strip()
                                    body[i][j] = cell

                    else:
                        row_list = []
                        for rows in body:
                            row_list.append([re.sub(html_pattern, '', row).strip() for row in rows])
                        body = row_list

                if col == 0:
                    heading = ['']
                if row == 0:
                    body = [['']]

                # Infer column type
                heading_type = infer_column_type_from_row_values(numericIdx, heading, body)

                if self.is_slice:
                    column_reps = slice_table(title, heading, body, table_tokenizer, heading_type, min_rows, max_tables)

                else:
                    column_reps = [Table(id=title,
                                       header=[Column(h.strip(), heading_type.get(h, 'text')) for h in heading],
                                       data=body
                                       ).tokenize(table_tokenizer)]

                caption = " ".join(heading) + " " + title + " " + secTitle + " " + caption
                caption_rep = table_tokenizer.tokenize(caption)
                for i, column_rep in enumerate(column_reps, 1):
                    tables.append([f"{tableId}-{i}", column_rep, caption_rep])

        # Save
        with open(os.path.join(processed_dir, self.table_file), 'wb') as f:
            torch.save(tables, f)
        # print('Done!')

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
                 max_query_length=15, prepare=False):
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
    # bert_model = BertModel.from_pretrained('bert-base-uncased')
    table_model = TableBertModel.from_pretrained('model/tabert_base_k3/model.bin')
    table_tokenizer = table_model.tokenizer
    query_tokenizer = table_tokenizer

    dataset = QueryTableDataset(data_dir='data/1',
                                data_type='train',
                                query_tokenizer=query_tokenizer,
                                table_tokenizer=table_tokenizer,
                                prepare=True,
                                )
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            collate_fn=query_table_collate_fn)

    for _ in range(1):
        for d in dataloader:
            print(d)
            break

    table_dataset = TableDataset(data_dir='data/1', 
                                 data_type='train',
                                 table_tokenizer=table_tokenizer,
                                 prepare=True,
                                 )
    dataloader = DataLoader(table_dataset,
                            batch_size=4,
                            collate_fn=table_collate_fn)

    for _ in range(1):
        for d in dataloader:
            print(d)
            break

    query_dataset = QueryDataset(data_dir='data/1', 
                                 data_type='test',
                                 query_tokenizer=query_tokenizer,
                                 prepare=True,
                                 )
    dataloader = DataLoader(query_dataset,
                            batch_size=4,
                            collate_fn=query_collate_fn)

    for _ in range(1):
        for d in dataloader:
            print(d)
            break

