#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 FAISS-based index components for dense retriver
"""

import os
import time
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import faiss
import torch

from model import QueryTableMatcher
from transformers import BertTokenizer
from table_bert import Table, Column


def faiss_search(args):
    # Model, BERT loader
    query_tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = QueryTableMatcher(args)
    model = model.load_from_checkpoint(
        checkpoint_path=args.ckpt_file,
    )

    # Data loader
    data_path = args.data_dir
    _table_encode(data_path=data_path, table_encoder=model)
    tables = torch.load(os.path.join(data_path, 'processed/table.idx'))
    db_idxs = torch.load(os.path.join(data_path, 'processed/db.idx'))

    # Indexing
    torch_tables = []
    for table in tables:
        torch_tables.append(table[0].squeeze())
    tables = torch.stack(torch_tables)
    tables = tables.detach().numpy()
    index = faiss.IndexFlatL2(768) # base L2 distance
    index.add(tables)

    # Query encoding
    # Sanity Check
    query_tokenized = query_tokenizer.encode_plus('pga leaderboard',
                                                  max_length=15,
                                                  padding='max_length',
                                                  truncation=True,
                                                  return_tensors="pt"
                                                  )
    qCLS = model.Qmodel(**query_tokenized)[1].detach().numpy()

    # knn Search
    st = time.time()
    scores, indexes = index.search(qCLS, args.topk)
    print(f">>> Search time ... {time.time() - st} sec ")
    db_ids = [[db_idxs[i] for i in query_top_idxs] for query_top_idxs in indexes]
    result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
    print(result[:5])
    return result


def _table_encode(data_path = './data/', table_encoder = None):
    if os.path.exists(os.path.join(data_path, 'processed/table.idx')) \
            and os.path.exists(os.path.join(data_path, 'processed/db.idx')):
        return
    else:
        path = Path(data_path + 'train.jsonl')
        data = []
        tableid_list = []
        with open(path) as f:
            for line in tqdm(f.readlines()[:100]):
                if not line.strip():
                    break
                # 테이블 기본 Meta data 파싱
                jsonStr = json.loads(line)
                tableId = jsonStr['docid']
                query = jsonStr['query']
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

                tableid_list.append(tableId)
                column_rep = Table(id=title,
                                   header=[Column(h.strip(), 'text') for h in heading],
                                   data=body
                                   ).tokenize(table_encoder.Tmodel.tokenizer)
                caption = " ".join(heading) + " " + title + " " + secTitle + " " + caption
                caption_rep = table_encoder.Tmodel.tokenizer.tokenize(caption)

                context_encoding, column_encoding, _ = table_encoder.Tmodel.encode(contexts=[caption_rep], tables=[column_rep])
                tp_concat_encoding = torch.mean(context_encoding, dim=1) + torch.mean(column_encoding, dim=1)
                data.append(tp_concat_encoding)
        tableid_dict = dict((idx, tid) for idx, tid in enumerate(tableid_list))
        # Save
        with open(os.path.join(data_path, 'processed/table.idx'), 'wb') as f:
            torch.save(data, f)

        with open(os.path.join(data_path, 'processed/db.idx'), 'wb') as f:
            torch.save(tableid_dict, f)
        print('Done!')



def add_generic_arguments(parser):
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--ckpt_file", default=None, type=str, required=True,
                        help="The ckpt file ")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--topk", type=int)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_arguments(parser)
    QueryTableMatcher.add_model_specific_args(parser)
    args = parser.parse_args()
    # predict
    faiss_search(args)

