import argparse
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import QueryTableMatcher
from dataset import TableDataset, QueryDataset, table_collate_fn, query_collate_fn
from faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer
from trec import TREC_evaluator

logger = logging.getLogger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_qrel_dict(args):
    qrel_dict = dict()
    with open(os.path.join(args.data_dir, 'test.jsonl.qrels'))as f:
        for line in f.readlines():
            if line.strip() == '':
                break
            qid, _, tid, rel = line.split('\t')
            try:
                qrel_dict[qid].append([tid, rel.strip(), 0])
            except:
                qrel_dict[qid] = [[tid, rel.strip(), 0]]

    return qrel_dict


def save_TREC_format(args, fn, ranked_dict):
    with open(os.path.join(args.data_dir, fn), 'w', encoding='utf-8') as f:
        for qid, results in sorted(ranked_dict.items(), key=(lambda x: int(x[0]))):
            sorted_results = sorted(results, key = lambda x : x[-1], reverse=True)
            for idx, result in enumerate(sorted_results):
                tid, rel, score = result
                if args.hnsw_index:
                    f.write(f"{qid}\t{0}\t{tid}\t{rel}\t{idx + 1}\tfaiss\n")
                else:
                    f.write(f"{qid}\t{0}\t{tid}\t{rel}\t{score}\tfaiss\n")


def add_generic_arguments(parser):
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir.")
    parser.add_argument("--ckpt_file", default=None, type=str, required=True, help="The ckpt file")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--topk", default=300, type=int)
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for query encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--load_index", action='store_true', help="The index file")
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')


def main(args):
    model = QueryTableMatcher.load_from_checkpoint(args.ckpt_file, map_location=lambda storage, loc: storage.cuda(0))
    model.to(device)
    model.eval()

    # vector_size = model_to_load.get_out_size()
    vector_size = 768
    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size, args.index_buffer)
    else:
        index = DenseFlatIndexer(vector_size, args.index_buffer)

    if args.load_index:
        if args.hnsw_index:
            index.deserialize_from('dtr_hnsw')
        else:
            index.deserialize_from('dtr')
    else:
        table_tokenizer = model.Tmodel.tokenizer
        table_dataset = TableDataset(data_dir=args.data_dir, 
                                     data_type='test',
                                     table_tokenizer=table_tokenizer,
                                     prepare=True,
                                     )
        dataloader = DataLoader(table_dataset,
                                batch_size=args.batch_size,
                                collate_fn=table_collate_fn)

        table_vectors = []
        with torch.no_grad():
            for i, d in enumerate(dataloader, 1):
                table_id, columns, captions = d 
                try:
                    values = model.table_forward(columns, captions)
                except:
                    continue
               
                for tid, vector in zip(table_id, values):
                    table_vectors.append((tid, vector.cpu().numpy())) 
                # print(f"epoch: {i}")
        
        index.index_data(table_vectors)
        if args.hnsw_index:
            index.serialize('dtr_hnsw')
        else:
            index.serialize('dtr')

    query_tokenizer = model.Tmodel.tokenizer
    query_dataset = QueryDataset(data_dir=args.data_dir, 
                                 data_type='test',
                                 query_tokenizer=query_tokenizer,
                                 prepare=True,
                                 )
    dataloader = DataLoader(query_dataset,
                            batch_size=args.batch_size,
                            collate_fn=query_collate_fn)

    qids, query_vectors = [], []
    with torch.no_grad():
        for q in dataloader:
            qid, query = q
            query = {k: v.to(device) for k, v in query.items()}
            vectors = model.query_forward(query)
            query_vectors.extend(vectors.cpu().numpy())
            qids.extend(qid)

    time0 = time.time()
    top_ids_and_scores = index.search_knn(np.array(query_vectors), args.topk)
    # print('index search time: %f sec.', time.time() - time0)

    # evaluation 
    qrel_dict = get_qrel_dict(args)
    result = {}
    for qid, id_and_score in zip(qids, top_ids_and_scores):
        retrieved = {}
        for index, score in zip(id_and_score[0], id_and_score[1]):
            tid = '-'.join(index.split('-')[:3])
            if tid in retrieved:
                continue
            retrieved[tid] = score

        qid_match_table_list = qrel_dict.get(qid, [])
        for idx, qrel_list in enumerate(qid_match_table_list):
            tid = qrel_list[0].strip()
            if args.hnsw_index:
                score = retrieved.get(tid, 0.0)
            else:
                score = retrieved.get(tid, -1000)
            qrel_list[-1] = score

        result[qid] = qid_match_table_list

    fn = 'test_hnsw.result' if args.hnsw_index else 'test.result'
    save_TREC_format(args, fn, result)
    trec_eval = TREC_evaluator(qrels_file=os.path.join(args.data_dir, 'test.jsonl.qrels'),
                               result_file=os.path.join(args.data_dir, fn),
                               trec_cmd="./trec_eval")
    print(trec_eval.get_ndcgs(metric='map'))
    print(trec_eval.get_ndcgs(metric='recip_rank'))
    print(trec_eval.get_ndcgs())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_arguments(parser)
    args = parser.parse_args()
    main(args)
