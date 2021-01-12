import argparse

import pytorch_lightning as pl
from data_module import QueryTableDataModule
from model import QueryTableMatcher
import subprocess


class TREC_evaluator(object):
    def __init__(self, qrels_file, result_file, trec_cmd = "./trec_eval"):
        self.rank_path = result_file
        self.qrel_path = qrels_file
        self.trec_cmd = trec_cmd

    def get_ndcgs(self, metric='ndcg_cut', qrel_path=None, rank_path=None, all_queries=False):
        if qrel_path is None:
            qrel_path = self.qrel_path
        if rank_path is None:
            rank_path = self.rank_path

        metrics = ['ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_15', 'ndcg_cut_20',
                   #'map_cut_5', 'map_cut_10', 'map_cut_15', 'map_cut_20',
                   'map', 'recip_rank']  # 'relstring'
        if all_queries:
            results = subprocess.run([self.trec_cmd, '-c', '-m', metric, '-q', qrel_path, rank_path],
                                     stdout=subprocess.PIPE).stdout.decode('utf-8')
            q_metric_dict = dict()
            for line in results.strip().split("\n"):
                seps = line.split('\t')

                metric = seps[0].strip()
                qid = seps[1].strip()
                if metric not in metrics:
                    continue
                if metric != 'relstring':
                    score = float(seps[2].strip())
                else:
                    score = seps[2].strip()
                if qid not in q_metric_dict:
                    q_metric_dict[qid] = dict()
                q_metric_dict[qid][metric] = score
            return q_metric_dict

        else:
            results = subprocess.run([self.trec_cmd, '-c', '-m', metric, qrel_path, rank_path],
                                     stdout=subprocess.PIPE).stdout.decode('utf-8')

            ndcg_scores = dict()
            for line in results.strip().split("\n"):
                seps = line.split('\t')
                metric = seps[0].strip()
                qid = seps[1].strip()
                if metric not in metrics or qid != 'all':
                    continue
                ndcg_scores[seps[0].strip()] = float(seps[2])
            return ndcg_scores


def result_to_trecformat(qrelsFile, predictFile, resultFile):
    ZERO_COL_ROW_TABLE_COS_VALUE = 0  # col == 0 or row == 0 의 Cos-simil값
    qrels_dict = {}

    # Read qrels file
    with open(qrelsFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            qid, iter, tid, rel = line.split('\t')
            qrels_dict["{}_{}".format(qid, tid)] = [qid, iter, tid, rel.strip(), f'{ZERO_COL_ROW_TABLE_COS_VALUE}', 'test\n']

    # Read predict file
    with open(predictFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            qid, tid, cosSim = line.split(',')
            qrels_dict["{}_{}".format(qid, tid)][-2] = cosSim.strip()

    # Save result
    with open(resultFile, 'w') as f:
        for _, v in qrels_dict.items():
            f.write("\t".join(v))


def predict(args):
    data_module = QueryTableDataModule(args)
    data_module.setup('test')

    model = QueryTableMatcher(args)
    model = model.load_from_checkpoint(
        checkpoint_path=args.ckpt_file,
    )
    # TODO: args도 checkpoint 대로 따라가서 output_file 이 덮어짐
    model.hparams.output_file = args.output_file
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.test(model, datamodule=data_module)


def add_generic_arguments(parser):
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--qrel_file", default=None, type=str, required=True,
                        help="The qrel file  file ")
    parser.add_argument("--result_file", default=None, type=str, required=True,
                        help="The result file ")
    parser.add_argument("--ckpt_file", default=None, type=str, required=True,
                        help="The ckpt file ")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--precision", default=32, type=int, help="Precision")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_arguments(parser)
    QueryTableMatcher.add_model_specific_args(parser)
    args = parser.parse_args()
    # predict
    predict(args)

    print(">>> Json to TREC format ... ", end='')
    result_to_trecformat(args.qrel_file, args.output_file, args.result_file)
    print("Done...")

    print(">>> Run TREC Script ... ", end='')
    trec_eval = TREC_evaluator(qrels_file=args.qrel_file,
                               result_file=args.result_file,
                               trec_cmd="./trec_eval")
    print("Done...")
    print(trec_eval.get_ndcgs())

