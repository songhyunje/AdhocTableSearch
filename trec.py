import subprocess
import os


class TREC_evaluator(object):
    def __init__(self, qrels_file=None, result_file=None, trec_cmd = "../trec_eval"):
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
                   'map', 'mrr', 'recip_rank']  # 'relstring'
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

