# 아래다가 이어붙인모델
from itertools import chain

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup

from table_bert import TableBertModel


class QueryTableMatcher(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.Qmodel = BertModel.from_pretrained(self.hparams.bert_path)
        self.Tmodel = TableBertModel.from_pretrained(self.hparams.tabert_path)
        self.criterion = nn.MarginRankingLoss(margin=1)
        self.avg_pooler = nn.AdaptiveAvgPool2d([1, 768])

    def forward(self, q, pos_column, pos_caption, neg_column=None, neg_caption=None):
        qCLS = self.Qmodel(**q)[0]
        #print(pos_caption)# (,)
        context_encoding, column_encoding, _ = self.Tmodel.encode(contexts=pos_caption, tables=pos_column)
        print(context_encoding.size()) # 2 5 768
        print(column_encoding.size()) # 2 4 768
        concat_vec = torch.cat((context_encoding, column_encoding), 1)
        print(concat_vec.size())
        print(qCLS.size())
        enter = input(">>>정지")
        tp_concat_encoding = self.avg_pooler(context_encoding) + self.avg_pooler(column_encoding)
        q_tp_cos = F.cosine_similarity(qCLS, tp_concat_encoding.squeeze(1))

        if neg_column is None:
            q_tn_cos = None
        else:
            context_encoding, column_encoding, _ = self.Tmodel.encode(contexts=neg_caption, tables=neg_column)
            tn_concat_encoding = self.avg_pooler(context_encoding) + self.avg_pooler(column_encoding)
            q_tn_cos = F.cosine_similarity(qCLS, tn_concat_encoding.squeeze(1))
        return q_tp_cos, q_tn_cos

    def infer(self, q, table_column, table_caption, qid, tidList):
        # For predict
        qCLS = self.Qmodel(**q)[1]
        tableLen = len(tidList[0])

        resultList = []
        for i in range(tableLen):
            context_encoding, column_encoding, _ = self.Tmodel.encode(contexts=(table_caption[0][i],), tables=(table_column[0][i],))
            table_concat_encoding = self.avg_pooler(context_encoding) + self.avg_pooler(column_encoding)
            q_t_cos = F.cosine_similarity(qCLS, table_concat_encoding.squeeze(1))
            #print("{} , {} : {} ".format(qid[0], tidList[0][i], q_t_cos))
            resultList.append([qid[0], tidList[0][i], str(abs(q_t_cos.item()))])
        return resultList

    def test_step(self, batch, batch_idx):
        resultList = self.infer(*batch)
        with open(self.hparams.output_file, 'a') as f:
            for r in resultList:
                f.write("{}\n".format(",".join(r)))


    def training_step(self, batch, batch_idx):
        tp_cos, tn_cos = self(*batch)
        nbatch = tp_cos.size(0)
        target = torch.ones(nbatch, device=self.device)
        loss = self.criterion(tp_cos, tn_cos, target)

        # result = pl.TrainResult(minimize=loss)
        # lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        # result.log('train_loss', loss, on_epoch=True)
        self.log('train_loss', loss, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        tp_cos, tn_cos = self(*batch)
        nbatch = tp_cos.size(0)
        target = torch.ones(nbatch, device=self.device)
        loss = self.criterion(tp_cos, tn_cos, target)

        # result = pl.EvalResult(checkpoint_on=loss)
        # result.log('val_loss', loss)
        self.log('val_loss', loss)
        return {'val_loss': loss}


    def test_step_end(self, test_step_outputs):
        pass

    @property
    def total_steps(self):
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        dataset_size = len(self.trainer.datamodule.train)
        return (dataset_size / effective_batch_size) * self.hparams.max_epochs

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        model_named_parameters = list(chain(self.Qmodel.named_parameters(), self.Tmodel.named_parameters()))
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model_named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model_named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup, num_training_steps=self.total_steps
        )

        # 추후 qmodel이랑 tmodel이 서로 다른 optimize를 해야할 수도 있음
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--bert_path", default=None, type=str, required=True)
        parser.add_argument("--tabert_path", default=None, type=str, required=True)
        parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate")
        # parser.add_argument("--qmodel_lr", default=1e-5,
        #                     type=float, help="The initial learning rate for query model.")
        # parser.add_argument("--tmodel_lr", default=1e-5,
        #                     type=float, help="The initial learning rate for table model.")
        parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup", default=0, type=int, help="Linear warmup over warmup_steps.")
        # parser.add_argument("--qmodel_warmup", default=5000, type=int, help="Linear warmup over warmup_steps.")
        # parser.add_argument("--tmodel_warmup", default=2000, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--max_epochs", default=10, type=int, help="Number of training epochs")
        parser.add_argument("--train_batch_size", default=2, type=int, help="Number of training epochs")
        parser.add_argument("--valid_batch_size", default=2, type=int, help="Number of training epochs")
        parser.add_argument("--test_batch_size", default=2, type=int, help="Number of training epochs")
        parser.add_argument("--output_file", default="./default.csv", type=str, help="output file name")
