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
        self.avg_pooler = nn.AdaptiveAvgPool2d([1, 768])

    def forward(self, q, pos_column, pos_caption, neg_column, neg_caption):
        qCLS = self.Qmodel(**q)[1]  # b x d
        context_encoding, column_encoding, _ = self.Tmodel.encode(contexts=pos_caption, tables=pos_column)
        tp_concat_encoding = self.avg_pooler(context_encoding) + self.avg_pooler(column_encoding)
        q_tp_cos = F.cosine_similarity(qCLS, tp_concat_encoding.squeeze(1))

        context_encoding, column_encoding, _ = self.Tmodel.encode(contexts=neg_caption, tables=neg_column)
        tn_concat_encoding = self.avg_pooler(context_encoding) + self.avg_pooler(column_encoding)
        q_tn_cos = F.cosine_similarity(qCLS, tn_concat_encoding.squeeze(1))
        return q_tp_cos, q_tn_cos

    def training_step(self, batch, batch_idx):
        tp_cos, tn_cos = self(*batch)
        logit_matrix = torch.cat([tp_cos.unsqueeze(1), tn_cos.unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        self.log('train_loss', loss.mean(), on_epoch=True)
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        tp_cos, tn_cos = self(*batch)
        logit_matrix = torch.cat([tp_cos.unsqueeze(1), tn_cos.unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        self.log('val_loss', loss.mean())

    def test_step(self, batch, batch_idx):
        q, column, caption, _, qid, tid = batch
        score = self.infer(q, column, caption)
        with open(self.hparams.output_file, 'a') as f:
            f.write("{},{},{}\n".format(qid[0], tid[0], abs(score.item())))

    def infer(self, q, column, caption):
        qCLS = self.Qmodel(**q)[1]  # b x d
        context_encoding, column_encoding, _ = self.Tmodel.encode(contexts=caption, tables=column)
        concat_encoding = self.avg_pooler(context_encoding) + self.avg_pooler(column_encoding)
        t_cos = F.cosine_similarity(qCLS, concat_encoding.squeeze(1))
        return t_cos

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
        parser.add_argument("--lr", default=1e-5, type=float, help="The initial learning rate")
        # parser.add_argument("--qmodel_lr", default=1e-5,
        #                     type=float, help="The initial learning rate for query model.")
        # parser.add_argument("--tmodel_lr", default=1e-5,
        #                     type=float, help="The initial learning rate for table model.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup", default=0, type=int, help="Linear warmup over warmup_steps.")
        # parser.add_argument("--qmodel_warmup", default=5000, type=int, help="Linear warmup over warmup_steps.")
        # parser.add_argument("--tmodel_warmup", default=2000, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--max_epochs", default=10, type=int, help="Number of training epochs")
        parser.add_argument("--train_batch_size", default=2, type=int, help="Number of training epochs")
        parser.add_argument("--valid_batch_size", default=2, type=int, help="Number of training epochs")
        parser.add_argument("--test_batch_size", default=2, type=int, help="Number of training epochs")
        parser.add_argument("--output_file", default="./default.csv", type=str, help="output file name")
