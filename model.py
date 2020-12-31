from itertools import chain

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup

from table_bert import TableBertModel


# CLS Model
class QueryTableMatcher(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.Qmodel = BertModel.from_pretrained(self.hparams.bert_path)
        self.Tmodel = TableBertModel.from_pretrained(self.hparams.tabert_path)
        self.avg_pooler = nn.AdaptiveAvgPool2d([1, 768])

    def nllloss(self, score, rel_score):
        """
        https://www.aclweb.org/anthology/2020.emnlp-main.550.pdf
        TODO     : e^sim(q, tp)가 0일떄는 어떻게할까 == 모든 table들이 negative일떄
        TODO Sol): 최대한 pos table 한개는 나오게끔 수정해야함
        """
        if not torch.sum(rel_score):
            return torch.tensor([0.0], dtype=torch.float).requires_grad_(True).to("cuda")
        # Rel == 0 => Negative의 score는 분자 계산에서 제외, score mask 진행
        score_mask = torch.where(rel_score == 0, rel_score, score)
        # e^0 == 1이라 0값이 아닌 인덱스만
        positive_score = score_mask[torch.nonzero(score_mask, as_tuple=True)]
        loss = -1 * torch.log( torch.sum(torch.exp(positive_score))
                               / torch.sum(torch.exp(score)))

        return loss

    def forward(self, q, column, caption, rel_score=None):
        qCLS = self.Qmodel(**q)[1] #b d
        context_encoding, column_encoding, _ = self.Tmodel.encode(contexts=caption, tables=column)
        tp_concat_encoding = self.avg_pooler(context_encoding) + self.avg_pooler(column_encoding)
        q_tp_cos = F.cosine_similarity(qCLS, tp_concat_encoding.squeeze(1))
        if rel_score == None:
            return q_tp_cos
        else:
            return q_tp_cos, rel_score

    def training_step(self, batch, batch_idx):
        q, column, caption, rel_score, _, _ = batch
        score, rel_score = self(q, column, caption, torch.tensor(rel_score, dtype=torch.float).to("cuda"))
        loss = self.nllloss(score, rel_score)
        self.log('train_loss', loss, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        q, column, caption, rel_score, _, _ = batch
        score, rel_score = self(q, column, caption, torch.tensor(rel_score, dtype=torch.float).to("cuda"))
        loss = self.nllloss(score, rel_score)
        self.log('val_loss', loss)
        return {'val_loss': loss.item()}

    def test_step(self, batch, batch_idx):
        q, column, caption, _, qid, tid = batch
        score = self(q, column, caption)
        with open(self.hparams.output_file, 'a') as f:
           f.write("{},{},{}\n".format(qid[0], tid[0], abs(score.item())))

    def infer(self, q, table_column, table_caption, qid, tidList):
        pass

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
