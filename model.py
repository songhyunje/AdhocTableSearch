from itertools import chain

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

from table_bert import TableBertModel


class QueryTableMatcher(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.Tmodel = TableBertModel.from_pretrained(self.hparams.tabert_path, self.hparams.config_file)
        self.norm = nn.LayerNorm(768)

        # attention
        self.attention = nn.Sequential(
            nn.Linear(768, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # self.linear = nn.Linear(768, 1)
        # self.linear = nn.Sequential(
        #     nn.Linear(768, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, query, tables, captions):
        # use cls vector 
        query_tokens = self.Tmodel.bert(**query)[0]            # B x Q x d
        # query = self.norm(query_tokens[:, 0, :])  # B x d
        query = F.normalize(query_tokens[:, 0, :], p=2, dim=1)  # B x d
        # for loop is terrible,
        # but it has no choice but to use it!
        reps = []  # B x 768
        for q, table, caption in zip(query, tables, captions):
            context_encoding, table_encoding, _ = self.Tmodel.encode(contexts=caption, tables=table)
            H = self.norm(context_encoding[:, 0, :] + torch.mean(table_encoding, dim=1))
            # H = context_encoding[:, 0, :] + torch.mean(table_encoding, dim=1)

            # 이 부분을 밖으로 빼면 될 듯 한데...
            # H = q * concat_encoding  # subN x 768
            # print(H.shape)

            # Max pooling
            # M, _ = torch.max(H, 0)   # T(# of table) x 768
            # reps.append(M)

            # Attention pooling
            A = self.attention(H)         # subN x 1
            A = torch.transpose(A, 1, 0)  # 1 x subN
            A = F.softmax(A, dim=1)       # softmax over subN

            M = torch.mm(A, H) # 1 x 768
            t_norm = F.normalize(M, p=2, dim=1)
            # print(q_norm.shape)
            # print(t_norm.shape)
            # print(torch.mm(q_norm, t_norm.transpose(0, 1)).shape)
            reps.append(torch.mm(q.unsqueeze(0), t_norm.transpose(0, 1)).squeeze(0))
            # reps.append(self.linear(q * self.norm(M)).squeeze(0))
            # reps.append(M.squeeze(0))

        return torch.stack(reps)   # B x 1
        # return self.linear(torch.stack(reps))   # B x 1

    def training_step(self, batch, batch_idx):
        query, tables, captions, rel = batch
        outputs = self(query, tables, captions)
        loss = F.binary_cross_entropy_with_logits(outputs, rel.unsqueeze(1))
        # logit_matrix = torch.cat([tp_cos.unsqueeze(1), tn_cos.unsqueeze(1)], dim=1)  # [B, 2]
        # lsm = F.log_softmax(logit_matrix, dim=1)
        # loss = -1.0 * lsm[:, 0]
        self.log('train_loss', loss.mean(), on_epoch=True)
        # tp_cos, tn_cos = self(*batch)
        # nbatch = tp_cos.size(0)
        # target = torch.ones(nbatch, device=self.device)
        # loss = self.criterion(tp_cos, tn_cos, target)
        # self.log('train_loss', loss.mean(), on_epoch=True)
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        query, tables, captions, rel = batch
        outputs = self(query, tables, captions)
        # tp_cos, tn_cos = self(*batch)
        # nbatch = tp_cos.size(0)
        # target = torch.ones(nbatch, device=self.device)
        loss = F.binary_cross_entropy_with_logits(outputs, rel.unsqueeze(1))
        # loss = self.criterion(tp_cos, tn_cos, target)
        self.log('val_loss', loss.mean())
        return loss.mean()

    def query_forward(self, query):
        # return self.norm(self.Tmodel.bert(**query)[1])  # B x d
        query_tokens = self.Tmodel.bert(**query)[0]            # B x Q x d
        # return self.norm(query_tokens[:, 0, :])  # B x d
        return F.normalize(query_tokens[:, 0, :], p=2, dim=1)

    def table_forward(self, tables, captions):
        reps = []
        for table, caption in zip(tables, captions):
            context_encoding, table_encoding, _ = self.Tmodel.encode(contexts=caption, tables=table)
            H = self.norm(context_encoding[:, 0, :] + torch.mean(table_encoding, dim=1))

            # Attention pooling
            A = self.attention(H)         # subN x 1
            A = torch.transpose(A, 1, 0)  # 1 x subN
            A = F.softmax(A, dim=1)       # softmax over subN

            # M = self.norm(torch.mm(A, H)) # 1 x 768
            M = torch.mm(A, H)
            M = F.normalize(M, p=2, dim=1)
            reps.append(M.squeeze(0))

        return torch.stack(reps)

    @property
    def total_steps(self):
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        dataset_size = len(self.trainer.datamodule.train)
        return (dataset_size / effective_batch_size) * self.hparams.max_epochs

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.Tmodel.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.Tmodel.named_parameters() if any(nd in n for nd in no_decay)],
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
        parser.add_argument("--tabert_path", default=None, type=str, required=True)
        parser.add_argument("--config_file", default=None, type=str, required=True)
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
        parser.add_argument("--max_epochs", default=5, type=int, help="Number of training epochs")
        parser.add_argument("--min_row", default=10, type=int, help="Minimum number of rows")
        parser.add_argument("--train_batch_size", default=2, type=int, help="Number of training epochs")
        parser.add_argument("--valid_batch_size", default=2, type=int, help="Number of training epochs")
        # parser.add_argument("--test_batch_size", default=2, type=int, help="Number of training epochs")

