from itertools import chain

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

from table_bert import TableBertModel


class QueryTableMatcher(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.tabert = TableBertModel.from_pretrained(self.hparams.tabert_path, self.hparams.config_file)

        self.table_linear = nn.Linear(table_model.config.hidden_size, 128)
        self.query_emb_head = nn.Linear(table_model.config.hidden_size, 128)
        # self.table_emb_head = nn.Linear(128, 128)
        self.norm = nn.LayerNorm(128)

        # attention
        # self.attention = nn.Sequential(
        #     nn.Linear(768, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 1)
        # )
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, query, tables):
        query_embs = self.query_forward(query)   # B x d 
        table_embs = self.table_forward(tables)  # B x d
        scores = torch.mm(query_embs, table_embs.transpose(0, 1))  # B x B
        # scores = torch.matmul(query_embs, table_embs.T)
        return scores

    def training_step(self, batch, batch_idx):
        query, tables, hard_tables, rel_pair_mask, hard_pair_mask = batch
        loss = in_batch_training(self, query, tables, rel_pair_mask, hard_tables, hard_pair_mask) 
        self.log('train_loss', loss, on_epoch=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        query, tables, hard_tables, rel_pair_mask, hard_pair_mask = batch
        loss = in_batch_training(self, query, tables, rel_pair_mask, hard_tables, hard_pair_mask) 
        self.log('val_loss', loss)
        return loss 

    def query_forward(self, query):
        # use cls vector 
        query_tokens = self.tabert.bert(**query)[0]             # B x Q x d
        return self.norm(self.query_emb_head(query_tokens[:, 0, :]))
        # return F.normalize(query_tokens[:, 0, :], p=2, dim=1)	# B x d
        # return self.norm(query_tokens[:, 0, :])  # B x d

    # def table_forward(self, tables):
    #     context_encoding, table_encoding, _ = self.tabert.encode(contexts=tables[1], tables=tables[0])
    #     H = self.table_linear(context_encoding[:, 0, :] + torch.mean(table_encoding, dim=1))

    #     print(H.shape)

    #     # Attention pooling
    #     A = self.attention(H)         # subN x 1
    #     A = torch.transpose(A, 1, 0)  # 1 x subN
    #     A = F.softmax(A, dim=1)       # softmax over subN

    #     print(torch.mm(A, H).shape)

    #     return self.norm(torch.mm(A, H)) # 1 x 768

    def table_forward(self, tables):
        reps = []
        # for table, caption in zip(tables, captions):
        for table, caption in zip(tables[0], tables[1]):
            context_encoding, table_encoding, _ = self.tabert.encode(contexts=caption, tables=table)
            # H = self.norm(context_encoding[:, 0, :] + torch.mean(table_encoding, dim=1))
            H = self.table_linear(context_encoding[:, 0, :] + torch.mean(table_encoding, dim=1))

            # Attention pooling
            A = self.attention(H)         # subN x 1
            A = torch.transpose(A, 1, 0)  # 1 x subN
            A = F.softmax(A, dim=1)       # softmax over subN

            M = self.norm(torch.mm(A, H)) # 1 x 768
            # M = self.norm(self.table_emb_head(M))

            # M = torch.mm(A, H)
            # M = F.normalize(M, p=2, dim=1)  # 1 x 768
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
        no_decay = ['bias', 'LayerNorm.bias']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup, num_training_steps=self.total_steps
        )

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--tabert_path", default=None, type=str, required=True)
        parser.add_argument("--config_file", default=None, type=str, required=True)
        parser.add_argument("--lr", default=1e-5, type=float, help="The initial learning rate")
        parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup", default=100, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--reload_dataloaders_every_n_epochs", default=0, type=int, help="")
        parser.add_argument("--max_epochs", default=5, type=int, help="Number of training epochs")
        parser.add_argument("--min_row", default=10, type=int, help="Minimum number of rows")
        parser.add_argument("--train_batch_size", default=2, type=int, help="Number of training epochs")
        parser.add_argument("--valid_batch_size", default=2, type=int, help="Number of training epochs")


# most parts are borrowed from DRhard 
def in_batch_training(forward, query, tables, rel_pair_mask, hard_tables, hard_pair_mask):
    batch_scores = forward(query, tables)
    batch_size = batch_scores.shape[0]
   
    single_positive_scores = torch.diagonal(batch_scores, 0)
    positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, batch_size).reshape(-1)

    if rel_pair_mask is None:
        rel_pair_mask = 1 - torch.eye(batch_size, dtype=batch_scores.dtype, device=batch_scores.device)          

    batch_scores = batch_scores.reshape(-1)
    logit_matrix = torch.cat([positive_scores.unsqueeze(1), batch_scores.unsqueeze(1)], dim=1)

    lsm = F.log_softmax(logit_matrix, dim=1)
    loss = -1.0 * lsm[:, 0] * rel_pair_mask.reshape(-1)
    first_loss, first_num = loss.sum(), rel_pair_mask.sum()

    other_batch_scores = forward(query, hard_tables)
    other_doc_num = len(hard_tables[0])
    other_batch_scores = other_batch_scores.reshape(-1)
    positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, other_doc_num).reshape(-1)
    other_logit_matrix = torch.cat([positive_scores.unsqueeze(1), other_batch_scores.unsqueeze(1)], dim=1)  

    other_lsm = F.log_softmax(other_logit_matrix, dim=1)
    other_loss = -1.0 * other_lsm[:, 0]

    if hard_pair_mask is not None:
        hard_pair_mask = hard_pair_mask.reshape(-1)
        other_loss = other_loss * hard_pair_mask
        second_loss, second_num = other_loss.sum(), hard_pair_mask.sum()
    else:
        second_loss, second_num = other_loss.sum(), len(other_loss)

    final_loss = ((first_loss+second_loss)/(first_num+second_num))
    return final_loss

