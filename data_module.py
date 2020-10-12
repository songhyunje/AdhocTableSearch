import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

from dataset import QueryTableDataset, query_table_collate_fn
from table_bert import TableBertModel


class QueryTableDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.data_dir = params.data_dir

        self.query_tokenizer = BertTokenizer.from_pretrained(params.bert_path)
        table_model = TableBertModel.from_pretrained(params.tabert_path)
        self.table_tokenizer = table_model.tokenizer

        self.train_batch_size = params.train_batch_size
        self.valid_batch_size = params.valid_batch_size
        if hasattr(params, 'test_batch_size'):
            self.test_batch_size = params.test_batch_size

    def prepare_data(self):
        # Download, tokenize, etc
        # Write to disk or that need to be done only from a single GPU in distributed settings
        QueryTableDataset(data_dir=self.data_dir, data_type='train',
                          query_tokenizer=self.query_tokenizer,
                          table_tokenizer=self.table_tokenizer,
                          prepare=True)
        QueryTableDataset(data_dir=self.data_dir, data_type='test',
                          query_tokenizer=self.query_tokenizer,
                          table_tokenizer=self.table_tokenizer,
                          prepare=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            table_full = QueryTableDataset(data_dir=self.data_dir, data_type='train')
            self.train, self.valid = random_split(table_full, [55, 5])

        if stage == 'test' or stage is None:
            self.test = QueryTableDataset(data_dir=self.data_dir, data_type='test')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size,
                          shuffle=True, collate_fn=query_table_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.valid_batch_size,
                          collate_fn=query_table_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size)


if __name__ == "__main__":
    args = argparse.Namespace()
    args.data_dir = 'data'
    args.bert_path = 'bert-base-uncased'
    args.tabert_path = 'model/tabert_base_k3/model.bin'
    args.train_batch_size = 2
    args.valid_batch_size = 2

    data_module = QueryTableDataModule(args)
    data_module.prepare_data()
    data_module.setup('fit')
    for batch in data_module.train_dataloader():
        print(batch[0])
        print(batch[1])
