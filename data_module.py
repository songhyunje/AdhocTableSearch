import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from dataset import QueryTableDataset, query_table_collate_fn
from table_bert import TableBertModel


class QueryTableDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.data_dir = params.data_dir

        table_model = TableBertModel.from_pretrained(params.tabert_path)
        # self.query_tokenizer = BertTokenizer.from_pretrained(params.bert_path)
        self.table_tokenizer = table_model.tokenizer
        self.query_tokenizer = table_model.tokenizer

        self.train_batch_size = params.train_batch_size
        self.valid_batch_size = params.valid_batch_size
        
        self.min_row = params.min_row

    def prepare_data(self):
        # Download, tokenize, etc
        # Write to disk or that need to be done only from a single GPU in distributed settings
        QueryTableDataset(data_dir=self.data_dir, data_type='train',
                          query_tokenizer=self.query_tokenizer,
                          table_tokenizer=self.table_tokenizer,
                          min_row=self.min_row,
                          prepare=True)


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            table_full = QueryTableDataset(data_dir=self.data_dir, data_type='train', min_row=self.min_row)
            self.train, self.valid = random_split(table_full, [len(table_full) - int(len(table_full) * 0.1),
                                                               int(len(table_full) * 0.1)])

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          collate_fn=query_table_collate_fn,
                          num_workers=4,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid,
                          batch_size=self.valid_batch_size,
                          collate_fn=query_table_collate_fn,
                          num_workers=4,
                          )


if __name__ == "__main__":
    args = argparse.Namespace()
    args.data_dir = 'data'
    args.tabert_path = 'model/tabert_base_k3/model.bin'
    args.train_batch_size = 2
    args.valid_batch_size = 2

    data_module = QueryTableDataModule(args)
    data_module.prepare_data()
    data_module.setup('fit')
    for batch in data_module.train_dataloader():
        print(*batch)

