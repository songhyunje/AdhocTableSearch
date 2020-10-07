import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

from dataset import QueryTableDataset, query_table_collate_fn
from table_bert import TableBertModel


class QueryTableDataModule(pl.LightningDataModule):
    def __init__(self,
                 query_file: str = './data/queries.txt',
                 table_file: str = './data/all.json',
                 train_batch_size=16,
                 valid_batch_size=16,
                 test_batch_size=16
                 ):
        super().__init__()
        self.query_file = query_file
        self.table_file = table_file

        self.query_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        table_model = TableBertModel.from_pretrained('model/tabert_base_k3/model.bin')
        self.table_tokenizer = table_model.tokenizer
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size

    # def prepare_data(self):
    #     # Download, tokenize, etc
    #     # Write to disk or that need to be done only from a single GPU in distributed settings
    #     QueryTableDataset(query_file=self.query_file,
    #                       table_file=self.table_file,
    #                       query_tokenizer=self.query_tokenizer,
    #                       table_tokenizer=self.table_tokenizer,
    #                       prepare=True
    #                       )

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            table_full = QueryTableDataset(query_file=self.query_file,
                                           table_file=self.table_file,
                                           query_tokenizer=self.query_tokenizer,
                                           table_tokenizer=self.table_tokenizer,
                                           )
            self.train, self.valid = random_split(table_full, [2700, 420])

        if stage == 'test' or stage is None:
            self.test = QueryTableDataset(query_file=self.query_file, table_file=self.table_file)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size,
                          shuffle=True, collate_fn=query_table_collate_fn,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.valid_batch_size,
                          collate_fn=query_table_collate_fn,
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, num_workers=4)


if __name__ == "__main__":
    data_module = QueryTableDataModule()
    data_module.setup('fit')
    for batch in data_module.train_dataloader():
        print(batch[0])
        print(batch[1])
