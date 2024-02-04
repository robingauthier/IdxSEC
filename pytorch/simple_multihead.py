import torch
import torch.nn as nn
import pytorch_lightning as pl

class SequenceLabeler(nn.Module):
    def __init__(self, num_tags, num_features):
        super(SequenceLabeler, self).__init__()
        self.linear = nn.Linear(num_features, num_tags)
        self.attention = nn.MultiheadAttention(num_tags, num_heads=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss()
        self.l1_reg = nn.L1Loss()

    def forward(self, x):
        import pdb;pdb.set_trace()
        x = self.linear(x)
        x, _ = self.attention(x, x, x)
        x = self.log_softmax(x)
        return x


class LitModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        num_tags=5
        num_features=8
        self.model = SequenceLabeler(num_tags, num_features)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        import pdb;pdb.set_trace()
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = nn.functional.mse_loss(x_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    import os
    os.environ['HF_DATASETS_OFFLINE']='1'
    # Load sample dataset
    ds = load_dataset('conll2003')
    #ds = load_dataset('/Users/sachadrevet/.cache/huggingface/datasets/conll2003/conll2003/1.0.0/9a4d16a94f8674ba3466315300359b0acd891b68b6c8743ddf60b9c702adce98')
    # Prepare data
    train_data = ds['train'].select(range(100))
    val_data = ds['validation'].select(range(100))

    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def tokenize(examples):
        return tokenizer(examples['tokens'], truncation=True)

    train_data = train_data.map(tokenize)
    val_data = val_data.map(tokenize)

    # Create PyTorch dataloaders
    train_loader = DataLoader(train_data, batch_size=32)
    val_loader = DataLoader(val_data, batch_size=32)



    # Define and train model
    model = LitModel()  # PyTorch Lightning model from previous example
    trainer = pl.Trainer()
    r=trainer.fit(model, train_loader, val_loader)

    #############################

    # init the autoencoder
    autoencoder = LitAutoEncoder(encoder, decoder)
    trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)


    import pdb;pdb.set_trace()

# ipython -i -m IdxSEC.pytorch.simple_multihead
if __name__=='__main__':
    main()