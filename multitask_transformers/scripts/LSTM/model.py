import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torchtext import data
from torchtext import datasets
from torchtext.data import Field, TabularDataset, BucketIterator
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from torch.autograd import Variable


# determine what device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.num_class_sarc = 3
        self.num_class_arg = 4
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        self.lstm_1 = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True,
        )
        self.lstm_2 = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True,
        )
        self.fc_sarc = nn.Linear(2 * hidden_dim, self.num_class_sarc)
        self.fc_arg = nn.Linear(2 * hidden_dim, self.num_class_arg)

    def forward(self, pt, pt_len, ct, ct_len):
        pt_embed = self.embedding(pt)
        pack_pt = pack_padded_sequence(
            pt_embed, pt_len, batch_first=True, enforce_sorted=False
        )
        out_packed_pt, _ = self.lstm_1(pack_pt)
        out_pt, _ = pad_packed_sequence(out_packed_pt, batch_first=True)

        ct_embed = self.embedding(ct)
        pack_ct = pack_padded_sequence(
            ct_embed, ct_len, batch_first=True, enforce_sorted=False
        )
        out_packed_ct, _ = self.lstm_2(pack_ct)
        out_ct, _ = pad_packed_sequence(out_packed_ct, batch_first=True)

        # print(out_pt[:, pt_len - 1, :].shape, out_ct[:, ct_len - 1, :].shape)
        out = torch.cat((out_pt[:, -1:, :], out_ct[:, -1:, :]), 2)
        out = torch.squeeze(out, 1)
        out_sarc = self.fc_sarc(out)
        out_arg = self.fc_arg(out)

        return out_sarc, out_arg


def dynamic_loss(arg_loss,sarc_loss):
    sigma_1 = Variable(torch.tensor(0.5), requires_grad=True)
    sigma_2 = Variable(torch.tensor(0.5), requires_grad=True)
    arg_loss_dyn = torch.mul(torch.div(1.0, torch.mul(2.0, torch.square(sigma_1))), arg_loss)           
    sarc_loss_dyn = torch.mul(torch.div(1.0, torch.mul(2.0, torch.square(sigma_2))), sarc_loss)
     
    loss = torch.add(arg_loss_dyn, sarc_loss_dyn)
    loss = torch.add(loss, torch.log(torch.mul(sigma_1, sigma_2)))
   
    return loss


def train(
    model, optimizer, train_it, test_it, criterion=nn.CrossEntropyLoss(), num_epochs=1, flood=False, flood_level=0., dynamic=False
):

    loss_epochs = []
    val_loss_epochs = []
    for epoch in range(num_epochs):
        model.train()
        sum_loss = 0.

        for ((pt, pt_len), (ct, ct_len), arg_labels, sarc_labels), _ in train_it:
            sarc_outs, arg_outs = model(pt, pt_len, ct, ct_len)
            arg_loss = criterion(arg_outs, arg_labels)
            sarc_loss = criterion(sarc_outs, sarc_labels)

            # Choose loss addition for multitask (direct-add vs dynamic)
            if dynamic:
                loss = dynamic_loss(arg_loss, sarc_loss)
            else:
                loss = torch.add(arg_loss, sarc_loss)
            
            # flooding
            if flood:
                loss = (loss-flood_level).abs() + flood_level

            loss.backward()
            sum_loss += loss.item()
            optimizer.step()

        avg_loss_epoch = sum_loss/len(train_it)
        avg_val_loss_epoch = eval(model, test_it, criterion)
        print("train loss: ", avg_loss_epoch, "val loss: ", avg_val_loss_epoch)

        loss_epochs.append(avg_loss_epoch)
        val_loss_epochs.append(avg_val_loss_epoch)

def eval(model, test_it, criterion):
    model.eval()

    sum_val_loss = 0.

    with torch.no_grad():
        for ((pt, pt_len), (ct, ct_len), arg_labels, sarc_labels), _ in test_it:
            sarc_outs, arg_outs = model(pt, pt_len, ct, ct_len)
            print(sarc_outs.shape, arg_outs.shape)
            arg_loss = criterion(arg_outs, arg_labels)
            sarc_loss = criterion(sarc_outs, sarc_labels)
            loss = torch.add(arg_loss, sarc_loss)
            # loss = dynamic_loss(arg_loss, sarc_loss)
            sum_val_loss += loss.item()

        return sum_val_loss/len(test_it)

if __name__ == "__main__":
    sarc_label_field = data.Field(sequential=False, use_vocab=True, batch_first=True)
    arg_label_field = data.Field(sequential=False, use_vocab=True, batch_first=True)
    text_field = data.Field(
        tokenize="spacy", lower=True, include_lengths=True, batch_first=True
    )

    fields = [
        ("pt", text_field),
        ("ct", text_field),
        ("arg", arg_label_field),
        ("sarc", sarc_label_field),
    ]

    train_ds, val_ds, test_ds = TabularDataset.splits(
        path="../../data/",
        train="train.v1.txt",
        validation="dev.v1.txt",
        test="test.v1.txt",
        format="tsv",
        fields=fields,
        skip_header=False,
    )

    # print(vars(train_ds[0]))

    text_field.build_vocab(train_ds)
    sarc_label_field.build_vocab(train_ds)
    arg_label_field.build_vocab(train_ds)

    # print(len(text_field.vocab))

    train_it, val_it, test_it = BucketIterator.splits(
        (train_ds, val_ds, test_ds), sort=False, batch_size=32, device=device
    )

    model = LSTM(vocab_size=len(text_field.vocab), embed_dim=300, hidden_dim=300).to(
        device
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, optimizer, train_it, test_it, nn.CrossEntropyLoss(), num_epochs=1, flood=False, flood_level=0.5)
