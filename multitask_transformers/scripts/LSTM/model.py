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
from sklearn.metrics import classification_report
from tqdm import tqdm


# determine what device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float("inf"))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(
            batch_size, -1, input_size
        )

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(
            batch_size, -1, hidden_size
        )

        return output, attn


class LSTM(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, hidden_dim, use_attention=False, num_layers=1
    ):
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
        self.fc_sarc_att = nn.Linear(hidden_dim, self.num_class_sarc)
        self.fc_arg_att = nn.Linear(hidden_dim, self.num_class_arg)
        self.use_attention = use_attention
        self.attention = Attention(hidden_dim)

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

        if self.use_attention:
            out, attn = self.attention(out_pt, out_ct)
            out = out[:, -1:, :]
            out = torch.squeeze(out, 1)
            out_sarc = self.fc_sarc_att(out)
            out_arg = self.fc_arg_att(out)

            return out_sarc, out_arg

        # print(out_pt[:, pt_len - 1, :].shape, out_ct[:, ct_len - 1, :].shape)
        out = torch.cat((out_pt[:, -1:, :], out_ct[:, -1:, :]), 2)
        out = torch.squeeze(out, 1)
        out_sarc = self.fc_sarc(out)
        out_arg = self.fc_arg(out)

        return out_sarc, out_arg


def dynamic_loss(arg_loss, sarc_loss):
    sigma_1 = Variable(torch.tensor(0.5), requires_grad=True)
    sigma_2 = Variable(torch.tensor(0.5), requires_grad=True)
    arg_loss_dyn = torch.mul(
        torch.div(1.0, torch.mul(2.0, torch.square(sigma_1))), arg_loss
    )
    sarc_loss_dyn = torch.mul(
        torch.div(1.0, torch.mul(2.0, torch.square(sigma_2))), sarc_loss
    )

    loss = torch.add(arg_loss_dyn, sarc_loss_dyn)
    loss = torch.add(loss, torch.log(torch.mul(sigma_1, sigma_2)))

    return loss


def train(
    model,
    optimizer,
    train_it,
    test_it,
    criterion=nn.CrossEntropyLoss(),
    num_epochs=1,
    flood=False,
    flood_level=0.0,
    dynamic=False,
):

    loss_epochs = []
    val_loss_epochs = []
    for epoch in range(num_epochs):
        # print("Epoch: ", str(epoch))
        model.train()

        with tqdm(total=len(train_it)) as epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch}")

            sum_loss = 0.0
            for ((pt, pt_len), (ct, ct_len), arg_labels, sarc_labels), _ in train_it:
                # pt = pt.to(device)
                # pt_len = pt_len.to(device)
                # ct = ct.to(device)
                # ct_len = ct_len.to(device)
                # arg_labels = arg_labels.to(device)
                # sarc_labels = sarc_labels.to(device)

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
                    loss = (loss - flood_level).abs() + flood_level

                loss.backward()
                sum_loss += loss.item()
                optimizer.step()

                # desc = f'Epoch {epoch} - loss {avg_loss:.4f}'
                # epoch_pbar.set_description(desc)
                epoch_pbar.update(1)

            avg_loss_epoch = sum_loss / len(train_it)
            avg_val_loss_epoch = eval(model, test_it, criterion)
            print("train loss: ", avg_loss_epoch, "val loss: ", avg_val_loss_epoch)

            loss_epochs.append(avg_loss_epoch)
            val_loss_epochs.append(avg_val_loss_epoch)


def eval(model, test_it, criterion):
    model.eval()

    sum_val_loss = 0.0
    arg_preds = []
    arg_labels_all = []
    sarc_preds = []
    sarc_labels_all = []

    with tqdm(total=len(test_it)) as val_pbar:
        val_pbar.set_description(f"Validating")

        with torch.no_grad():
            for ((pt, pt_len), (ct, ct_len), arg_labels, sarc_labels), _ in test_it:
                # pt = pt.to(device)
                # pt_len = pt_len.to(device)
                # ct = ct.to(device)
                # ct_len = ct_len.to(device)
                # arg_labels = arg_labels.to(device)
                # sarc_labels = sarc_labels.to(device)

                sarc_outs, arg_outs = model(pt, pt_len, ct, ct_len)
                arg_loss = criterion(arg_outs, arg_labels)
                sarc_loss = criterion(sarc_outs, sarc_labels)
                loss = torch.add(arg_loss, sarc_loss)
                # loss = dynamic_loss(arg_loss, sarc_loss)
                sum_val_loss += loss.item()

                arg_pred = list(torch.max(arg_outs[:, 1:], 1)[1] + 1)
                arg_preds.extend([i.item() for i in arg_pred])
                arg_labels_all.extend([i.item() for i in arg_labels])

                sarc_pred = list(torch.max(sarc_outs[:, 1:], 1)[1] + 1)
                sarc_preds.extend([i.item() for i in sarc_pred])
                sarc_labels_all.extend([i.item() for i in sarc_labels])

                val_pbar.update(1)

            print(
                classification_report(arg_labels_all, arg_preds),
                "\n",
                classification_report(sarc_labels_all, sarc_preds),
            )
            return sum_val_loss / len(test_it)


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

    model = LSTM(
        vocab_size=len(text_field.vocab),
        embed_dim=300,
        hidden_dim=300,
        use_attention=True,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(
        model,
        optimizer,
        train_it,
        test_it,
        nn.CrossEntropyLoss(),
        num_epochs=1,
        flood=False,
        flood_level=0.5,
    )
