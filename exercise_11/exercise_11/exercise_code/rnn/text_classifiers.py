import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import pytorch_lightning as pl

from .rnn_nn import Embedding, RNN, LSTM


class RNNClassifier(pl.LightningModule):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        self.hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }

        ########################################################################
        # TODO: Initialize an RNN network for sentiment classification         #
        # hint: A basic architecture can have an embedding, an rnn             #
        # and an output layer                                                  #
        ########################################################################

        self.embedding = Embedding(num_embeddings, embedding_dim, 0)
        self.rnn = (LSTM if use_lstm else RNN)(input_size=embedding_dim, hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size, 1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, sequence, lenghts=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        output = None

        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################

        embeddings = self.embedding(sequence)
        if not lenghts == None:
            embeddings = pack_padded_sequence(embeddings, lenghts)

        h_seq, h = self.rnn(embeddings)
        if isinstance(h, tuple):
            h = h[0]

        output = self.output(h.squeeze()).sigmoid().view(-1)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return output

    def training_step(self, batch, batch_idx):
        x, y = (batch["data"], batch["label"])
        bce = nn.BCELoss()
        target = self.forward(x)
        train_loss = bce(target, y)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = (batch["data"], batch["label"])
        bce = nn.BCELoss()
        target = self.forward(x)
        val_loss = bce(target, y)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
