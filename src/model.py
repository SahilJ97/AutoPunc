from abc import ABC
import torch
from transformers import RobertaModel
from torch.nn.modules import Module
from torch.nn import Linear, BatchNorm1d, Dropout, LSTM
from torch.nn.functional import softmax, relu
from .data import POSSIBLE_LABELS


class AutoPuncModel(Module, ABC):
    def __init__(
            self,
            n_prosodic_feat=4,
            pretrained_model="roberta-base",
            window_size=100,
            h_space_dim=1500,
            dropout=.2,
            n_threads=1,
            text_embedding_size=768,  # depends on transformer model. 768 for RoBERTa
            prosodic_embedding_size=4
    ):
        super(AutoPuncModel, self).__init__()
        self.transformer = RobertaModel.from_pretrained(pretrained_model)
        self.window_size = window_size
        self.n_threads = n_threads
        self.h_space_dim = h_space_dim
        self.dropout = dropout
        self.prosodic_embedding_size = prosodic_embedding_size

        # Model
        self.transformer = RobertaModel.from_pretrained(pretrained_model)
        self.h_0 = torch.randn(1, 1, self.prosodic_embedding_size)  # trainable initial hidden state for LSTM-RNN
        self.c_0 = torch.randn(1, 1, self.prosodic_embedding_size)  # trainable initial cell state for LSTM-RNN
        self.rnn = LSTM(n_prosodic_feat, self.prosodic_embedding_size, num_layers=1)
        self.hidden_layer = Linear(text_embedding_size + prosodic_embedding_size, h_space_dim)
        self.batch_norm = BatchNorm1d(window_size, affine=False)  # batch norm without learnable params
        self.dropout_layer = Dropout(p=self.dropout)
        self.final_layer = Linear(h_space_dim, len(POSSIBLE_LABELS))

    def forward(self, x):
        # Token embeddings
        tokens = x["tokens"]  # shape: (batch size, self.window_size)
        token_embeddings = self.transformer(tokens)[0]  # shape: (batch size, self.window_size, embedding size)
        # Token embedding takes ages!!!
        print("Here")

        # Prosody embeddings
        prosodic_feat = x["pros_feat"]  # shape:
        # (batch size, self.window_size, padded sequence length, self.n_prosodic_feat)
        batch_size, seq_len = prosodic_feat.size()[0], prosodic_feat.size()[2]
        prosodic_feat = prosodic_feat.reshape(
            (batch_size*self.window_size, seq_len, self.n_prosodic_feat)
        )  # combine batch and window dimensions in order to run features through LSTM in one go
        lstm_init_state = (
            torch.cat([self.h_0 for i in range(batch_size)], dim=1),
            torch.cat([self.c_0 for i in range(batch_size)], dim=1)
        )
        print("Here")
        pros_embeddings = self.rnn(
            prosodic_feat, lstm_init_state
        )[0][-1]  # get final output
        pros_embeddings = pros_embeddings.reshape(
            (batch_size, self.window_size, self.prosodic_embedding_size)
        )  # restore batch and window dimensions
        print("Here")

        # Remainder of forward pass
        concat_embeddings = torch.cat([token_embeddings, pros_embeddings], dim=-1)
        hidden_vecs = [self.hidden_layer(batch) for batch in concat_embeddings]
        hidden_vecs = relu(torch.stack(hidden_vecs))
        normalized = self.batch_norm(hidden_vecs)
        normalized = self.dropout_layer(normalized)
        probs = softmax(self.final_layer(normalized), dim=0)
        return probs

    def predict(self, speech_data):
        """Predict labels for a full speech using cross-context aggregation"""
        zero_tensor = torch.tensor((0. for label in POSSIBLE_LABELS))
        context_predictions = []
        for i in range(len(speech_data["tokens"] - self.window_size + 1)):
            window = {
                "tokens": speech_data["tokens"][i:i+self.window_size],
                "pros_feat": speech_data["pros_feat"][i:i + self.window_size]
            }
            context_predictions.append(self.forward([window])[0])
        aggregated_probs = context_predictions[0]
        for cp in context_predictions[1:]:
            aggregated_probs.append(zero_tensor)
            for i, j in zip(
                    range(len(aggregated_probs)-self.window_size, len(aggregated_probs)),
                    range(self.window_size)
            ):
                aggregated_probs[i] = aggregated_probs[i] + cp[j]
        return torch.stack(aggregated_probs)

