import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class ReviewClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=100,
        hidden_size=200,
        layers=1,
        dropout=.2
    ):
        super().__init__()
        if layers == 1:
            dropout = 0

        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.encode = nn.Embedding(vocab_size, embedding_size)
        self.decode = nn.Linear(hidden_size, 1)
        self.recurrent = nn.LSTM(
            embedding_size, hidden_size, num_layers=layers, batch_first=True,
            dropout=dropout
        )
        # TODO?: init_weights

    def forward(self, input, input_lengths):
        encoded = self.encode(input)
        encoded = self.dropout(encoded)
        packed = pack_padded_sequence(
            encoded, input_lengths, batch_first=True, enforce_sorted=False
        )
        _, (state, _) = self.recurrent(packed)
        state = self.dropout(state)
        decoded = self.decode(state)
        decoded = self.sigmoid(decoded)
        return decoded


if __name__ == "__main__":
    import data_handler

    train_loader = data_handler.get("train", cutoff=2)
    model = ReviewClassifier(data_handler.vocab_size)

    for (reviews, labels, review_sizes) in train_loader:
        result = model(reviews, review_sizes)
        print(f"Result:\t{result}")
        print(f"\nLabels:\t{labels}")
        print("\nLoss:", end="\t")
        print(nn.BCELoss()(
            result.reshape(-1), torch.tensor(labels, dtype=torch.float)
        ))
