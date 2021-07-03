import torch
import os
import utility
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from copy import deepcopy


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

    @staticmethod
    def save(model, name="model"):
        model_state = deepcopy(model.state_dict())
        torch.save(model_state, f"{name}.pt")

    @staticmethod
    def load(model, name="model"):
        assert os.path.exists(f"{name}.pt")
        model.load_state_dict(torch.load(f"{name}.pt"))
        model.eval()


# -----------------------------------------------------------------------------


def _plot_loss(loss_values):
    import matplotlib.pyplot as plt
    plt.plot(range(1, len(loss_values)+1), loss_values)
    plt.show()


@utility.measure_time
def train(
    model,
    train_loader,
    epochs=2,
    device="cpu",
    verbose=False,
    graphic=False
):
    loss_values = []
    loss_fn = nn.BCELoss()

    model.to(device)
    model.train()
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=1e-3, momentum=.9
    )

    for epoch in range(epochs):
        for (reviews, labels, review_sizes) in train_loader:
            reviews = reviews.to(device)
            labels = torch.tensor(labels, dtype=torch.float).to(device)

            predictions = model(reviews, review_sizes).reshape(-1)
            loss = loss_fn(predictions, labels)
            loss_values.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose:
            print(f"Epoch #{epoch + 1}: {loss.item()}")

    if graphic:
        _plot_loss(loss_values)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import data_handler

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_loader = data_handler.get("train", cutoff=200)
    test_loader = data_handler.get("test", cutoff=20)

    model = ReviewClassifier(data_handler.vocab_size)
    train(model, train_loader, device=device, verbose=True, graphic=True)

    model.eval()
    for (reviews, labels, review_sizes) in test_loader:
        result = model(reviews.to(device), review_sizes)
        print("\nLoss:", end="\t")
        print(nn.BCELoss()(
            result.reshape(-1),
            torch.tensor(labels, dtype=torch.float).to(device)
        ))

    ReviewClassifier.save(model)

    new_model = ReviewClassifier(data_handler.vocab_size)
    ReviewClassifier.load(new_model)
