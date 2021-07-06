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

        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.encode = nn.Embedding(vocab_size, embedding_size)
        self.decode = nn.Linear(hidden_size, 1)
        self.recurrent = nn.LSTM(
            embedding_size, hidden_size, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        # TODO?: init_weights

    def forward(self, input, input_lengths):
        encoded = self.encode(input)
        encoded = self.dropout(encoded)
        packed = pack_padded_sequence(
            encoded, input_lengths, batch_first=True, enforce_sorted=False
        )
        _, (state, _) = self.recurrent(packed)
        # take state from the last layer of LSTM
        state = state[-1]
        state = self.dropout(state)
        decoded = self.decode(state)
        decoded = self.sigmoid(decoded)
        return decoded

    @staticmethod
    def save(model, name="model"):
        model_state = deepcopy(model.state_dict())
        torch.save(model_state, f"{name}.pt")

    @staticmethod
    def load(vocab_size, name="model"):
        assert os.path.exists(f"{name}.pt")

        model = ReviewClassifier(vocab_size)
        model.load_state_dict(torch.load(f"{name}.pt"))
        model.eval()

        return model


# -----------------------------------------------------------------------------


def _plot_loss(loss_values):
    import matplotlib.pyplot as plt
    plt.plot(range(1, len(loss_values)+1), loss_values)
    plt.show()


def _format_percentage(value, precision=2):
    return f"{round(value * 100, precision)}%"


def _extract_labels(labels):
    return [label.item() for label in labels]


def _extract_predictions(predictions, classify=round):
    """
        Argument `classify` determines how to map
        predictions given by the model (in range [0, 1])
        to a binary set of labels ({0, 1}).

        By default, rounding is used ([0, .5] -> 0, (.5, 1] -> 1)
    """
    return [classify(prediction.item()) for prediction in predictions]


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
            average_loss = sum(loss_values)/len(loss_values)
            print(f"Epoch #{epoch + 1}: {average_loss}")

    if graphic:
        _plot_loss(loss_values)


@utility.measure_time
def test(model, test_loader, device):
    # TODO?: provide metrics as an argument
    from sklearn.metrics import accuracy_score

    all_predictions = torch.tensor([])
    all_labels = []

    model.eval()
    for (reviews, labels, review_sizes) in test_loader:
        all_labels.extend(labels)

        reviews = reviews.to(device)
        predictions = model(reviews, review_sizes).reshape(-1)
        all_predictions = torch.cat((
            all_predictions, predictions.cpu().detach()
        ))

    predictions = _extract_predictions(all_predictions)
    labels = _extract_labels(all_labels)

    accuracy = accuracy_score(labels, predictions)
    print(f"Accuracy: {_format_percentage(accuracy)}")
    return accuracy


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import data_handler

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_loader = data_handler.get("train", cutoff=200)
    test_loader = data_handler.get("test", cutoff=20)

    model = ReviewClassifier(data_handler.vocab_size, layers=1)
    train(model, train_loader, device=device, verbose=True)
    test(model, test_loader, device=device)

    # ReviewClassifier.save(model)
    # new_model = ReviewClassifier.load(data_handler.vocab_size)
    # test(new_model.to(device), test_loader)
