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
        bidirectional=False,
        dropout=.2
    ):
        super().__init__()

        if layers == 1:
            dropout = 0

        # store model hyperparameters so they can be
        # saved/loaded with the model itself
        self._hyperparameters = {
            "vocab_size": vocab_size,
            "embedding_size": embedding_size,
            "hidden_size": hidden_size,
            "layers": layers,
            "bidirectional": bidirectional,
            "dropout": dropout
        }

        # int(bidirectional) --> map {False, True} to {0, 1}
        directions = 1 + int(bidirectional)

        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.encode = nn.Embedding(vocab_size, embedding_size)
        self.decode = nn.Linear(directions * hidden_size, 1)
        self.recurrent = nn.LSTM(
            embedding_size, hidden_size, num_layers=layers, batch_first=True,
            dropout=dropout, bidirectional=bidirectional
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
        if self._hyperparameters["bidirectional"]:
            # TODO: check if this is correct
            state_1 = state[-1]
            state_2 = state[-2]
            state = torch.cat((state_1, state_2), dim=1)
        else:
            state = state[-1]

        state = self.dropout(state)
        decoded = self.decode(state)
        decoded = self.sigmoid(decoded)
        return decoded

    @staticmethod
    def save(model, name="model"):
        model_state = deepcopy(model.state_dict())
        torch.save(
            {
                "state": model_state,
                "hyperparameters": model._hyperparameters
            },
            f"{name}.pt"
        )

    @staticmethod
    def load(name="model"):
        assert os.path.exists(f"{name}.pt")

        model_data = torch.load(f"{name}.pt")
        model = ReviewClassifier(**model_data["hyperparameters"])
        model.load_state_dict(model_data["state"])
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

    try:
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

    # enable early exit
    except KeyboardInterrupt:
        print(
            f"Stopped training (during epoch {epoch + 1} of {epochs})."
        )


@utility.measure_time
def test(model, test_loader, device, output=True):
    # TODO?: provide metrics as an argument
    from sklearn.metrics import accuracy_score

    all_predictions = torch.tensor([])
    all_labels = []

    model.to(device)
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
    if output:
        print(f"Accuracy: {_format_percentage(accuracy)}")

    return accuracy


def compare_to_saved(model, test_loader, device, name="model"):
    """
        Compare performance of a given model with the saved (best) one - 
        the new one replaces it if it's better.
    """
    saved_exists = os.path.exists(f"{name}.pt")

    if saved_exists:
        print("Comparing models...")
        current_model_accuracy = test(model, test_loader, device, output=False)

        saved_model = ReviewClassifier.load(name)
        saved_model_accuracy = test(
            saved_model, test_loader, device, output=False
        )

    if (not saved_exists) or (current_model_accuracy > saved_model_accuracy):
        if saved_exists:
            message = f"New model has higher accuracy - \
{_format_percentage(current_model_accuracy)} compared to \
{_format_percentage(saved_model_accuracy)} of the old model. \
Saving new model to '{name}.pt' ."
        else:
            message = f"No model named '{name}.pt' was found - \
saving new model."

        print(message)
        ReviewClassifier.save(model, name)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import data_handler

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_loader = data_handler.get("train", cutoff=200)
    test_loader = data_handler.get("test", cutoff=20)

    model = ReviewClassifier(
        data_handler.vocab_size, layers=2, bidirectional=True
    )
    train(model, train_loader, device=device, verbose=True)
    test(model, test_loader, device=device)

    # ReviewClassifier.save(model)
    # new_model = ReviewClassifier.load()
    # test(new_model, test_loader, device)

    compare_to_saved(model, test_loader, device, name="test_model")
