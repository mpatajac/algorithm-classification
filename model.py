import torch
import os
import utility
from utility import base_path
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from copy import deepcopy


class Attention(nn.Module):
    # https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/models/text/classifier/rnn.py
    def __init__(self, hidden_size):
        super().__init__()
        self.concat_linear = nn.Linear(2 * hidden_size, hidden_size)
        self.alignment_function = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, recurrent_outputs, recurrent_hidden):
        alignment_scores = self.alignment_function(recurrent_outputs)
        alignment_scores = torch.bmm(
            alignment_scores, recurrent_hidden.unsqueeze(2)
        )
        attention_weights = self.softmax(alignment_scores.squeeze(2))

        context = torch.bmm(
            recurrent_outputs.transpose(1, 2), attention_weights.unsqueeze(2)
        ).squeeze(2)
        attention_output = torch.cat((context, recurrent_hidden), dim=1)
        attention_output = torch.tanh(self.concat_linear(attention_output))

        return attention_output, attention_weights


class AlgorithmClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=100,
        hidden_size=200,
        layers=1,
        bidirectional=False,
        dropout=.2,
        attention="cat"
    ):
        super().__init__()

        assert attention in ["sum", "cat"]

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
            "dropout": dropout,
            "attention": attention
        }

        # int(b: bool) --> map {False, True} to {0, 1}
        directions = 1 + int(bidirectional)
        attention_multiplier = 1 + int(attention == "cat")

        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attention = Attention(directions * hidden_size)
        self.decode = nn.Linear(
            attention_multiplier * directions * hidden_size, 1
        )
        self.recurrent = nn.LSTM(
            embedding_size, hidden_size, num_layers=layers, batch_first=True,
            dropout=dropout, bidirectional=bidirectional
        )
        # TODO?: init_weights

    def forward(self, input, input_lengths):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        packed = pack_padded_sequence(
            embedded, input_lengths, batch_first=True, enforce_sorted=False
        )
        output, (state, _) = self.recurrent(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # take state from the last layer of LSTM
        if self._hyperparameters["bidirectional"]:
            state = torch.cat((state[-1], state[-2]), dim=1)
        else:
            state = state[-1]
        state = self.dropout(state)

        # TODO?: use `attention_weights` for visualization
        attention_output, _ = self.attention(output, state)

        if self._hyperparameters["attention"] == "cat":
            state = torch.cat((state, attention_output), dim=1)
        else:
            state = state + attention_output

        decoded = self.decode(state)
        output = self.sigmoid(decoded)
        return output

    @staticmethod
    def save(model, name="model"):
        model_state = deepcopy(model.state_dict())
        torch.save(
            {
                "state": model_state,
                "hyperparameters": model._hyperparameters
            },
            f".{base_path}/models/{name}.pt"
        )

    @staticmethod
    def load(name="model"):
        assert os.path.exists(f".{base_path}/models/{name}.pt")

        model_data = torch.load(f".{base_path}/models/{name}.pt")
        model = AlgorithmClassifier(**model_data["hyperparameters"])
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


def _extract_categories(categories):
    return [category.item() for category in categories]


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
    loss_fn = nn.BCELoss()

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    try:
        for epoch in range(epochs):
            loss_values = []

            for (algorithms, categories, algorithm_sizes) in train_loader:
                algorithms = algorithms.to(device)
                categories = torch.tensor(
                    categories, dtype=torch.float
                ).to(device)

                predictions = model(algorithms, algorithm_sizes).reshape(-1)
                loss = loss_fn(predictions, categories)
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
    all_categories = []

    model.to(device)
    model.eval()
    for (algorithms, categories, algorithm_sizes) in test_loader:
        all_categories.extend(categories)

        algorithms = algorithms.to(device)
        predictions = model(algorithms, algorithm_sizes).reshape(-1)
        all_predictions = torch.cat((
            all_predictions, predictions.cpu().detach()
        ))

    predictions = _extract_predictions(all_predictions)
    categories = _extract_categories(all_categories)

    accuracy = accuracy_score(categories, predictions)
    if output:
        print(f"Accuracy: {_format_percentage(accuracy)}")

    return accuracy


def compare_to_saved(model, test_loader, device, name="model"):
    """
        Compare performance of a given model with the saved (best) one - 
        the new one replaces it if it's better.
    """
    saved_exists = os.path.exists(f".{base_path}/models/{name}.pt")

    if saved_exists:
        print("Comparing models...")
        current_model_accuracy = test(model, test_loader, device, output=False)

        saved_model = AlgorithmClassifier.load(name)
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
        AlgorithmClassifier.save(model, name)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import data_handler

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_loader = data_handler.get("train")
    test_loader = data_handler.get("test")

    model = AlgorithmClassifier(
        data_handler.vocab_size, layers=2, bidirectional=True
    )
    train(model, train_loader, device=device, verbose=True)
    test(model, test_loader, device=device)

    # ReviewClassifier.save(model)
    # new_model = ReviewClassifier.load()
    # test(new_model, test_loader, device)

    compare_to_saved(model, test_loader, device, name="test_model")
