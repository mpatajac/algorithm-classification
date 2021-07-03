import argparse
import torch

import utility
import data_handler
import model
from model import ReviewClassifier

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--data-cutoff", type=int, default=25000)
parser.add_argument("--dropout", type=float, default=.2)
parser.add_argument("--layers", type=int, default=1)
parser.add_argument("--embedded", type=int, default=100)
parser.add_argument("--hidden", type=int, default=200)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--save-name", type=str, default="model")
parser.add_argument("-v", "--verbose", action="store_true")

args = parser.parse_args()

# -----------------------------------------------------------------------------


@utility.measure_time
def main():
    global args

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_loader = data_handler.get("train", cutoff=args.data_cutoff)
    test_loader = data_handler.get("test", cutoff=args.data_cutoff)

    classifier = ReviewClassifier(
        data_handler.vocab_size,
        embedding_size=args.embedded,
        hidden_size=args.hidden,
        layers=args.layers,
        dropout=args.dropout
    )

    model.train(
        classifier,
        train_loader,
        device=device,
        epochs=args.epochs,
        verbose=args.verbose
    )

    model.test(classifier, test_loader, device)

    ReviewClassifier.save(classifier, args.save_name)


if __name__ == "__main__":
    main()