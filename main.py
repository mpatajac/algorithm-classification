import argparse
import torch

import utility
import data_handler
import model
from model import AlgorithmClassifier

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--dropout", type=float, default=.2)
parser.add_argument("--layers", type=int, default=1)
parser.add_argument("--embedded", type=int, default=100)
parser.add_argument("--hidden", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--save-name", type=str, default="model")
parser.add_argument(
    "--attention", type=str, choices=["sum", "cat"], default="cat"
)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--bidirectional", action="store_true")
parser.add_argument("--evaluate", action="store_true")

args = parser.parse_args()

# -----------------------------------------------------------------------------


@utility.measure_time
def evaluate(name="model", set="test"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_loader = data_handler.get(set)
    classifier = AlgorithmClassifier.load(name)
    model.test(classifier, test_loader, device)


@utility.measure_time
def main():
    global args

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_loader = data_handler.get("train", batch_size=args.batch_size)
    validation_loader = data_handler.get("val", batch_size=args.batch_size)
    test_loader = data_handler.get("test", batch_size=args.batch_size)

    classifier = AlgorithmClassifier(
        data_handler.vocab_size,
        embedding_size=args.embedded,
        hidden_size=args.hidden,
        layers=args.layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        attention=args.attention
    )

    model.train(
        classifier,
        train_loader,
        validation_loader=validation_loader,
        device=device,
        epochs=args.epochs,
        save_name=args.save_name,
        verbose=args.verbose
    )

    model.test(classifier, test_loader, device)


if __name__ == "__main__":
    if args.evaluate:
        evaluate(name=args.save_name)
    else:
        main()
