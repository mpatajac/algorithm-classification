import os
import utility
import re
import string
import torch
from num2words import num2words
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# globals
vocab_size = 0

# -----------------------------------------------------------------------------


def _read_files(mode, sentiment, cutoff):
    assert mode in ["train", "test"]
    assert sentiment in ["pos", "neg"]

    data_path = f"./data/{mode}/{sentiment}"
    file_names = os.listdir(data_path)[:cutoff]

    reviews = []
    for file_name in file_names:
        with open(f"{data_path}/{file_name}", 'r', encoding="utf-8") as f:
            reviews.append(f.read())

    return reviews


@utility.measure_time
def load(kwargs):
    """
            Load reviews into a list.
            First half of the list are positive reviews,
            and the second are the negative ones.

            `kwargs`:
            mode:	"train"  |  "test"
            cutoff:	[1, 25k] |  None
    """

    assert "mode" in kwargs.keys(), "Expected keyword argument `mode`."
    assert kwargs["mode"] in [
        "train", "test"], f"Invalid mode: expected \"train\" or \"test\", got \"{kwargs['mode']}\"."

    mode = kwargs["mode"]
    cutoff = int(kwargs["cutoff"]) if "cutoff" in kwargs.keys() else 25000

    return _read_files(mode, "pos", cutoff) + _read_files(mode, "neg", cutoff)

# -----------------------------------------------------------------------------


def _remove_br(review):
    # pretty sure there are always two `br`s in a row
    # but removing them individually, just in case
    return re.sub("<br />", " ", review)


def _remove_puctuation(review):
    # we want to remove all punctuation
    # except for dashes and apostrophes
    characters_to_remove = re.sub("-|'", "", string.punctuation)
    return re.sub(f"[{characters_to_remove}]", "", review)


def _collapse_spaces(review):
    # replace multiple successive whitespace characters
    # with a single space
    return re.sub("\s+", " ", review)


def _lower_text(review):
    return review.lower()


def _map_numbers(review):
    # replace all numbers with their word representations
    return re.sub("\d+", lambda n: num2words(n.group(0)), review)


def _split_words(review):
    return review.split()


@utility.measure_time
def clean(reviews):
    return list(utility.pipe_map(
        reviews,
        _remove_br,
        _remove_puctuation,
        _collapse_spaces,
        _lower_text,
        _map_numbers,
        _split_words,
    ))


# -----------------------------------------------------------------------------

def _build_dictionary():
    global vocab_size

    with open("./data/imdb.vocab", "r", encoding="utf-8") as vocab:
        # remove `\n` from the end of each word
        words = list(map(str.strip, vocab.readlines()))

        # assign `<pad>` to index 0 (to use 0 as pad value)
        i2w = ['<pad>'] + words + ['<unk>']
        vocab_size = len(i2w)

        w2i = {word: idx for (idx, word) in enumerate(i2w)}

    return w2i


def _map_to_indices(reviews, word_mapping):
    return list(map(
        lambda review: list(map(
            lambda word:
                word_mapping[
                    word if word in word_mapping.keys() else "<unk>"
                ], review
        )), reviews
    ))


@utility.measure_time
def index(reviews):
    # since we are doing sequence classification,
    # we only need word-to-index mapping
    word_mapping = _build_dictionary()
    return _map_to_indices(reviews, word_mapping)

# -----------------------------------------------------------------------------


class ReviewDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def pad_collate(batch):
    (reviews, sentiments) = zip(*batch)

    # get original lengths for packing
    review_lengths = [len(review) for review in reviews]

    # pad sequences to longest in batch
    # TODO?: set `padding_value` to `w2i["<pad>"]`
    padded_reviews = pad_sequence(reviews, batch_first=True, padding_value=0)

    return padded_reviews, sentiments, review_lengths


def _create_labels(review_count):
    # first half of the reviews are positive, other half is negative
    per_category = review_count // 2
    return [1 for _ in range(per_category)] + [0 for _ in range(per_category)]


def _convert_to_tensor(reviews, labels):
    reviews = [torch.tensor(review) for review in reviews]
    labels = torch.tensor(labels)

    return reviews, labels


@utility.measure_time
def to_loader(reviews):
    labels = _create_labels(len(reviews))
    reviews, labels = _convert_to_tensor(reviews, labels)
    dataset = ReviewDataset(reviews, labels)

    # TODO?: `batch_size` as a variable
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=pad_collate
    )

    return dataloader


# -----------------------------------------------------------------------------


@utility.measure_time
def get(mode, cutoff=25000):
    assert mode in ["train", "test"]
    assert 1 <= cutoff <= 25000

    # TODO?: save and load data(loader) to/from file
    return utility.pipe(
        {
            "mode": mode,
            "cutoff": cutoff
        },
        load,
        clean,
        index,
        to_loader
    )

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # loader = get(mode="train")
    loader = get(mode="train", cutoff=2)
