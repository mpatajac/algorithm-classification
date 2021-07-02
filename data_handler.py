import os
import utility
import re
import string
from num2words import num2words

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


def _remove_br(reviews):
    return list(
        map(
            # pretty sure there are always two `br`s in a row
            # but removing them individually, just in case
            lambda review: re.sub("<br />", " ", review), reviews
        )
    )


def _remove_puctuation(reviews):
    # we want to remove all punctuation
    # except for dashes and apostrophes
    characters_to_remove = re.sub("-|'", "", string.punctuation)
    return list(
        map(
            lambda review: re.sub(
                f"[{characters_to_remove}]", "", review
            ), reviews
        )
    )


def _collapse_spaces(reviews):
    return list(
        map(
            # remove multiple successive whitespace characters
            # with a single space
            lambda review: re.sub("\s+", " ", review), reviews
        )
    )


def _lower_text(reviews):
    return list(map(str.lower, reviews))


def _split_words(reviews):
    return list(
        map(
            lambda review: review.split(), reviews
        )
    )


def _map_numbers(reviews):
    # replace all numbers with their word representations
    return list(
        map(
            lambda review: re.sub(
                "\d+", lambda n: num2words(n.group(0)), review
            ), reviews
        )
    )


@utility.measure_time
def clean(reviews):
    return utility.pipe(
        reviews,
        _remove_br,
        _remove_puctuation,
        _collapse_spaces,
        _lower_text,
        _map_numbers,
        _split_words,
    )


# -----------------------------------------------------------------------------

def _build_dictionary():
    with open("./data/imdb.vocab", "r", encoding="utf-8") as vocab:
        # remove `\n` from the end of each word
        words = list(map(str.strip, vocab.readlines()))

        # assign `<pad>` to index 0 (to use 0 as pad value)
        i2w = ['<pad>'] + words + ['<unk>']

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


if __name__ == "__main__":
    reviews = utility.pipe(
        {"mode": "train", "cutoff": 2},
        load,
        clean,
        index
    )

    [print(review) for review in reviews]
