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
    cutoff = int(kwargs["cutoff"]) if kwargs["cutoff"] else 25000

    return _read_files(mode, "pos", cutoff) + _read_files(mode, "neg", cutoff)

# -----------------------------------------------------------------------------


def _remove_br(reviews):
    return list(
        map(
            # pretty sure there are always two `br`s in a row
            # but removing them individually, just in case
            lambda review: re.sub("<br />", "", review), reviews
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
                "\d+", num2words(review.group(0)), review
            ), reviews
        )
    )


def clean(reviews):
    return utility.pipe(
        reviews,
        _remove_br,
        _remove_puctuation,
        _collapse_spaces,
        str.lower,
        _split_words,
        _map_numbers
    )


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    reviews = load({"mode": "train", "cutoff": 2})
    [print(review + '\n') for review in reviews]
