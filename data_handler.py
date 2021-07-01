import os

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


if __name__ == "__main__":
    reviews = load({"mode": "train", "cutoff": 2})
    [print(review + '\n') for review in reviews]
