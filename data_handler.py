import os
import utility
import torch
import pickle
from utility import base_path
from itertools import chain
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# globals
vocab_size = 8566

# -----------------------------------------------------------------------------


@utility.measure_time
def prepare(set):
    assert set in ["train", "test", "val"], "Invalid data set."
    assert os.path.exists(f".{base_path}/data/seq_{set}"), \
        f"Can't find 'seq_{set}', please create it using `task_utils.llvm_ir_to_trainable(set)`."

    all_indices = [[] for _ in range(104)]
    category_count = [0 for _ in range(104)]
    root_directory_name = f".{base_path}/data/seq_{set}"

    # collect
    directories = os.listdir(root_directory_name)
    for directory in directories:
        # use 0-index
        category = int(directory) - 1

        files = os.listdir(f"{root_directory_name}/{directory}")
        files_in_category = len(files)
        category_count[category] = files_in_category

        directory_indices = []
        for file in files:
            with open(f"{root_directory_name}/{directory}/{file}", 'r') as f:
                indices = [int(line.strip()) for line in f.readlines()]
                directory_indices.append(indices)

        all_indices[category] = directory_indices

    # flatten
    all_indices = list(chain.from_iterable(all_indices))

    # save
    with open(f".{base_path}/data/{set}_data.pt", "wb") as f:
        pickle.dump({
            "indices": all_indices,
            "category_count": category_count
        }, f, -1)


# -----------------------------------------------------------------------------


class AlgorithmDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def pad_collate(batch):
    (algorithms, categories) = zip(*batch)

    # get original lengths for packing
    algorithm_lengths = [len(algorithm) for algorithm in algorithms]

    # pad sequences to longest in batch
    padded_algorithms = pad_sequence(
        algorithms, batch_first=True, padding_value=0
    )

    return padded_algorithms, categories, algorithm_lengths


def _create_categories(category_count):
    nested_categories = [
        [category for _ in range(count)]
        for (category, count) in enumerate(category_count)
    ]
    # flatten the nested categories
    return list(chain.from_iterable(nested_categories))


def _convert_to_tensor(algorithms, categories):
    algorithms = [torch.tensor(algorithm) for algorithm in algorithms]
    categories = torch.tensor(categories)

    return algorithms, categories


def to_loader(algorithms, category_count, batch_size=64):
    categories = _create_categories(category_count)
    algorithms, categories = _convert_to_tensor(algorithms, categories)
    dataset = AlgorithmDataset(algorithms, categories)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate
    )

    return dataloader


@utility.measure_time
def get(set, batch_size=64):
    assert set in ["train", "test", "val"]
    assert os.path.exists(f".{base_path}/data/{set}_data.pt"), \
        f"Can't find '{set}_data.pt', please prepare it using function `prepare(set)`."

    with open(f".{base_path}/data/{set}_data.pt", "rb") as f:
        stored_data = pickle.load(f)
        indices = stored_data["indices"]
        category_count = stored_data["category_count"]

    return to_loader(indices, category_count, batch_size)

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    prepare("test")
    # loader = get("test")
    # for batch in loader:
    #     indices, categories, _ = batch
    #     print(indices[0], categories[0])
    #     break
