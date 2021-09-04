# Algorithm classification

This repository contains the source code for the ***Source code classification problem***, a paper in which we attempt to determine
the functionality of a code snippet using programming language processing. We propose a solution in the form of a neural network based on the long short-term memory (LSTM) recurrent network and compare our results to several other papers that had discussed this topic. You can read the full paper [here](.) (TBA).


## Usage
Here are a few examples of common ways one would use this program:

1. Train a model with default hyperparameters for 5 epochs with verbose output, save the model as `test_model.pt`.
```
python main.py --epochs=5 --save-name=test_model -v
```

2. Train a model with a bidirectional, 3-layered LSTM.
```
python main.py --layers=3 --bidirectional
```

3. Evaluate an existing model (`mymodel.pt`)
```
python main.py --evaluate --save-name=mymodel
```


For a detailed explanation of the usage and implementation details, please read the **technical documentation** (`docs` subdirectory).

