# Ensemble TabPFN

TabPFN is a transformer architecture prosposed by [Hollman et al](https://arxiv.org/abs/2207.01848) for classification on small tabular datasets. It is a Prior-Data Fitted Network that has been trained once and does not require fine tuning for new datasets. It works by approximating the distribution of new data to the prior synthetic data it has seen during training. In a machine learning pipeline, this network can be "fit" on a training dataset in under a second and can generate predictions for the test set in a single forward pass in the network. However there are limitations in the current architecture, namely, the training dataset can contain only upto 1000 inputs with upto 100 numerical features. In addition, the network can predict only upto 10 classes in a multi-class classification problem. With EnsembleTabPFN, we address two of these issues where we have extended the original model to work with datasets containing more than 1000 samples and 100 features, using data and feature subsampling strategies.
EnsembleTabPFN is fully compatible with Scikit-learn API and can be used in a modelling pipeline.


# Installation

## From source

```bash

git clone https://github.com/ersilia-os/ensemble-tabpfn.git
cd ensemble-tabpfn
pip install .
```

## From PyPI

```python
pip install ensemble-tabpfn
```

## Using Poetry

```python

git clone https://github.com/ersilia-os/ensemble-tabpfn.git
cd ensemble-tabpfn
poetry install --without dev,test,docs
```

# Usage

```python

from ensemble_tabpfn import EnsembleTabPFN
from sklearn.metrics import accuracy_score

clf = EnsembleTabPFN()
clf.fit(X_train, y_train)
y_hat = clf.predict(y_test)
acc = accuracy_score(y_test, y_hat)
```
