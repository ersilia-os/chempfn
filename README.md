# Ensemble TabPFN

**TabPFN** is a transformer architecture proposed by [Hollman et al.](https://arxiv.org/abs/2207.01848) for classification on small tabular datasets. It is a Prior-Data Fitted Network (PFN) that has been trained once and does not require fine tuning for new datasets.

TabPFN works by approximating the distribution of new data to the prior synthetic data it has seen during training. In a machine learning pipeline, this network can be "fit" on a training dataset in under a second and can generate predictions for the test set in a single forward pass in the network.

With **EnsembleTabPFN**, we address some of the limitations of the original TabPFN model. It has been extended to work with datasets containing more than 1000 samples and 100 features, using data and feature subsampling strategies. EnsembleTabPFN is fully compatible with the [Scikit-learn API](https://scikit-learn.org/stable/index.html) and can be used in a modeling pipeline.

## Installation

### From source

```bash

git clone https://github.com/ersilia-os/ensemble-tabpfn.git
cd ensemble-tabpfn
pip install .
```

### From PyPI

```python
pip install ensemble-tabpfn
```

## Usage

```python

from ensemble_tabpfn import EnsembleTabPFN
from sklearn.metrics import accuracy_score

clf = EnsembleTabPFN()
clf.fit(X_train, y_train)
y_hat = clf.predict(y_test)
acc = accuracy_score(y_test, y_hat)
```

## Citation

If you use this package, please cite the [origianl authors](https://arxiv.org/abs/2207.01848) of the model and [this package](https://github.com/ersilia-os/ensemble-tabpfn/blob/master/CITATION.cff).

## License

This package is licensed under a GPL-3.0 license.
