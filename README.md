# ChemPFN

**TabPFN** is a transformer architecture proposed by [Hollman et al.](https://arxiv.org/abs/2207.01848) for classification on small tabular datasets. It is a Prior-Data Fitted Network (PFN) that has been trained once and does not require fine tuning for new datasets.

TabPFN works by approximating the distribution of new data to the prior synthetic data it has seen during training. In a machine learning pipeline, this network can be "fit" on a training dataset in under a second and can generate predictions for the test set in a single forward pass in the network.

With **ChemPFN**, we address some of the limitations of the original TabPFN model and extend it to work with Chemical datasets using [Ersilia Compound Embeddings](https://pypi.org/project/eosce/). Using data and feature subsampling strategies, ChemPFN bypasses the limitation of 1000 rows and 100 features inherent in TabPFN. It is fully compatible with the [Scikit-learn API](https://scikit-learn.org/stable/index.html) and can be used in a modeling pipeline like any Scikit-learn estimator.

ChemPFN, when fit, creates ensembles of data points and input dimenions, if required. During the predict stage, it creates an ensemble of TabPFN models fit on the training set to generate predictions for the test set. These intermediate ensemble results are then aggregated to produce the final prediction. With this approach, the model is able to fit in under a second, however predictions can be slow based on configuration ([see below](https://github.com/ersilia-os/ensemble-tabpfn/blob/main/README.md#usage)), or the underlying hardware.

This model can be used directly with SMILES data without the need for prior featurization. Additionally, we provided a utility to explore this model on Antimicrobials dataset from ChEMBL.


## Installation

```bash

git clone https://github.com/ersilia-os/chempfn.git
cd chempfn
pip install .
```

## Usage

By default, ChemPFN generates 100 data samples of size 1000 each to work with TabPFN. This can be configured to a lower number (for example, `max_iters=10`) to speeed up prediction. 

```python

from chempfn import ChemPFN
from sklearn.metrics import accuracy_score

clf = ChemPFN(max_iters=100)
clf.fit(X_train, y_train)
y_hat = clf.predict(y_test)
acc = accuracy_score(y_test, y_hat)
```

### Explore Antimicrobial Datasets

We provide a utility class to retrieve pre processed antimicrobial datasets. We list below the pathogens that are currently supported. For each pathogen, we allow the user to select a confidence level (hc or lc) for obtaining the assay activity.

- Acinetobacter baumannii
- Campylobacter spp.
- Enterococcus faecium
- Enterobacter spp.
- Escherichia coli
- Helicobacter pylori
- Klebsiella pneumoniae
- Mycobacterium tuberculosis
- Neisseria gonorrhoeae
- Plasmodium spp.
- Pseudomonas aeruginosa
- Schistosoma mansoni
- Staphylococcus aureus
- Streptococcus pneumoniae

```python
# Import the dataset loader
from chempfn.utils import AntiMicrobialsDatasetLoader

dataset_loader = AntiMicrobialsDatasetLoader()
df = dataset_loader.load('ecoli', 'hc')
```

## Citation

If you use this package, please cite the [original authors](https://arxiv.org/abs/2207.01848) of the model and [this package](https://github.com/ersilia-os/chempfn/blob/master/CITATION.cff).

## License

This package is licensed under a GPL-3.0 license.
