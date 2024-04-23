from typing import Optional
import time 

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import numpy as np
from tqdm import tqdm
from eosce.models import ErsiliaCompoundEmbeddings

from .ensemble_tabpfn import EnsembleTabPFN


class ChemPFN:
    def __init__(
        self,
        max_iters: int = 10,
        random_state: Optional[int] = None,
        early_stopping_rounds: int = 5,
        tolerance: float = 1e-2,
        verbose: bool = False
    ) -> None:
        self.etpfn = EnsembleTabPFN(
            max_iters=max_iters,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            tolerance=tolerance,  
            verbose=verbose,
            baseline=False 
        )
        self.eosce = ErsiliaCompoundEmbeddings()

    def fit(self, smiles_list, y):
        self.etpfn.fit(self.eosce.transform(smiles_list), y)

    def predict(self, smiles_list):
        return self.etpfn.predict(self.eosce.transform(smiles_list))

    def predict_proba(self, smiles_list):
        return self.etpfn.predict_proba(self.eosce.transform(smiles_list))

    def evaluate(self, smiles_list, y):
        data = {}
        data["n_pos"] = int(np.sum(y))
        data["n_neg"] = len(y) - len(np.sum(y))
        splitter = StratifiedKFold(shuffle=True, random_state=42, n_splits=5)
        aurocs = []
        t0 = time.time()
        for train_idx, test_idx in tqdm(splitter.split([i for i in range(len(smiles_list))], y)):
            train_smiles = [smiles_list[idx] for idx in train_idx]
            test_smiles = [smiles_list[idx] for idx in test_idx]
            train_y = [y[idx] for idx in train_idx]
            test_y = [y[idx] for idx in test_idx]
            self.fit(train_smiles, train_y)
            y_hat = self.predict_proba(test_smiles)[:,1]
            fpr, tpr, _ = roc_curve(test_y, y_hat)
            auroc = auc(fpr, tpr)
            aurocs += [auroc]
        t1 = time.time()
        data["auroc_mean"] = np.mean(aurocs)
        data["auroc_std"] = np.std(aurocs)
        data["time_elapsed_sec"] = t1 - t0
        return data
            
