from typing import Optional
from .ensemble_tabpfn import EnsembleTabPFN
from eosce.models import ErsiliaCompoundEmbeddings


class ChemPFN:
    def __init__(
        self,
        max_iters: int = 10,
        random_state: Optional[int] = None,
        early_stopping_rounds: int = 5,
        tolerance: float = 1e-2,
        verbose: bool = True
    ) -> None:
        self.etpfn = EnsembleTabPFN(
            max_iters=max_iters,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            tolerance=tolerance,  
            verbose=verbose,  
        )
        self.eosce = ErsiliaCompoundEmbeddings()

    def fit(self, smiles_list, y):
        self.etpfn.fit(self.eosce.transform(smiles_list), y)

    def predict(self, smiles_list):
        return self.etpfn.predict(self.eosce.transform(smiles_list))

    def predict_proba(self, smiles_list):
        return self.etpfn.predict_proba(self.eosce.transform(smiles_list))
