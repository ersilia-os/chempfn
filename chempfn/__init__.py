from .ensemble_tabpfn import EnsembleTabPFN
from eosce.models import ErsiliaCompoundEmbeddings

class ChemPFN:
    def __init__(self):
        self.etpfn = EnsembleTabPFN(random_state=42)
        self.eosce = ErsiliaCompoundEmbeddings()

    def fit(self, smiles_list, y):
        self.etpfn.fit(self.eosce.transform(smiles_list), y)

    def predict(self, smiles_list):
        return self.etpfn.predict(self.eosce.transform(smiles_list))

    def predict_proba(self, smiles_list):
        return self.etpfn.predict_proba(self.eosce.transform(smiles_list))