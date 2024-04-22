import enum
import pandas as pd


class TabPFNConstants(enum.IntEnum):
    """Constants mapping TabPFN's limitations."""

    MAX_INP_SIZE: int = 1000
    MAX_FEAT_SIZE: int = 100


class AntiMicrobialsDatasetTypes(enum.Enum):
    ORGANISM = "org_all"
    MIC = "MIC"
    ACTIVITY = "Activity"
    ALL = "all"


class AntiMicrobialsDatasetCutoff(enum.Enum):
    LOW = "lc"
    HIGH = "hc"


class AntiMicrobialsPathogens(enum.Enum):
    ABAUMANNI = "abaumannii"
    CALBICANS = "calbicans"
    CAMPYLORBACTER = "campylorbacter"
    ECOLI = "ecoli"
    EFAECIUM = "efaecium"
    ENTEROBACTER = "enterobacter"
    HPYLORI = "hpylori"
    KPNEUMONIAE = "kpneumoniae"
    MTUBERCULOSIS = "mtuberculosis"
    NGONORRHOEAE = "ngonorrhoeae"
    PAERUGINOSA = "paeruginosa"
    PFALCIPARUM = "pfalciparum"
    SAUREUS = "saureus"
    SMANSONII = "smansonii"
    SPNEUMONIAE = "spneumoniae"


class AntiMicrobialsDatasetLoader:
    """Class to load the AntiMicrobial dataset from S3."""

    def __init__(self) -> None:
        self.base_url = "https://chempfn-data.s3.eu-central-1.amazonaws.com"
        self.folder = "data/new_processing"

    def _check_params(self, pathogen: str, dataset_type: str, cutoff: str) -> None:
        if pathogen not in [e.value for e in AntiMicrobialsPathogens]:
            print(AntiMicrobialsPathogens.values())
            raise ValueError(f"Invalid pathogen: {pathogen}")

        if dataset_type not in [e.value for e in AntiMicrobialsDatasetTypes]:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        if cutoff not in [e.value for e in AntiMicrobialsDatasetCutoff]:
            raise ValueError(f"Invalid cutoff: {cutoff}")

    def load(
        self,
        pathogen: str,
        cutoff: str = AntiMicrobialsDatasetCutoff.HIGH.value,
        dataset_type: str = AntiMicrobialsDatasetTypes.ORGANISM.value,
    ) -> pd.DataFrame:
        """Load the AntiMicrobial dataset from S3.

        Parameters
        ----------
        pathogen : str
            Pathogen name.
        dataset_type : str
            One of AntiMicrobialsDatasetTypes
        cutoff : str
            One of AntiMicrobialsDatasetCutoff

        Returns
        -------
        pd.DataFrame
            AntiMicrobial dataset.
        """
        self._check_params(pathogen, dataset_type, cutoff)
        url = f"{self.base_url}/{self.folder}/{pathogen}/{pathogen}_{dataset_type}_{cutoff}.csv"
        return pd.read_csv(url)

