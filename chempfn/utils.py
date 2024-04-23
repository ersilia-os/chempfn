import enum
import pandas as pd


class TabPFNConstants(enum.IntEnum):
    """Constants mapping TabPFN's limitations."""

    MAX_INP_SIZE: int = 1000
    MAX_FEAT_SIZE: int = 100

class AntiMicrobialsDatasetCutoff(enum.Enum):
    LOW = "lc"
    HIGH = "hc"

ANTI_MICROBIALS_DATASET_TYPES = {
    "organism": "org_all",
    "mic": "MIC",
    "activity": "Activity",
    "all": "all"
}

ANTIMICROBIAL_PATHOGENS = {
    "acinetobacter baumannii": "abaumannii",
    "campylobacter spp.": "campylobacter",
    "enterococcus faecium": "efaecium",
    "enterobacter spp.": "enterobacter",
    "escherichia coli": "ecoli",
    "helicobacter pylori": "hpylori",
    "klebsiella pneumoniae": "kpneumoniae",
    "mycobacterium tuberculosis": "mtuberculosis",
    "neisseria gonorrhoeae": "ngonorrhoeae",
    "plasmodium spp.": "pfalciparum",
    "pseudomonas aeruginosa": "paeruginosa",
    "schistosoma mansoni": "smansoni",
    "staphylococcus aureus": "saureus",
    "streptococcus pneumoniae": "spneumoniae"
}


class AntiMicrobialsDatasetLoader:
    """Class to load the AntiMicrobial dataset from S3."""

    def __init__(self) -> None:
        self.base_url = "https://chempfn-data.s3.eu-central-1.amazonaws.com"
        self.folder = "data/new_processing"

    def _check_dataset_type(self, dataset_type: str) -> str:
        if dataset_type in ANTI_MICROBIALS_DATASET_TYPES:
            return ANTI_MICROBIALS_DATASET_TYPES[dataset_type]
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    def _check_pathogen(self, pathogen: str) -> str:
        if pathogen in ANTIMICROBIAL_PATHOGENS:
            return ANTIMICROBIAL_PATHOGENS[pathogen]
        if pathogen in ANTIMICROBIAL_PATHOGENS.values():
            return pathogen
        raise ValueError(f"Invalid pathogen: {pathogen}")
        
    def _check_cutoff(self, cutoff: str) -> str:
        if cutoff in [e.value for e in AntiMicrobialsDatasetCutoff]:
            return cutoff
        else:
            raise ValueError(f"Invalid cutoff: {cutoff}")

    def _validate_input(self, pathogen: str, dataset_type: str, cutoff: str) -> tuple:
        pathogen = self._check_pathogen(pathogen.lower())
        dataset_type = self._check_dataset_type(dataset_type.lower())
        cutoff = self._check_cutoff(cutoff.lower())
        return (pathogen, cutoff, dataset_type)

    def load(
        self,
        pathogen: str,
        cutoff: str = AntiMicrobialsDatasetCutoff.HIGH.value,
        dataset_type: str = ANTI_MICROBIALS_DATASET_TYPES['organism'],
    ) -> pd.DataFrame:
        """Load the AntiMicrobial dataset from S3.

        Parameters
        ----------
        pathogen : str
            Pathogen name.
        dataset_type : str
            One of ANTI_MICROBIALS_DATASET_TYPES
        cutoff : str
            One of AntiMicrobialsDatasetCutoff

        Returns
        -------
        pd.DataFrame
            AntiMicrobial dataset.
        """
        pathogen, cutoff, dataset_type = self._validate_input(pathogen, dataset_type, cutoff)
        url = f"{self.base_url}/{self.folder}/{pathogen}/{pathogen}_{dataset_type}_{cutoff}.csv"
        return pd.read_csv(url)

