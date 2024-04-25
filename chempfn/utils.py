import enum
from urllib.error import HTTPError
import pandas as pd
from typing import Optional


class TabPFNConstants(enum.IntEnum):
    """Constants mapping TabPFN's limitations."""

    MAX_INP_SIZE: int = 1000
    MAX_FEAT_SIZE: int = 100


class AntiMicrobialsDatasetCutoff(enum.Enum):
    LOW = "lc"
    HIGH = "hc"


ANTI_MICROBIALS_DATASET_TYPES = {
    "organism all": "org_all",
    "mic": "MIC",
    "activity": "Activity",
    "all": "all",
    "ic50": "IC50",
    "iz": "IZ",
    "inhibition": "Inhibition",
    "organism": "org",
    "protein": "prot",
    "protein all": "prot_all",
}

TOP_N_ASSAYS = {"0": "top_0", "1": "top_1", "2": "top_2", "": ""}

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
    "streptococcus pneumoniae": "spneumoniae",
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

    def _check_top_n_assays(self, num: str) -> str:
        if num in TOP_N_ASSAYS:
            return TOP_N_ASSAYS[num]
        else:
            raise ValueError(f"Invalid top_n_assays: {num}")

    def _validate_input(
        self, pathogen: str, dataset_type: str, cutoff: str, assay_num: str
    ) -> tuple:
        pathogen = self._check_pathogen(pathogen.lower())
        dataset_type = self._check_dataset_type(dataset_type.lower())
        cutoff = self._check_cutoff(cutoff.lower())
        assay_num = self._check_top_n_assays(assay_num)
        return (pathogen, cutoff, dataset_type, assay_num)

    def load(
        self,
        pathogen: str,
        cutoff: str = AntiMicrobialsDatasetCutoff.HIGH.value,
        dataset_type: str = ANTI_MICROBIALS_DATASET_TYPES["organism"],
        assay_num: str = "",
    ) -> Optional[pd.DataFrame]:
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
        _pathogen, _cutoff, _dataset_type, _assay_num = self._validate_input(
            pathogen, dataset_type, cutoff, assay_num
        )
        url = f"{self.base_url}/{self.folder}/{_pathogen}/{_pathogen}_{_dataset_type}_{_cutoff}_{_assay_num}.csv"
        try:
            df = pd.read_csv(url)
        except HTTPError:
            print(
                f"Dataset unavailable for Pathogen: {pathogen}, Dataset Type: {dataset_type}, cutoff: {cutoff}"
                + f", and Top {assay_num} assay."
                if assay_num
                else "."
            )
            return
        return df
