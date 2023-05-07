import os
import streamlit as st
from eosce.models import ErsiliaCompoundEmbeddings
from lol import LOL
from tabpfn import TabPFNClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
TEXT_AREA_HEIGHT = os.environ.get("text_height", 400)

st.set_page_config(
    page_title="Ensemble TabPFN",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_example():
    example_file = os.path.join(ROOT, "examples_carcinogen_tdc.csv")
    return pd.read_csv(example_file)


@st.cache_data
def read_doc():
    with open(os.path.join(ROOT, "doc.md"), "r") as f:
        return f.read()


@st.cache_resource
def load_embedder():
    return ErsiliaCompoundEmbeddings()


@st.cache_resource
def load_tabpfn():
    return TabPFNClassifier(device="cpu")


@st.cache_resource
def load_lolp():
    return LOL(n_components=100)


dex = load_example()
clf = load_tabpfn()
reducer = load_lolp()
embedder = load_embedder()


st.title("Fast molecular property prediction")

st.sidebar.title("Ensemble-TabPFN")


doc = read_doc()
st.sidebar.markdown(doc)

st.sidebar.header("About us")
st.sidebar.markdown(
    ":earth_africa: The [Ersilia Open Source Initiative](https://ersilia.io) is a non-profit organisations aimed at building AI/ML capacity in the Global South."
)


def filter_molecules(mols):
    valid = []
    for m in mols:
        mol = Chem.MolFromSmiles(m)
        if mol is None:
            st.warning("Molecule {0} is not valid".format(m))
            continue
        valid += [m]
    valid = [x for x in valid if x != ""]
    return valid


cols = st.columns(3)

mols_act = (
    cols[0]
    .text_area("Active molecules", height=TEXT_AREA_HEIGHT)
    .split(os.linesep)
)

mols_inact = (
    cols[1]
    .text_area("Inactive molecules", height=TEXT_AREA_HEIGHT)
    .split(os.linesep)
)
mols_query = (
    cols[2]
    .text_area("Query molecules", height=TEXT_AREA_HEIGHT)
    .split(os.linesep)
)

mols_act = filter_molecules(mols_act)
mols_inact = filter_molecules(mols_inact)
mols_query = filter_molecules(mols_query)

if len(mols_act) == 0 or len(mols_inact) == 0 or len(mols_query) == 0:
    st.info(
        "Please input molecules in the text boxes above. Use SMILES strings. Below is an example from a Therapeutic Data Commons (carcinogens dataset). Copy-paste the molecule lists to try it out!"
    )
    cols = st.columns(3)
    cols[0].text("\n".join(dex[dex["act"] == 1].head(10)["smiles"]))
    cols[1].text("\n".join(dex[dex["act"] == -1].head(10)["smiles"]))
    cols[2].text("\n".join(dex[dex["act"] == 0].head(10)["smiles"]))


else:
    smiles_train = mols_act + mols_inact
    y_train = [1] * len(mols_act) + [0] * len(mols_inact)
    smiles_query = mols_query

    X_train = embedder.transform(smiles_train)
    X_query = embedder.transform(smiles_query)

    reducer.fit(X_train, y_train)
    X_train = reducer.transform(X_train)
    X_query = reducer.transform(X_query)

    clf.fit(X_train, y_train)
    y_hat = clf.predict_proba(X_query)[:, 1]
    clf.remove_models_from_memory()

    results = pd.DataFrame({"smiles": smiles_query, "proba": y_hat})

    def get_molecule_image(smiles):
        m = Chem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(m)
        opts = Draw.DrawingOptions()
        opts.bgColor = None
        im = Draw.MolToImage(m, size=(200, 200), options=opts)
        return im

    cols = st.columns(4)
    c_i = 0
    for v in results.iterrows():
        idx = v[0]
        if c_i > 3:
            c_i = 0
            cols = st.columns(4)
        r = v[1]
        image = get_molecule_image(r["smiles"])
        cols[c_i].image(image)
        texts = ["{0}: {1}".format(idx + 1, r["smiles"])]
        texts += ["Proba: {0:.3f}".format(r["proba"])]
        cols[c_i].text("\n".join(texts))
        c_i += 1

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df(results)

    st.download_button(
        "Download as CSV",
        csv,
        "predictions.csv",
        "text/csv",
        key="download-csv",
    )
