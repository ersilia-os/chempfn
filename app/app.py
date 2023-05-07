import os
import streamlit as st
from eosce.models import ErsiliaCompoundEmbeddings
from lol import LOL
from tabpfn import TabPFNClassifier
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
example_file = os.path.join(ROOT, "example.csv")
dex = pd.read_csv(example_file)

TEXT_AREA_HEIGHT = os.environ.get("text_height", 400)

st.set_page_config(
    layout="wide",
)

st.title("Fast chemical classification with EnsembleTabPFN")

st.sidebar.title("Welcome to Ersilia")
st.sidebar.header("Learn more")

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

cols[0].text("\n".join(dex[dex["act"] == 1]["smiles"]))
cols[1].text("\n".join(dex[dex["act"] == -1]["smiles"]))
cols[2].text("\n".join(dex[dex["act"] == 0]["smiles"]))

st.write(mols_act)
smiles_train = mols_act + mols_inact
y_train = [1] * len(mols_act) + [0] * len(mols_inact)
smiles_query = mols_query

embedder = ErsiliaCompoundEmbeddings()

X_train = embedder.transform(smiles_train)
X_query = embedder.transform(smiles_query)

reducer = LOL(n_components=100)
reducer.fit(X_train, y_train)
X_train = reducer.transform(X_train)
X_query = reducer.transform(X_query)

clf = TabPFNClassifier()
clf.fit(X_train, y_train)
y_hat = clf.predict_proba(X_query)[:, 1]

results = pd.DataFrame({"smiles": smiles_query, "proba": y_hat})

st.dataframe(results)
