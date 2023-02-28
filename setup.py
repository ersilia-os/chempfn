import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ensemble-tabpfn",
    author="Ersilia",
    author_email="hello@ersilia.io",
    description="Ensemble TabPFN",
    keywords="automl, tabpfn, tabular",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Documentation": "",
        "Source Code": "https://github.com/ersilia-os/ensemble-tabpfn",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires=">=3.8",
    install_requires=[""],
    extras_require={
        # 'dev': ['check-manifest'],
        # 'test': ['coverage'],
    },
)
