import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ensemble-tabpfn",
    author="Dhanshree Arora",
    author_email="dhanshree.arora@gmail.com",
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
        # see https://pypi.org/classifiers/
        # 'Development Status :: 5 - Production/Stable',
        # 'Intended Audience :: Developers',
        # 'Topic :: Software Development :: Build Tools',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: 3.7',
        # 'Programming Language :: Python :: 3.8',
        # 'Programming Language :: Python :: 3.9',
        # 'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires=">=3.6",
    install_requires=[""],
    extras_require={
        # 'dev': ['check-manifest'],
        # 'test': ['coverage'],
    },
)
