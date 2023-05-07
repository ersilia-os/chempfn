:rocket: **TabPFN** is a transformer architecture proposed by [Hollman et al.](https://arxiv.org/abs/2207.01848) for classifying **small tabular datasets**. The model has been trained once and does not require training again on novel datasets. It works by approximating the distribution of new data to the prior synthetic data seen during training.

:zap: TabPFN guarantees **blazing fast** run times by "fitting" on a training dataset in under a second and generating predictions for the query set in a single forward pass in the network. [Ensemble-TabPFN](https://github.com/ersilia-os/ensemble-tabpfn) is an extension of TabPFN to allow working with datasets of any scale by performing strategic data and feature subsampling.

:pill: This application is an adaptation of Ensemble-TabPFN to work with **small molecule inputs** in an attempt to provide the cheminformatics community with an out-of-box solution to generate molecular property predictions.
