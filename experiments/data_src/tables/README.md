These are the links to download the datasets:

- [Public BI Benchmark](https://storage.googleapis.com/pneuma_open/pneuma_public_bi.tar)
- [Chicago Open Data](https://storage.googleapis.com/pneuma_open/pneuma_chicago_10K.tar)
- [Chembl](https://storage.googleapis.com/pneuma_open/pneuma_chembl_10K.tar)
- [FetaQA](https://storage.googleapis.com/pneuma_open/pneuma_fetaqa.tar)
- [Adventure Works](https://storage.googleapis.com/pneuma_open/pneuma_adventure_works.tar)

Download a dataset using this command:

```bash
wget [url of the dataset]
```

Extract the tables in the dataset by running this command:

```bash
tar -xvf [name of the tar file]
```

Then, pre-process the names of the tables by using the `preprocess.ipynb` code.
