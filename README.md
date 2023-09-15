# FullSSPrUCe

This is the main public repository supporting the paper [Rapid Prediction of Full Spin Systems using Uncertainty-Aware Machine Learning](https://doi.org/10.1039/D3SC01930F). In the paper, we describe a new machine learning model for Full Spin System Prediction with UnCertainty (FullSSPrUCe). In this repository, we provide the code which trains and evaluates these models, as well as ways to install and run the models locally. The [paper website](https://spectroscopy.ai/papers/fullsspruce/) also provides an API to run the models via the web.    

To modify existing models or train your own, see [the tutorial](fullsspruce/) for a more detailed explanation of how to get up and running with training a model. 

## Setting up the anaconda environment

We only support the [Anaconda python distribution](https://www.anaconda.com/distribution/), given the complexities of the machine learning frameworks that we depend on. You should install the necessary packages using conda. 

```
conda env create --name respredict -f environment.yml
```

Then to activate this environment run
```
conda activate respredict
```

### Installing FullSSPrUCe

After setting up the anaconda environment, install the FullSSPrUCe package to get access to our models and functions.

```
python setup.py install
```

## Using standalone mode

If you have molecules whose shifts you would like to predict you can
use the standalone runner. This runner accepts a pickled list of rdkit molecules. 

You can create an example file by running 
```
python standalone_example_files.py
``` 
which will create an example RDKit file `example.rdkit`. 

To test with the example molecules run:

```
fullsspruce --no-sanitize --1H --no-cuda example.rdkit output.json
```

which should generate something like the following to `output.json` (numbers may vary slightly due to both floating point issues and possibly updated model files):

```
{
    "predictions": [
        {
            "1H": [
                {
                    "atom_idx": 15,
                    "pred_mu": 0.8638967275619507,
                    "pred_std": 0.06089625135064125
                },
                {
                    "atom_idx": 16,
                    "pred_mu": 0.8638967275619507,
                    "pred_std": 0.06089625135064125
                },
[...clipped...]
```                       

## Using Predictor

Along with the commandline version of fullsspruce, we also have a Predictor which can be imported into other Python code to easily make predictions in your own pipelines. After installing the FullSSPrUCe package, the Predictor can be accessed and used as follows:

```
from fullsspruce.predictor import Predictor

p = Predictor() 
preds_list, meta_list = p.predict(mols_list, properties=['1H', '13C', 'coupling'])
```

`mols_list` should be a list of rdkit Mols, and the returned objects will be lists of dictionaries specifying the predicted values for each Mol in the order input, as well as meta information about time taken to make predictions. Here, we used the default predictor for protons, carbons, and coupling, but other options can be specified, including using your own models.

## Other files / pipelines

`process_nmrshiftdb_dataset.py` : Process the nmrshiftdb data into our dataset files.

`distance_features_pipeline.py` : Pre-generate distance features to save time during training or evaluation. 

