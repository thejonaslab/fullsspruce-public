# FullSSPrUCe Tutorial

The FullSSPrUCe codebase is centered around training and evaluating models for predicting the chemical shifts of atoms in NMR experiments. Here, we will lay out the basics of how to train and evaluate a model, focusing on where simple adjustments can be made during experimentation. 

Training and evaluation have been separated, so that a model can be trained in a different manner than it is evaluated (such as using different data). However, both require a dataset which is a pickled list of molecules and their atoms' observed shift values. This processed data is then used to train or evaluate a model as follows.

## Training a Model

Training a model is done by a call to `forward_train.py`, which may look something like: 

```
CUDA_VISIBLE_DEVICES=1 python forward_train.py expconfig/carbon_GNN.yaml myexpname
```

The main way that we make adjustments to the training is through the yaml file which is passed in. This file specifies most of the details for the training run, starting with what datasets to use for training and testing, as well as how to split those up. Our default models are specified in the `expconfig` directory, with a long list of additional experiments in `expconfig_additional_experiments`. The second input is an experiment name that will be used to name output files. 

### Preprocessing Data

The raw data should be passed in to the trainer through the yaml file. This raw data should be in the format of a pandas DataFrame that has been pickled. The DataFrame should have the following columns to be understood properly: `molecule_id`, `rdmol`, `spect_dict`, and `morgan4_crc32`. The molecule id is simply an identifier. The rdmol is an RDKit molecule object. The spectrum dictionary is a dictionary mapping the atoms for which spectra are available to their shift values. The morgan entry is the morgan fingerprint of the molecule. We also include the smile strings and a spectrum id in our entries, but these are not necessary in the current training pipeline, and serve only for easier analysis. 

From the raw data we need to create a training dataset and a testing dataset. We have functions that do this in `netutil.py`, specifically `CVSplit` and `make_dataset`. In `make_dataset`, we create a dataset object, which is a collection of feature vectors and their labels. This requires us to featurize the molecules, which we can do in various ways. The current featurization is `feat_tensor_mol`, which provides features for each atom in the molecule such as if it is in a ring or not (see: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0374-3/tables/1). We have one of these datasets for each phase we specified in the yaml file, such as testing or training. The yaml file specification of the dataset and how to split it might look like this:

```
exp_data:
  data :
    - filename: 'processed_data/nmrshiftdb_128_128_HCONFSPCl_wcharge_13C.shifts.dataset.pickle'
      phase: train
      
    - filename: 'processed_data/nmrshiftdb_128_128_HCONFSPCl_wcharge_13C.shifts.dataset.pickle'
      phase: test

  extra_data : []
  filter_max_n: 0
  cv_split :
    how: morgan_fingerprint_mod
    mod : 10
    test : [0, 1]
```

It's important to note here how and why we split our data into training and testing. One issue that can come up with the datasets we have looked at is replication of data. When molecules show up multiple times, putting the copies into different phases (e.g. one in test, one in train) can be problematic. To avoid this, we use a hash of the Morgan fingerprint of each molecule as an identifier, then split into training and testing based on those identifiers. This ensures that any duplicates will be sorted into the same phase. This specification is made in the yaml file under `cv_split` and implemented in `netutil.py`.

The yaml file is also where we specify which neural network model to use and what hyperparameters to use on it. Our canonical models with their hyperparameters are specified under the `expconfig` directory, such as in `carbon_GNN.yaml`, and the details of the model can be seen in `nets.py` under `GraphVertConfigBootstrapWithMultiMax`. The loss function and optimizer are also specified here. 

### Checkpoints, Validation and Metadata

During training, we output information about the process. Before running any epochs, we output the metadata about the run to a file, which is unchanging throughout. Then, every certain number of epochs, we write out checkpoints and validation information. The checkpoints are written to a file like the metadata. Combining the metadata and a checkpoint gives the full information about the model at a given epoch. The validation information is written to tensorboard, which allows for real-time monitoring of performance. 

When checkpoints and metadata are written to files, they are put into the checkpoints folder, which must exist at runtime. The experiment name is the prefix of each file. The metadata goes in `expname.meta`, while checkpoint N saves both `expname.N.state` and `expname.N.model`. The state file contains a dictionary for reloading a model, while the model file contains the model itself. Both of these methods have their advantages (as discussed here: https://pytorch.org/tutorials/beginner/saving_loading_models.html), so we save both. We also output a copy of the yaml file used to initialize the run to `expname.yaml` for reproducability, and a json file containing extended information about the initialization of the run, including the featurization used, to `expname.json`. Lastly, we also output a pickled version of the pre-processed data for each phase (e.g. training, etc.) to the files `expname.data.phase_number.phase_number.data`. 

### Running Epochs

In `netutil.py`, there are two functions of interest to us. The first is `generic_runner`, which is called once to train the model. It loops through the epochs, handling the learning rate scheduler, the checkpointing, and the validation. Each epoch, it calls on `run_epoch`, which handles all of the details of training the model, including evaluating the model on the current batch of data, calculating gradients, and updating weights (if desired). `run_epoch` is also used to validate the model, by setting it to predictions only. 

## Evaluating a Model

The above steps allow for the training of a model, producing a series of output files that include checkpoints and metadata for evaluation of the model. While tensorboard does allow for some validation during training, post-hoc evaluation can be extremely valuable in identifying the causes of training issues. `forward_evaluate_pipeline.py` is an evaluation script that produces a text file with summary statistics about the model's performance during training. 

### Specifying the Experiment

In the [evaluation script](forward_evaluate_pipeline.py), we specify some experiments to evaluate. In these lines, we create a dictionary of experiments. Each experiment maps to a dictionary of parameters. The `model` parameter specifies the location of the checkpoints and experiment name of the experiment to be evaluated. If all of our checkpoints (and metadata, etc.) from the training run are still in checkpoints, this will look like `checkpoints/expname`. We also specify which checkpoints we would like to evaluate in the `checkpoints` list. The leading zeros in the file names are not included here. Then we specify the dataset to use and how to split it up in `dataset` and `cv_sets`. `cv_sets` is formatted similarly to the same argument in the yaml file. Lastly, we include the batch size and type of shifts being measured in `batch_size` and `nuc`. 

In most cases, much of this information, especially the dataset will be the same as at training time, and therefore could be accessible without specification here. However, we may want to evaluate in different ways than we trained. Particularly, we may want to evaluate on a different dataset than we trained on. The checkpoints list is also very important, because it allows us to see how well we are improving over time, and increase or decrease the granularity of our evaluation. 

### Outputs

The evaluation script takes in the information from these experiements dictionaries and finds the proper files. It follows the same processes to create the datasets, then creates a model from the metadata and checkpoints. It then gets predictions from the model, which indirectly calls to `run_epoch` again. These predictions are used to run the desired evaluation functions.

At each step, we output some of this information into the directory `forward.preds`. Again we prefix each file with the name of the experiment as specified in the EXPERIMENTS dictinonary. First, we create a file containing a DataFrame with all of the data used to `expname.feather`. Then, for each of the desired epochs, we create a file containing a DataFrame of each data point and that epoch's model's prediction. These files look like `expname.feather.epoch_number`. At the end, we run some evaluation functions, which are currently `compute_summary_stats`, `summary_textfile` and `per_confidence_stats`. This first computes various summary statistics, then writes certain statistics to files such as `expname.summary.txt` and `expname.per_conf_stats.txt`. Pickle and feather versions are also sometimes included for ease of analysis. 

As an example, we show the most highly trained model on carbon 13 shifts. Below we show the output written to the file `expname.per_conf_stats.txt`, which splits each epoch's data by how confident we were in our predictions and displays the mean absolute error of our predictions for the highest confidence data. Only one epoch is shown for this model.

```
                                mae                                                  
frac_data                      0.10      0.20      0.50      0.90      0.95      1.00
stats_group   phase epoch                                                            
('13C', None) test  450    0.415395  0.511040  0.753887  1.066205  1.126635  1.212573
              train 450    0.302139  0.345696  0.451862  0.518482  0.530290  0.555544
```

To replicate this run, create an experiment in `forward_evaluate_pipeline.py` as follows:

```
EXPERIMENTS['carbon_GNN'] = { 'model': 'checkpoints/fs_def_13C_5_27.carbon_GNN.537094139799',
                              'checkpoints' : [450],
                              'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (0, 1)}], 
                              'pred_fields' : ['pred_shift_mu', 'pred_shift_std'],
                              'batch_size' : 32, 
                              'nuc' : '13C',
                              'dataset' : "processed_dbs/shifts.nmrshiftdb.128_128_HCONFSPCl_wcharge_13C.dataset.pickle"
}
```

This will evaluate our best model for carbon 13 shifts. Our best models are stored in the models directory as the default models. The other files in the directory are symlinks to the defaults to fit the format expected in the evaluation pipeline.

## Modifictions

There are four main places that this pipeline can be easily modified to be adapted to new problems:

### The YAML

At training, we take in a yaml file that specifies the majority of information about the training run. The most common way this file could be modified is to update the dataset to use. Hyperparameter tuning of the neural net can also take place in the yaml.

### Featurization

New featurizations of molecules are another way that improvements can be made to the predictive ability of the model. Introducing a new featurization can be done by creating a new featurization function and replacing the call to the current one with the new one. The current featurization function, `feat_tensor_mol`, lives in `molecule_features.py` and is called by `__getitem__` in `netdataio.py`. 

### A new NN architecture

The current neural network architecture is at `nets.GraphVertConfigBootstrapWithMultiMax`. If a new NN architecture is desired, it should inherit the nn.Module class from pytorch (`torch.nn.Module`). The yaml file should also be updated accordingly.

### Evaluation

Updating the way that the model is evaluated can also be very useful, mainly for understanding which of the above changes will be most beneficial. `forward_evaluate_pipeline` should of course be updated with the correct information on the run to be evaluated. New methods of evaluation can also be created. The simplest way to do so is to modify an existing function, such as `per_confidence_stats`, to output the desired statistics, such as the minimum and maximum error during an epoch, rather than just the mean. New functions entirely can also be created, and then added to the pipeline_run call at the end of the file. 

The file `dp4_analysis.py` provides another way to evaluate the success of a model. Rather than comparing the predictions to a ground truth, we determine whether the predictions can be used to identify the correct stereoisomer of a given molecule. The setup is very similar, with an example provided for the GNN only model. 