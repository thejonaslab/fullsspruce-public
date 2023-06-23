import datetime as dt
import os
import yaml
import pandas as pd
from rdkit import Chem
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from fullsspruce import util
from fullsspruce import netutil
from fullsspruce.forward_train import train

def _train(exp_config_name):
    checkpoint_dir = '/tmp/checkpoints'
    exp_extra_name = f'unit_test.{str(dt.datetime.now().timestamp())}'

    exp_config = yaml.load(open(exp_config_name, 'r'), Loader=yaml.FullLoader)
    exp_name = os.path.basename(exp_config_name.replace(".yaml", ""))

    featurize_config_update = exp_config['featurize_config']
    featurize_config = netutil.DEFAULT_FEATURIZE_CONFIG
    util.recursive_update(featurize_config, featurize_config_update)

def test_train_inference_shifts():
    assert True

def test_train_inference_coupling():
    assert True