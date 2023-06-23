from rdkit import Chem
import numpy as np
from datetime import datetime
import click
import pickle
import time
import json 
import sys
import warnings
from fullsspruce import predictor
import os

warnings.filterwarnings("ignore")


@click.command()
@click.argument('infile', default=None)
@click.argument('outfile',  default=None)
@click.option('--13C', 'pred_13C', is_flag=True, default=False,
              help='Predict 13C shifts')
@click.option('--1H', 'pred_1H', is_flag=True, default=False,
              help='Predict 1H shifts')
@click.option('--coupling','pred_coupling',  is_flag=True, default=False,
              help='Predict coupling')
@click.option('--num_data_workers', default=0, type=click.INT)
@click.option('--cuda/--no-cuda', default=True)
@click.option('--prog-bar/--no-prog-bar', default=True)
@click.option("--version", default=False, is_flag=True)
@click.option("--use_confs", default=False, is_flag=True)
@click.option("--save_confs", default='', type=click.STRING)
@click.option('--sanitize/--no-sanitize', help="sanitize the input molecules",
              default=True)
@click.option('--addhs', help="Add Hs to the input molecules", default=False)
def predict(infile, outfile, 
            pred_1H, pred_13C, pred_coupling,
            cuda=False, sanitize=True, addhs=True,
            print_data = None, version=False,
            num_data_workers=0, prog_bar=True,
            use_confs=False, save_confs=''):

    ts_start = time.time()

    mol_supplier = [Chem.Mol(m) for m in pickle.load(open(infile, 'rb'))]

    write_out_confs = True
    if save_confs == '':
        write_out_confs = False

    geometry = {
        'use_confs': use_confs,
        'save_confs': write_out_confs
    }

    p = predictor.Predictor(geometry = geometry,
                            use_gpu = cuda,
                            prog_bar = prog_bar,
                            num_workers = num_data_workers)


    raw_mols = []
    for m in mol_supplier:
        raw_mols.append(m)


    mols = [Chem.Mol(m) for m in raw_mols] 
        
    if sanitize:
        [Chem.SanitizeMol(m) for m in mols]
    

    tgt_properties = []
    if pred_1H: tgt_properties.append('1H')
    if pred_13C: tgt_properties.append('13C')
    if pred_coupling: tgt_properties.append('coupling')
    
    predictions, meta = p.predict(mols, properties=tgt_properties)

    ts_end = time.time()

    if write_out_confs:
        pickle.dump(meta['mols_with_confs'], open(save_confs, 'wb'))
        del meta['mols_with_confs']

    output_dict = {'predictions' : predictions,
                   'meta' : meta, 
                   
                   'cmd_meta' : {
                       'ts_start' : datetime.fromtimestamp(ts_start).isoformat(), 
                       'ts_end': datetime.fromtimestamp(ts_end).isoformat(), 
                       'runtime_sec' : ts_end - ts_start,
                       'git_commit' : os.environ.get("GIT_COMMIT", ""),
                       'rate_mol_sec' : len(predictions) / (ts_end - ts_start),
                       }
              }
    json_str = json.dumps(output_dict, sort_keys=False, indent=4)
    with open(outfile, 'w') as fid:
        fid.write(json_str)

if __name__ == "__main__":
    predict()
