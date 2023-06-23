"""
Tests of the stand-alone prediction runner, whihc is the most end-user
interface part of the entire prediction API. 

"""

import pytest
import pickle
from rdkit import Chem

import subprocess
import subprocess
import json



@pytest.fixture(scope="session")
def mols_file(tmp_path_factory):


    mols = [Chem.AddHs(Chem.MolFromSmiles('CCC')),
            Chem.AddHs(Chem.MolFromSmiles('OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O')), # glucose
            ]
    
    rdkit_filename = tmp_path_factory.mktemp("mols") / "mols.rdkit"
    print('rdkit_filename=', rdkit_filename)
    with open(rdkit_filename, 'wb') as fid:
            pickle.dump(mols, fid)
    
    return rdkit_filename

def test_predictor_basic(mols_file, tmp_path):
    """
    basic "does it run" cmd
    """
    output_json = tmp_path / "output.json"


    cmd = f"fullsspruce --no-sanitize --1H --no-cuda {mols_file} {output_json}"

    subprocess.check_output(cmd, shell=True)

    
    output_json = json.load(open(output_json, 'r'))

    assert 'meta' in output_json
    assert 'cmd_meta' in output_json
    assert 'predictions' in output_json
    assert len(output_json['predictions']) == 2

            
@pytest.fixture(scope="session")
def fullsspruce_errors(tmp_path_factory):
    


    mols = [Chem.AddHs(Chem.MolFromSmiles('CCC')),
            Chem.AddHs(Chem.MolFromSmiles('OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O')), # glucose,
            # allow up to a64
            Chem.AddHs(Chem.MolFromSmiles('CC1CCC2C(C)C(O)CC3C(C12)C3(C)C')),
            # assume "too big": 
            Chem.AddHs(Chem.MolFromSmiles('C'*100)),
            # Uranium hexaflloride, we're never going to support uranium:
            Chem.AddHs(Chem.MolFromSmiles('F[U](F)(F)(F)(F)F')), 
            Chem.AddHs(Chem.MolFromSmiles('C'))
            ]
    errors = [False,
                False,
                False,
                True,
                True,
                False
                ]

    rdkit_filename = tmp_path_factory.mktemp("mols") / "mols_with_errors.rdkit"
    print('rdkit_filename=', rdkit_filename)
    with open(rdkit_filename, 'wb') as fid:
            pickle.dump(mols, fid)
    
    return rdkit_filename, errors

def test_predictor_errors(fullsspruce_errors, tmp_path):
    """
    basic "does it run" cmd
    """
    output_json = tmp_path / "output.json"

    mols_file_errors, errors = fullsspruce_errors

    cmd = f"fullsspruce --no-sanitize --1H --no-cuda {mols_file_errors} {output_json}"

    subprocess.check_output(cmd, shell=True)
    
    output_json = json.load(open(output_json, 'r'))

    assert 'meta' in output_json
    assert 'cmd_meta' in output_json
    assert 'predictions' in output_json
    assert len(output_json['predictions']) == len(errors)
    for i, v in enumerate(output_json['predictions']):
        if not errors[i]:
            print(i, "Not e")
            assert not v is None
        else:
            print(i, "E")
            assert v is None


    
