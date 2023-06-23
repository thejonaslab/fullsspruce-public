import numpy as np
import rdkit
import pytest
import pickle
from rdkit import Chem
from rdkit.Chem.rdmolops import AddHs
from rdkit.Chem import rdqueries

from fullsspruce.predictor import Predictor
from fullsspruce.geom_util import geometryFeaturizerGenerator

MOL_SMILES = [
    ('[H]c1c([H])c([H])c([C@@]2([H])C([H])([H])[C@]3([H])[C@]4([H])N(C(=O)OC4([H])[H])C([H])([H])[C@]23[H])c([H])c1[H]', True),
    ('[H]c1c([H])c([H])c2c(c1[H])c(=O)c1c([H])c([H])c3c(c([H])c([H])c(=O)n3[H])c1n2[H]', True),
    ('F[U](F)(F)(F)(F)F', False),
    ('CC1CCC2C(C)C(O)CC3C(C12)C3(C)C', True),
    ('C'*100, False),
    ('[H]C([H])([H])C1=C(C([H])([H])[H])C2=C(C([H])([H])C([H])([H])[C@@](C([H])([H])[H])(C([H])([H])C([H])([H])C([H])([H])[C@]([H])(C([H])([H])[H])C([H])([H])C([H])([H])C([H])([H])[C@]([H])(C([H])([H])[H])C([H])([H])C([H])([H])C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H])O2)[C@]2(Oc3c(C([H])([H])[H])c(C([H])([H])[H])c4c(c3C2([H])[H])C([H])([H])C([H])([H])[C@@](C([H])([H])[H])(C([H])([H])C([H])([H])C([H])([H])[C@]([H])(C([H])([H])[H])C([H])([H])C([H])([H])C([H])([H])[C@]([H])(C([H])([H])[H])C([H])([H])C([H])([H])C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H])O4)C1=O', False),
    ('[H]/C(=C(/C(=O)OC([H])([H])[H])N([H])C(=O)C([H])([H])[H])c1sc(C([H])([H])[H])c([H])c1[H]', True),
    ('[H]C([H])([H])[C@]12O[C@](C([H])([H])[H])(C([H])([H])C1([H])[H])[C@@]1(C([H])([H])C([H])([H])[C@](Cl)(C([H])([H])[H])[C@]([H])(Br)C1([H])[H])[C@@]2([H])C([H])([H])[H]', False)
]

def test_predict_ETKDG():
    # Set up Predictor using ETKDG geoms
    p = Predictor() 
    
    # Predict a list of mols from their smiles strings
    mols_list = [AddHs(Chem.MolFromSmiles(m)) for m, _ in MOL_SMILES]
    valid_list = [v for _, v in MOL_SMILES]
    preds_list, meta_list = p.predict(mols_list, properties=['1H', '13C', 'coupling'])

    # Verify that expected predictions are all returned
    for i, mol in enumerate(mols_list):
        if not valid_list[i]:
            assert preds_list[i] == None
            continue
        N = mol.GetNumAtoms()
        q_H = rdqueries.AtomNumEqualsQueryAtom(1)
        num_H = len(mol.GetAtomsMatchingQuery(q_H))
        q_C = rdqueries.AtomNumEqualsQueryAtom(6)
        num_C = len(mol.GetAtomsMatchingQuery(q_C))
        assert len(preds_list[i]['1H']) == num_H
        assert len(preds_list[i]['13C']) == num_C
        assert len(preds_list[i]['coupling']) >= num_H #== ((N-1)*N)/2

    # Predict the mols one at a time
    preds_ind = []
    for i, m in enumerate(mols_list):
        try:
            pred, err = p.predict(m, properties=['1H', '13C', 'coupling'])
        except:
            print(err)
            assert not valid_list[i]
            preds_ind += [None]
        else:
            preds_ind += [pred]
    
    # Verify that predictions are the same
    assert recursive_equal(preds_list, preds_ind)


def test_predict_ETKDG_prop_subset():
    """
    Can we predict subsets of NMR properties? 
    """

    properties = ['1H', '13C', 'coupling']

    for tgt_props in [['1H'], ['13C'], ['coupling'],
                  ['1H', '13C'], ['1H', 'coupling'],
                  ['13C', 'coupling']]:
        
        # Set up Predictor using ETKDG geoms
        p = Predictor() 

        # Predict a list of mols from their smiles strings
        mols_list = [AddHs(Chem.MolFromSmiles(m)) for m, _ in MOL_SMILES]
        valid_list = [v for _, v in MOL_SMILES]
        preds_list, meta_list = p.predict(mols_list, properties=tgt_props)

        # Verify that expected predictions are all returned
        for i, mol in enumerate(mols_list):
            if not valid_list[i]:
                assert preds_list[i] == None
                continue
            N = mol.GetNumAtoms()
            q_H = rdqueries.AtomNumEqualsQueryAtom(1)
            num_H = len(mol.GetAtomsMatchingQuery(q_H))
            q_C = rdqueries.AtomNumEqualsQueryAtom(6)
            num_C = len(mol.GetAtomsMatchingQuery(q_C))
            if '1H' in tgt_props:
                assert len(preds_list[i]['1H']) == num_H
            if '13C' in tgt_props:
                assert len(preds_list[i]['13C']) == num_C
            if 'coupling' in tgt_props:
                assert len(preds_list[i]['coupling']) >= num_H #== ((N-1)*N)/2

        # Predict the mols one at a time
        preds_ind = []
        for i, m in enumerate(mols_list):
            try:
                pred, _ = p.predict(m, properties=tgt_props)
            except:
                assert not valid_list[i]
                preds_ind += [None]
            else:
                preds_ind += [pred]

        # Verify that predictions are the same
        assert recursive_equal(preds_list, preds_ind)

def test_predict_no_valid():
    # Set up Predictor using ETKDG geoms
    p = Predictor() 
    
    # Predict a list of mols from their smiles strings
    mols_list = [AddHs(Chem.MolFromSmiles(m)) for m, v in MOL_SMILES if not v]
    with pytest.raises(ValueError):
        preds_list, meta_list = p.predict(mols_list, properties=['1H', '13C', 'coupling'])

def test_precalc_valid():
    # Test Predictor's precomputation validity function
    p = Predictor()

    for s, b in MOL_SMILES:
        m = AddHs(Chem.MolFromSmiles(s))
        valid, reason = p.is_valid_mol(m)
        assert valid == b

def test_accuracy():
    p = Predictor()

    test_data = pickle.load(open('testing_mols.pickle', 'rb'))
    mols_list = [m for m, _, _, _, _ in test_data]

    preds_list, meta_list = p.predict(mols_list, properties=['1H', '13C', 'coupling'])
    p_errs, c_errs, coup_errs = [], [], []

    for  (m, sm, sp_p, sp_c, cd), preds in zip(test_data, preds_list):
        for a_idx, value in sp_p.items():
            p_errs += [abs(value - get_shift_pred(a_idx, preds['1H']))]
        for a_idx, value in sp_c.items():
            c_errs += [abs(value - get_shift_pred(a_idx, preds['13C']))]
        for (a1, a2), value in cd.items():
            coup_errs += [abs(value - get_coup_pred(a1, a2, preds['coupling']))]

    assert np.mean(p_errs) <= 0.25
    assert np.mean(c_errs) <= 1.30
    assert np.mean(coup_errs) <= 1.0

def get_shift_pred(a_idx, preds):
    for p in preds:
        if a_idx == p['atom_idx']:
            return p['pred_mu']
    raise IndexError("No match for atom index.", a_idx)

def get_coup_pred(a1, a2, preds):
    for p in preds:
        if a1 == p['atom_idx1'] and a2 == p['atom_idx2']:
            return p['coup_mu']
    raise IndexError("No match for atom index.", a1, a2)

def recursive_equal(a, b):
    """
    Check if a and b are equal by recursively traversing each. All elements
    in a and b, including a and b themselves, must be dictionaries, lists, 
    or equal to return True.
    """
    if not type(a) is type(b):
        print("Type failure")
        return False
    else:
        if a is None or b is None:
            return a == b
        elif isinstance(a, dict):
            for key, val in a.items():
                if not recursive_equal(val, b[key]):
                    print("dictionary", key)
                    return False
        elif isinstance(a, list):
            for i, val in enumerate(a):
                if not recursive_equal(val, b[i]):
                    print("list")
                    return False
        else:
            if not np.allclose(a, b, atol=1e-04):
                print("Not close", a, b)
                return False
    return True
        


