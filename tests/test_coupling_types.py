
import numpy as np

from rdkit import Chem
from fullsspruce.featurize import netdataio
# import sys

def test_coupling_correct():
    """
    Simple sanity check for coupling encoder
    """
    MAX_N = 10
    mol = Chem.AddHs(Chem.MolFromSmiles('CC'))
    
    coupling_types_lut = [('CH', 1),
                          ('HH', 2)]

    ct = netdataio.coupling_types(mol, MAX_N, coupling_types_lut)

    ## assume mols are [C, C, H, H, H, H, H, H]
    first_c = np.array([-2, -1, 0, 0, 0, -1, -1, -1, -2, -2])
    
    np.testing.assert_allclose(ct[0], first_c)
    np.testing.assert_allclose(ct[:, 0], first_c)

    first_h = np.array([0, -1, -2, 1, 1, -1, -1, -1, -2, -2])
    
    np.testing.assert_allclose(ct[2], first_h)
    np.testing.assert_allclose(ct[:, 2], first_h)
    

