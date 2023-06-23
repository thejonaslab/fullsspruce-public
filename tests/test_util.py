import numpy as np
import pandas as pd
import pytest

from rdkit import Chem
from fullsspruce import util

def test_methyl_average_coupling():
    """
    VERY simple test
    """
    m = Chem.AddHs(Chem.MolFromSmiles("CC"))
    # will be C1 C2 H1 H1 H1 H2 H2 H2

    coupling_dict = {(0, 1) : 0.3, 
                     (0, 2) : 1.0, 
                     (0, 3) : 2.0, 
                     (0, 4) : 3.5,
                     (3, 4) : 0.2,
                     (3, 5) : 0.4,
                     (4, 5) : 0.6}

    avg_dict = util.methyl_average_coupling(m, coupling_dict)

    assert avg_dict[(0, 1)] == 0.3
    np.testing.assert_allclose(avg_dict[(0, 2)], 2.1666666666)
    np.testing.assert_allclose(avg_dict[(0, 3)], 2.1666666666)
    np.testing.assert_allclose(avg_dict[(0, 4)], 2.1666666666)
    
    for i in range(m.GetNumAtoms()):
        assert (i, i) not in avg_dict
