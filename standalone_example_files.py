"""
Simple script to generate test files for standalone testing
"""

from rdkit import Chem
import pickle

ibuprofen_smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
acetaminophen_smiles = "CC(=O)Nc1ccc(O)cc1"
asprin_smiles = "O=C(C)Oc1ccccc1C(=O)O"

smiles = [ibuprofen_smiles, acetaminophen_smiles , asprin_smiles]

mols = [Chem.MolFromSmiles(s) for s in smiles]
mols = [Chem.AddHs(m) for m in mols]
[Chem.SanitizeMol(m) for m in mols]

TOTAL_N = 10
out_mols = []
for i in range(TOTAL_N // len(mols)):
    for m in mols:
        out_mols.append(m)
        
FILENAME = "example"

pickle.dump([m.ToBinary() for m in out_mols], 
            open(f'{FILENAME}.rdkit', 'wb'))