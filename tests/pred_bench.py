from rdkit import Chem
import pickle

MAX_ATOM_N = 32

def gen_mols():

    out_mols = []
    for i in range(1000):
        smiles = 'C'* ((i % 14) + 5)
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

        if mol.GetNumAtoms() <= MAX_ATOM_N:
            out_mols.append(mol)

    print(len(out_mols))

    pickle.dump(out_mols, open("mols.benchmark.pickle", 'wb'))

if __name__ == "__main__":
    gen_mols()
