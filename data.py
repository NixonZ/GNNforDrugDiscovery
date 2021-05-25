from graph import *
import pandas as pd 

def read_graphs_from_dataset(dir):
    '''
    Reads dataset which is stored locally in directory @param: dir
    '''
    data = pd.read_csv(dir)
    mols = []
    for smiles in data["SMILES"]:
        G = read_molecule_nx(smiles)
        mol = read_molecule_mol(smiles)
        mols.append(mol)