from rdkit import Chem
from models.sascorer import calculateScore



# compute SA score
def SAscore(smile):
    mol = Chem.MolFromSmiles(smile)
    return calculateScore(mol)