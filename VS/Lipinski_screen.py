# -*- coding: utf-8 -*-
"""
SMILES Molecular Filter Pipeline
Function:
1. Remove empty SMILES data
2. Verify chemical validity of SMILES via RDKit
3. Filter molecules compliant with Lipinski's Rule of Five (max 1 violation)
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Lipinski, Descriptors

# ====================== Configurations ======================
# Relative file paths (adapt to your working directory)
INPUT_CSV = "processed_smiles.csv"
OUTPUT_CSV = "Lipinski_filtered_smiles.csv"
# ============================================================

def validate_smiles(smiles_list: list) -> list:
    """
    Filter chemically valid SMILES strings using RDKit
    Args:
        smiles_list: List of raw SMILES strings
    Returns:
        List of chemically valid SMILES
    """
    valid_smiles = []
    for smi in smiles_list:
 # Skip non-string and empty values
        if not isinstance(smi, str) or not smi.strip():
            continue
        try:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is not None:
                valid_smiles.append(smi.strip())
        except Exception as e:
            print(f"[Warning] Failed to parse SMILES: {smi} | Error: {str(e)}")
    return valid_smiles


def filter_lipinski_rule(smiles_list: list) -> list:
    """
    Filter molecules by Lipinski's Rule of Five
    Allow maximum 1 rule violation for drug-likeness
    Args:
        smiles_list: List of valid SMILES strings
    Returns:
        List of SMILES compliant with Lipinski rules
    """
    filtered_smiles = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue

        # Calculate molecular descriptors
        mol_weight = Descriptors.MolWt(mol)
        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)
        log_p = Descriptors.MolLogP(mol)

        # Count rule violations
        violations = 0
        if mol_weight > 500:
            violations += 1
        if h_donors > 5:
            violations += 1
        if h_acceptors > 10:
            violations += 1
        if log_p > 5:
            violations += 1

        # Keep molecules with at most 1 violation
        if violations <= 1:
            filtered_smiles.append(smile)
    return filtered_smiles


def main():
    # Load raw data and drop empty SMILES
    df = pd.read_csv(INPUT_CSV, dtype={"SMILES": str})
    raw_smiles = df["SMILES"].dropna().tolist()
    print(f"Total raw SMILES data: {len(raw_smiles)}")

    # Step 1: Filter chemically valid SMILES
    valid_smiles = validate_smiles(raw_smiles)
    print(f"Chemically valid SMILES after filtering: {len(valid_smiles)}")

    # Step 2: Filter by Lipinski's Rule of Five
    lipinski_smiles = filter_lipinski_rule(valid_smiles)
    print(f"SMILES compliant with Lipinski Rule: {len(lipinski_smiles)}")

    # Save filtered results
    result_series = pd.Series(lipinski_smiles)
    result_series.to_csv(OUTPUT_CSV, index=False, header=["SMILES"])
    print(f"Filtered data saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()