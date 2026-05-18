import pandas as pd
from rdkit import Chem

def validate_smiles(smiles):
    """
    使用 RDKit 验证 SMILES 字符串是否有效
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def main():
    # 文件名（相对路径，适合 GitHub 项目）
    INPUT_SPR = "spr_data.csv"
    INPUT_G4LDB = "G4LDB.csv"
    OUTPUT_FILE = "spr_data_g4ldb_valid.csv"

    # 读取数据
    spr_data = pd.read_csv(INPUT_SPR)
    g4ldb_data = pd.read_csv(INPUT_G4LDB)

    # 基于 ligandId 左连接合并
    merged_data = pd.merge(spr_data, g4ldb_data, on="ligandId", how="left")
    print(f"合并后数据条数: {len(merged_data)}")

    # 过滤有效 SMILES
    merged_data["valid_smiles"] = merged_data["smiles"].apply(validate_smiles)
    valid_data = merged_data[merged_data["valid_smiles"]].drop(columns=["valid_smiles"])

    print(f"过滤有效 SMILES 后条数: {len(valid_data)}")

    # 保存结果
    valid_data.to_csv(OUTPUT_FILE, index=False)
    print(f"结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
