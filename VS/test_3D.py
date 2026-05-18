import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import time


def can_generate_conformer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        # 尝试生成一个三维构象
        mol = Chem.AddHs(mol)  # 添加氢原子
        conf_id = AllChem.EmbedMolecule(mol, randomSeed=42)
        return conf_id != -1  # -1 表示失败
    except:
        return False


# 输入输出文件路径
input_file = "spr_selected_smiles.csv"
output_valid_file = 'vaild_smiles.csv'

# 读取 CSV 文件（假设 SMILES 存在 'smiles' 列）
df = pd.read_csv(input_file)

# 获取数据长度并开始计时
total_rows = len(df)
start_time = time.time()
processed_count = 0

# 筛选出能生成构象的 SMILES
valid_smiles = []

for idx, row in df.iterrows():
    smiles = row['SMILES']
    if can_generate_conformer(smiles):
        print(row['SMILES'])
        valid_smiles.append(row)
    processed_count += 1

    # 每处理 100 条打印一次进度
    if processed_count % 1 == 0 or processed_count == total_rows:
        elapsed_time = time.time() - start_time
        print(f"已处理 {processed_count}/{total_rows} 条记录，耗时: {elapsed_time:.2f} 秒")

# 转换为 DataFrame
valid_df = pd.DataFrame(valid_smiles)

# 保存结果到新文件
valid_df.to_csv(output_valid_file, index=False)

print(f"已筛选出 {len(valid_df)} 条可生成构象的 SMILES，结果保存至: {output_valid_file}")


