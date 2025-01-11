import os

import torch

# from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def parse_dimacs_to_list(filename, max_var=100):
    with open(filename, "r") as f:
        lines = f.readlines()
    clauses = []
    for line in lines:
        line = line.strip()
        if line.startswith("c") or line.startswith("p"):
            continue  # コメント・ヘッダー行をスキップ

        # リテラルをリストに変換し、0で終わるのでそれを除去
        clause = [int(lit) for lit in line.split() if lit != "0"]
        if clause:
            # 2 * max_varのリストで初期化
            encoded_clause = [0] * (2 * max_var)
            for lit in clause:
                if lit > 0:
                    encoded_clause[lit - 1] = 1  # 前半部分：正のリテラル
                else:
                    encoded_clause[max_var + abs(lit) - 1] = 1  # 後半部分：負のリテラル
            clauses.append(encoded_clause)

    return clauses


# Datasetクラス
class SATDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): DIMACSファイルが格納されているディレクトリのパス
        """
        self.files = []

        # ディレクトリ内のファイルを再帰的に探索
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".dimacs"):
                    filepath = os.path.join(subdir, file)
                    self.files.append(filepath)

        assert len(self.files) > 0, "No DIMACS files found in the specified directory."

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]

        # DIMACSファイルから節句を抽出
        clauses_list = parse_dimacs_to_list(filepath)
        clauses_tensor = torch.tensor(clauses_list, dtype=torch.long)
        # print(clauses_tensor.shape) # (clauses, max_var*2)

        # ファイル名から is_sat (SAT=1 or UNSAT=0) を取得
        is_sat = int(filepath.split("sat=")[-1].split(".")[0])  # sat=1 or sat=0
        label_tensor = torch.tensor(is_sat, dtype=torch.long)

        return clauses_tensor, label_tensor


if __name__ == "__main__":
    dimacs_dir = "/work/gg45/g45004/tf-backtrack/src/dataset/neurosat/dimacs/train/"
    sat_dataset = SATDataset(dimacs_dir)

    # Datasetサイズの確認
    print(f"Dataset size: {len(sat_dataset)}")
    clause, label = sat_dataset[0]
    print(clause.shape, label)  # torch.Size([75, 200]) tensor(0)
    clause, label = sat_dataset[1]
    print(clause.shape, label)  # torch.Size([83, 200]) tensor(1)

    # dimac_sample_path = "/work/gg45/g45004/tf-backtrack/src/dataset/neurosat/dimacs/train/sr5/grp1/sr_n=0005_pk2=0.30_pg=0.40_t=4_sat=0.dimacs"
    # a = parse_dimacs_to_list(dimac_sample_path)
    # a = torch.tensor(a, dtype=torch.long)
    # print(a.shape) # torch.Size([24, 10]
