import torch
from torch.utils.data import Dataset
import json

OP_TO_CODE = {
    '+': 1,
    '-': 2,
    '*': 3,
    '/': 4,
    '=': 5
}
OFFSET = len(OP_TO_CODE)

class CountDownDataset(Dataset):
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)


    def __len__(self):
        return len(self.data)

    def parse_expression(self, expression):
        parts = []
        num = ""
        for char in expression:
            if char.isdigit():
                num += char  # 数字を連結
            else:
                if num:
                    parts.append(int(num) + OFFSET)  # 数値にOFFSETを加算してリストに追加
                    num = ""
                parts.append(OP_TO_CODE[char])  # 演算子をコードに変換して追加
        if num:
            parts.append(int(num) + OFFSET)  # 最後の数値を追加

        return parts

    def __getitem__(self, idx):
        sample = self.data[idx]
        nums = [num + OFFSET for num in sample['nums']]  # numsにOFFSETを加算
        target = sample['target'] + OFFSET  # ターゲットにOFFSETを加算

        solution_steps = []
        for step in sample['solution']:
            solution_steps.extend(self.parse_expression(step))  # すべての手順をリストに展開

        # テンソル形式に変換
        input_tensor = torch.tensor(nums + [target], dtype=torch.long)
        output_tensor = torch.tensor(solution_steps, dtype=torch.long)

        return input_tensor, output_tensor
    

# 使用例
json_file_path = "/work/gg45/g45004/tf-backtrack/data/b4_3_random/train1_b4_t100_n10_random.json"  # JSONファイルパス
dataset = CountDownDataset(json_file_path)

# 動作確認
for i in range(len(dataset)):
    x, y = dataset[i]
    print(x.shape, y.shape) # torch.Size([5]) torch.Size([15])
    print(f"Input (nums): {x.numpy()}, Output (solution steps): {y.numpy()}")