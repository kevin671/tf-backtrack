import json
import os

import torch
from torch.utils.data import Dataset

OP_TO_CODE = {"+": 1, "-": 2, "*": 3, "/": 4, "=": 5}
OFFSET = len(OP_TO_CODE)


class CountDownDataset(Dataset):
    CLS_TOKEN = 0

    def __init__(self, args, control):
        if control == 0:
            file_path = args.file_name
        elif control == 1:
            file_path = args.file_name.replace("train", "val")
        json_file_path = os.path.join(args.data_dir, file_path)
        with open(json_file_path, "r") as f:
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
                    parts.append(
                        int(num) + OFFSET
                    )  # 数値にOFFSETを加算してリストに追加
                    num = ""
                parts.append(OP_TO_CODE[char])  # 演算子をコードに変換して追加
        if num:
            parts.append(int(num) + OFFSET)  # 最後の数値を追加

        return parts

    def __getitem__(self, idx):
        sample = self.data[idx]
        nums = [num + OFFSET for num in sample["nums"]]  # numsにOFFSETを加算
        target = sample["target"] + OFFSET  # ターゲットにOFFSETを加算

        solution_steps = []
        for step in sample["solution"]:
            solution_steps.extend(
                self.parse_expression(step)
            )  # すべての手順をリストに展開

        # テンソル形式に変換
        _input_tensor = torch.tensor(nums + [target], dtype=torch.long)
        _output_tensor = torch.tensor(solution_steps, dtype=torch.long)

        # input_tensorの後ろにoutput_tensorを予測するためにその長さだけCLS_TOKENを追加
        # 逆にoutput_tensorの先頭にはロスを計算しないためにラベル0を追加
        input_tensor = torch.cat(
            [
                _input_tensor,
                torch.tensor([self.CLS_TOKEN] * len(_output_tensor), dtype=torch.long),
            ]
        )
        output_tensor = torch.cat(
            [
                torch.tensor([self.CLS_TOKEN] * len(_input_tensor), dtype=torch.long),
                _output_tensor,
            ]
        )

        return input_tensor, output_tensor


if __name__ == "__main__":
    # 使用例
    json_file_path = "/work/gg45/g45004/tf-backtrack/data/b4_3_random/train1_b4_t100_n10_random.json"  # JSONファイルパス
    dataset = CountDownDataset(json_file_path)

    # 動作確認
    for i in range(len(dataset)):
        x, y = dataset[i]
        # print(x.shape, y.shape)  # torch.Size([5]) torch.Size([15])
        print(x.shape, y.shape)  # torch.Size([20]) torch.Size([20])
        print(f"Input (nums): {x.numpy()}, Output (solution steps): {y.numpy()}")
        # Input (nums): [26 20 83  7 74  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0], Output (solution steps): [  0   0   0   0   0  26   1  83   5 104  20   3   7   5  35 104   2  35 5  74]
