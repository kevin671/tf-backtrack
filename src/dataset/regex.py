import random

import pandas as pd
import torch
from torch.utils.data import Dataset

CHARS = "abcdefghijklmnopqrstuvwxyz"


class RegexDataset(Dataset):
    def __init__(self, args, control):
        file_name = args.data_dir
        maxlen = args.maxlen
        if control == 0:
            data = pd.read_csv(f"{file_name}/train.csv")
        elif control == 1:
            data = pd.read_csv(f"{file_name}/test.csv")

        dictionary = {"<pad>": 0, "True": 1, "False": 2, "*": 3, ".": 4, "|": 5}
        for i in range(26):
            dictionary[CHARS[i]] = i + len(dictionary)

        self.X = []
        self.Y = []
        for _, row in data.iterrows():
            regex = row["regex"]
            string = row["string"]
            label = "True" if row["label"] == 1 else "False"

            regex_ints = [dictionary[c] for c in regex]
            string_ints = [dictionary[c] for c in string]
            separator = [dictionary["|"]]
            x = regex_ints + separator + string_ints
            padding = [0 for _ in range(maxlen - len(x))]
            y = (
                [0 for _ in range(len(x) - 1)]
                + [dictionary[label]]
                + [0 for _ in range(args.maxlen - len(x))]
            )
            x = x + padding
            self.X.append(torch.Tensor(x).int())
            self.Y.append(torch.Tensor(y).long())
        self.X = torch.stack(self.X)
        self.Y = torch.stack(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def generate_matching_string(regex):
    """正規表現にマッチする文字列を生成"""
    string = []
    for char in regex:
        if char == ".":
            string.append(random.choice(CHARS))  # 任意の1文字
        elif char.endswith("*"):
            if random.random() < 0.2:
                string.append("")  # 0回一致
            else:
                string.append(
                    random.choice(CHARS) * random.randint(1, 5)
                )  # 1回以上一致
        else:
            string.append(char)  # そのままの文字
    return "".join(string)


def perturb_string_to_false(regex, matching_string, max_attempts=10):
    """マッチしない文字列を生成する"""

    while True:
        # 1～max_attempts の範囲で操作回数をランダムに決定
        num_operations = random.randint(1, max_attempts)

        perturbed = list(matching_string)

        # サンプリングした回数だけ操作を行う
        for _ in range(num_operations):
            choice = random.choice(["remove", "replace", "add"])

            if choice == "remove" and len(perturbed) > 1:
                pos = random.randint(0, len(perturbed) - 1)
                perturbed.pop(pos)  # ランダムな位置の文字を削除
            elif choice == "replace":
                pos = random.randint(0, len(perturbed) - 1)
                perturbed[pos] = random.choice(CHARS)  # ランダムな文字に置換
            elif choice == "add":
                pos = random.randint(0, len(perturbed))
                perturbed.insert(pos, random.choice(CHARS))  # ランダムな位置に追加

        perturbed = "".join(perturbed)

        # マッチしない場合はその文字列を返す
        if not is_match(regex, perturbed):
            return perturbed


def is_match(regex, s):
    """動的計画法による正規表現マッチング判定"""
    dp = [[False] * (len(s) + 1) for _ in range(len(regex) + 1)]
    dp[0][0] = True

    for i in range(1, len(regex) + 1):
        if regex[i - 1] == "*" and i >= 2:
            dp[i][0] = dp[i - 2][0]

    for i in range(1, len(regex) + 1):
        for j in range(1, len(s) + 1):
            if regex[i - 1] == s[j - 1] or regex[i - 1] == ".":
                dp[i][j] = dp[i - 1][j - 1]
            elif regex[i - 1] == "*" and i >= 2:
                dp[i][j] = dp[i - 2][j] or (
                    dp[i][j - 1]
                    if regex[i - 2] == s[j - 1] or regex[i - 2] == "."
                    else False
                )

    return dp[len(regex)][len(s)]


def generate_dataset(
    dataset_size=1000, regex_length=10, string_max_length=20, max_attempts=10
):
    """
    正解と不正解が半々になるデータセットを生成
    """
    dataset = []

    while len(dataset) < dataset_size:
        # ランダムな正規表現を生成
        regex = []
        for _ in range(regex_length):
            r = random.random()
            if r < 0.3:
                regex.append(".")  # 任意の1文字
            elif r < 0.6:
                regex.append(
                    random.choice(CHARS) + "*"
                )  # 任意の文字の0回以上の繰り返し
            else:
                regex.append(random.choice(CHARS))  # 文字
        regex = "".join(regex)

        # **正解ラベル1 (True) の文字列を生成（whileで再生成）**
        matching_string = generate_matching_string(regex)
        while len(matching_string) > string_max_length:
            matching_string = generate_matching_string(regex)  # 超えたら再生成

        # Trueデータの追加
        dataset.append((regex, matching_string, 1))

        # Falseデータの追加（確実にFalseになる文字列を生成）
        try:
            false_string = perturb_string_to_false(regex, matching_string, max_attempts)
            dataset.append((regex, false_string, 0))
        except RuntimeError as e:
            print(e)

    return dataset


if __name__ == "__main__":
    import os

    # データセット生成
    complexity = 40
    string_max_length = complexity * 2
    max_attempts = complexity

    samples = [100000, 1000]
    mode = ["train", "test"]

    for i in range(2):
        file_path = f"data/regex/{complexity}/{mode[i]}.csv"
        dataset = generate_dataset(
            dataset_size=samples[i],
            regex_length=complexity,
            string_max_length=string_max_length,
            max_attempts=max_attempts,
        )
        df = pd.DataFrame(dataset, columns=["regex", "string", "label"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
