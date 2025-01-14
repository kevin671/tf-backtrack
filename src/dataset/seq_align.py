import numpy as np

alphabet = [i for i in "abcdefghijklmnopqrstuvwxyz"]


### Global Alignment: Needleman-Wunsch Algorithm


### Levenshtein distance: Global Alignment with (match=0, mismatch=1, gap=1), Minimize cost


def get_seq(diff, args):
    using = np.random.randint(args.using) + 3
    using = min(using, len(alphabet))
    available = np.random.choice(alphabet, using, replace=False)
    str1 = np.random.randint(using, size=args.length)
    str1 = [available[i] for i in str1]
    if np.random.rand() < 0.4:
        length = np.random.randint(args.length - 3, args.length + 3)
        str2 = np.random.randint(using, size=length)
        str2 = [available[i] for i in str2]
    else:
        str2 = str1[:]
        for _ in range(diff):
            a = np.random.randint(3)
            if a == 0 and len(str2) > 2:
                p = np.random.randint(len(str2))
                str2 = str2[:p] + str2[p + 1 :]
            elif a == 1:
                p = np.random.randint(len(str2))
                str2 = str2[:p] + [np.random.choice(available)] + str2[p + 1 :]
            else:
                p = np.random.randint(len(str2) + 1)
                str2 = str2[:p] + [np.random.choice(available)] + str2[p:]
    if str1 == str2 or len(str2) >= args.length + 3 or len(str2) < args.length - 3:
        return get_seq(diff, args)
    if len(str1) > len(str2):
        return str2, str1
    return str1, str2


def solve(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 3
            matrix[i][j] = min(
                matrix[i - 1][j] + 2, matrix[i][j - 1] + 2, matrix[i - 1][j - 1] + d
            )
    return matrix[-1][-1]


### Longest Common Subsequence: Global Alignment with (match=1, mismatch=0, gap=0), Maximize


### Local Alignment: Smith-Waterman Algorithm


import torch

### Dataset Class for Sequence Alignment
from torch.utils.data import Dataset


class SequenceAlignmentDataset(Dataset):
    def __init__(self, args, control):
        num_range = args.num_range
        dictionary = {"<pad>": 0, "<sep>": 1, "<eos>": 2, "|": 3, "[CLS]": 4}
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        for i in range(26):
            dictionary[alphabet[i]] = i + len(dictionary)
        for i in range(num_range):
            dictionary[str(i)] = i + len(dictionary) + len(alphabet)

        file_name = args.data_dir
        if control == 0:
            with open(f"{file_name}/train.txt", "r") as f:
                self.X = f.read().splitlines()
        elif control == 1:
            with open(f"{file_name}/test.txt", "r") as f:
                self.X = f.read().splitlines()

        def toToken(sentences):
            token_list = list()
            y_list = list()
            for sentence in sentences:
                # remove the last answer token
                #  l j y k p r k y l p p k l y y j k j z r j r p z r <sep> 67
                parts = sentence.split("<sep>")
                sentence = parts[0]
                ans = int(parts[1].strip())
                arr = [dictionary[s] for s in sentence.split()]
                padding = [0 for _ in range(args.maxlen - len(arr))]
                y = (
                    [0 for _ in range(len(arr) - 1)]
                    + [ans]
                    + [0 for _ in range(args.maxlen - len(arr))]
                )
                arr = arr + padding
                token_list.append(torch.Tensor(arr))
                y_list.append(torch.Tensor(y))
            return torch.stack(token_list).int(), torch.stack(y_list).long()

        self.X, self.Y = toToken(self.X)
        # self.Y = getY(self.X).long()
        # self.X = self.X[:, :-1]
        # self.Z = torch.argmax(torch.where(self.X == dictionary["<sep>"], 1, 0), dim=1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]  # , self.Z[idx]


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Sequence Alignment")
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument("--using", type=int, default=26)
    parser.add_argument("--train_size", type=float, default=1e6)
    parser.add_argument("--test_size", type=float, default=1e3)
    parser.add_argument("--save_dir", type=str, default="data/ED")
    args = parser.parse_args()

    # out_dir: data/seq_align/{args.length}
    out_dir = f"{args.save_dir}/{args.length}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/train.txt", "w") as f:
        for _ in range(int(args.train_size)):
            str1, str2 = get_seq(0, args)
            ans = solve(str1, str2)
            line = f"{' '.join(str1)} | {' '.join(str2)} <sep> {ans}\n"
            f.write(line)

    print("Train data for edit distance is generated.")

    with open(f"{out_dir}/test.txt", "w") as f:
        for _ in range(int(args.test_size)):
            str1, str2 = get_seq(0, args)
            ans = solve(str1, str2)
            line = f"{' '.join(str1)} | {' '.join(str2)} <sep> {ans}\n"
            f.write(line)

    print("Test data for edit distance is generated.")
