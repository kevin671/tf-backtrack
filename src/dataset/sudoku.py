import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SudokuDataset(Dataset):
    def __init__(self, args, control):
        """
        Sudoku Dataset for Machine Learning
        Args:
            csv_file (str): Path to the CSV file containing the dataset.
            transform (callable, optional): A function/transform to apply to the data.
        """
        if control == 0:
            csv_file = os.path.join(args.data_dir, args.file_name)
        elif control == 1:
            csv_file = os.path.join(
                args.data_dir, args.file_name.replace("train", "test")
            )
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Puzzle and solution as strings
        puzzle_str = self.data.loc[idx, "puzzle"]
        solution_str = self.data.loc[idx, "solution"]

        puzzle = np.array(
            [int(char) if char != "." else 0 for char in puzzle_str], dtype=int
        )
        solution = np.array([int(char) for char in solution_str], dtype=int)

        # clues = self.data.loc[idx, 'clues']
        # difficulty = self.data.loc[idx, 'difficulty']

        # sample = {
        #    'puzzle': torch.tensor(puzzle, dtype=torch.long)
        #    'solution': torch.tensor(solution, dtype=torch.long)
        #    'clues': torch.tensor(clues, dtype=torch.float32),
        #    'difficulty': torch.tensor(difficulty, dtype=torch.float32)
        # }

        return torch.tensor(puzzle, dtype=torch.long), torch.tensor(
            solution, dtype=torch.long
        )


if __name__ == "__main__":
    # df = pd.read_csv(filename)
    # print(df.shape) # (3000000, 5)
    # puzzle, solution, clues, difficulty
    # print(df['puzzle'][0]) # 1..5.37..6.3..8.9......98...1.......8761..........6...........7.8.9.76.47...6.312
    # print(df['solution'][0]) # 198543726643278591527619843914735268876192435235486179462351987381927654759864312
    # print(df['clues'][0]) # 27
    # print(df['difficulty'][0]) # 2.2

    filename = "data/sudoku/train.csv"
    dataset = SudokuDataset(filename)
    print(len(dataset))  # 3000000
    puzzle, solution = dataset[0]
    print(puzzle.shape)  # torch.Size([81])
    print(solution.shape)  # torch.Size([81])

    """make dataset
    filename = "data/sudoku-3m.csv"
    df = pd.read_csv(filename)

    # split dataset into all, easy, medium, hard
    # all
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.shape)
    train_df = df.iloc[:100000]
    print(train_df.shape)
    test_df = df.iloc[100000:101000]
    print(test_df.shape)
    os.makedirs("data/sudoku/random", exist_ok=True)
    train_df.to_csv("data/sudoku/random/train.csv", index=False)
    test_df.to_csv("data/sudoku/random/test.csv", index=False)

    # extract dataset of rating 0.0
    df = df[df["difficulty"] == 0.0]
    # suffle
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df.iloc[:100000]
    print(train_df.shape)
    test_df = df.iloc[100000:101000]
    print(test_df.shape)
    os.makedirs("data/sudoku/easy", exist_ok=True)
    train_df.to_csv("data/sudoku/easy/train.csv", index=False)
    test_df.to_csv("data/sudoku/easy/test.csv", index=False)

    # extract dataset of rating not 0.0 but less than 3.0
    df = pd.read_csv(filename)
    df = df[(df["difficulty"] != 0.0) & (df["difficulty"] < 3.0)]
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.shape)
    train_df = df.iloc[:100000]
    print(train_df.shape)
    test_df = df.iloc[100000:101000]
    print(test_df.shape)
    os.makedirs("data/sudoku/medium", exist_ok=True)
    train_df.to_csv("data/sudoku/medium/train.csv", index=False)
    test_df.to_csv("data/sudoku/medium/test.csv", index=False)

    # extract dataset of rating 3.0 or more
    df = pd.read_csv(filename)
    df = df[df["difficulty"] >= 3.0]
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.shape)
    train_df = df.iloc[:100000]
    print(train_df.shape)
    test_df = df.iloc[100000:101000]
    print(test_df.shape)
    os.makedirs("data/sudoku/hard", exist_ok=True)
    train_df.to_csv("data/sudoku/hard/train.csv", index=False)
    test_df.to_csv("data/sudoku/hard/test.csv", index=False)
    """
