import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Sudoku_Dataset(Dataset):
    def __init__(self, csv_file='data/sudoku-3m.csv', transform=None):
        """
        Sudoku Dataset for Machine Learning
        Args:
            csv_file (str): Path to the CSV file containing the dataset.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Puzzle and solution as strings
        puzzle_str = self.data.loc[idx, 'puzzle']
        solution_str = self.data.loc[idx, 'solution']

        puzzle = np.array([int(char) if char != '.' else 0 for char in puzzle_str], dtype=np.int64)
        solution = np.array([int(char) for char in solution_str], dtype=np.int64)

        #clues = self.data.loc[idx, 'clues']
        #difficulty = self.data.loc[idx, 'difficulty']

        #sample = {
        #    'puzzle': torch.tensor(puzzle, dtype=torch.long)
        #    'solution': torch.tensor(solution, dtype=torch.long)
        #    'clues': torch.tensor(clues, dtype=torch.float32),
        #    'difficulty': torch.tensor(difficulty, dtype=torch.float32)
        #}

        if self.transform:
            sample = self.transform(sample)

        return torch.tensor(puzzle, dtype=torch.long), torch.tensor(solution, dtype=torch.long)

if __name__ == '__main__':
    filename = 'data/sudoku-3m.csv'
    #df = pd.read_csv(filename)
    # print(df.shape) # (3000000, 5)
    # puzzle, solution, clues, difficulty
    #print(df['puzzle'][0]) # 1..5.37..6.3..8.9......98...1.......8761..........6...........7.8.9.76.47...6.312
    #print(df['solution'][0]) # 198543726643278591527619843914735268876192435235486179462351987381927654759864312
    #print(df['clues'][0]) # 27
    #print(df['difficulty'][0]) # 2.2

    dataset = Sudoku_Dataset(filename)
    print(len(dataset)) # 3000000
    puzzle, solution = dataset[0]
    print(puzzle.shape) # torch.Size([81])
    print(solution.shape) # torch.Size([81])