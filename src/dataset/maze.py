import heapq
import json
import os

import numpy as np


# Generate a dataset of maze navigation tasks with 81-character strings for input and solution
def generate_maze_dataset(num_samples, n, num_walls_range, file_path):
    """
    Generates a dataset of maze navigation tasks in a seq-to-seq format.
    Args:
        num_samples (int): Number of samples to generate.
        n (int): Size of the maze (n x n).
        num_walls_range (tuple): Range of number of walls to place (min, max).

    Returns:
        df (DataFrame): A DataFrame containing maze data as strings.
    """
    data = []
    for _ in range(num_samples):

        solution_path = None
        distance = 0
        # 迷路を生成して、解が見つかるまで繰り返し生成
        while solution_path is None or distance <= n:
            # ランダムな start と goal を生成（同じ位置にならないようにする）
            while True:
                start = (np.random.randint(0, n), np.random.randint(0, n))
                goal = (np.random.randint(0, n), np.random.randint(0, n))
                if start != goal:
                    break

            # start = (0, 0)
            # goal = (n - 1, n - 1)
            num_walls = np.random.randint(*num_walls_range)
            maze = np.zeros((n, n), dtype=int)

            # Place start, goal, and walls
            maze[start] = 1  # Start cell
            maze[goal] = 2  # Goal cell
            free_positions = [
                (i, j)
                for i in range(n)
                for j in range(n)
                if (i, j) != start and (i, j) != goal
            ]
            wall_positions = np.random.choice(
                len(free_positions), num_walls, replace=False
            )
            for pos in wall_positions:
                maze[free_positions[pos]] = -1  # Wall cell

            solution_path, distance = solve_maze_a_star(maze, start, goal)
            # print(solution_path, distance)

        # Convert maze to input string (81 characters for 9x9 grid)
        maze[goal] = 1
        # print(maze)
        maze_string = maze.flatten()
        solved_maze = maze.copy()  # 元の迷路をコピー
        for pos in solution_path:
            if maze[pos] == 0:  # 壁やスタート、ゴール以外の経路を 1 にする
                solved_maze[pos] = 1
        solved_maze_string = solved_maze.flatten()

        # -1を2に変換
        maze_string = np.where(maze_string == -1, 2, maze_string).tolist()
        solved_maze_string = np.where(
            solved_maze_string == -1, 2, solved_maze_string
        ).tolist()

        data.append(
            {"maze": maze_string, "solution": solved_maze_string, "distance": distance}
        )

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f)


def heuristic(a, b):
    """ヒューリスティック関数（マンハッタン距離を使用）"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def solve_maze_a_star(maze, start=(0, 0), goal=None):
    """
    A* アルゴリズムを用いて迷路の最短経路と距離を求める。
    Args:
        maze (numpy.ndarray): n x n の迷路 (0: 空白, -1: 壁, 1: スタート, 2: ゴール)。
        start (tuple): スタート地点 (x, y)。
        goal (tuple): ゴール地点 (x, y)。デフォルトは (n-1, n-1)。

    Returns:
        path (list): スタートからゴールまでの経路 (座標リスト)。
        distance (int): 最短距離。到達不能の場合は -1。
    """
    n = maze.shape[0]
    if goal is None:
        goal = (n - 1, n - 1)

    # 優先度付きキュー (最小ヒープ)
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}  # 経路を復元するための辞書
    g_score = {start: 0}  # スタートから各ノードへの実際の距離
    f_score = {start: heuristic(start, goal)}  # 推定総コスト

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            # ゴールに到達した場合、経路を復元
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()  # スタートからゴールまでの順にする
            return path, len(path) - 1

        # 現在のノードの隣接ノードを探索
        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
        for neighbor in neighbors:
            x, y = neighbor
            if 0 <= x < n and 0 <= y < n and maze[x, y] != -1:  # 壁でない場所のみ
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # ゴールに到達できなかった場合
    return None, -1


import torch
from torch.utils.data import Dataset


class MazeDataset(Dataset):
    def __init__(self, args, control):
        file_path = os.path.join(args.data_dir, args.file_name)
        if control == 1:
            file_path = file_path.replace("train", "test")

        with open(file_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        maze = sample["maze"]  # 入力文字列
        solution = sample["solution"]  # 出力文字列
        # distance = sample["distance"]  # 距離
        # 入力とラベルの tensor 形式に変換
        input_tensor = torch.tensor(maze, dtype=torch.long)
        output_tensor = torch.tensor(solution, dtype=torch.long)
        return input_tensor, output_tensor


if __name__ == "__main__":
    complexity = 9
    wall_density = [0.2, 0.6]
    num_walls_range = (
        int(complexity**2 * wall_density[0]),
        int(complexity**2 * wall_density[1]),
    )

    num_sample = [100000, 1000]
    mode = ["train", "test"]

    # file_path = "data/maze/train.json"
    # generate_maze_dataset(num_samples=100000, n=complexity, num_walls_range=num_walls_range, file_path=file_path)

    for i in range(2):
        file_path = f"data/maze/n_{complexity}_wall_{wall_density[0]}_{wall_density[1]}/{mode[i]}.json"
        generate_maze_dataset(
            num_samples=num_sample[i],
            n=complexity,
            num_walls_range=num_walls_range,
            file_path=file_path,
        )

    # maze_dataset = MazeDataset()
    # print(len(maze_dataset))
    # for i in range(len(maze_dataset)):
    #    sample = maze_dataset[i]
    #    input_tensor, output_tensor = sample
    #    print(input_tensor, output_tensor)
    #    print(
    #        input_tensor.size(), output_tensor.size()
    #    )  # torch.Size([25]) torch.Size([25])
