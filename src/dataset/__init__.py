import torch
from torch.utils.data import DataLoader

from dataset.countdown import CountDownDataset
from dataset.maze import MazeDataset
from dataset.regex import RegexDataset
from dataset.sat import SATDataset
from dataset.seq_align import SequenceAlignmentDataset
from dataset.sudoku import SudokuDataset


def get_dataset(args, i):
    if args.dataset_name == "countdown":
        return CountDownDataset(args, i)
    elif args.dataset_name == "sudoku":
        return SudokuDataset(args, i)
    elif args.dataset_name == "sat":
        return SATDataset(args)
    elif args.dataset_name == "ED":
        return SequenceAlignmentDataset(args, i)
    elif args.dataset_name == "regex":
        return RegexDataset(args, i)
    elif args.dataset_name == "maze":
        return MazeDataset(args, i)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")


def get_loader(args):
    number = 2
    datasets = [get_dataset(args, i) for i in range(number)]
    samplers = [
        torch.utils.data.distributed.DistributedSampler(datasets[i])
        for i in range(number)
    ]
    dataloaders = [
        DataLoader(
            datasets[i],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=10,
            drop_last=False,
            sampler=samplers[i],
            pin_memory=True,
        )
        for i in range(number)
    ]
    return dataloaders[0], dataloaders[1]
