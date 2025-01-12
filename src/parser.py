import argparse
from typing import Any, Dict, Optional


def parse_train_args(args: Optional[Dict[str, Any]] = None):
    parser = argparse.ArgumentParser(description="train")

    # experiment config
    parser.add_argument("--wandb_name", type=str, default="default")
    # parser.add_argument("--dataset", type=str, required=True, choices=["sudoku"])
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", type=str, default="")

    # model config
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--n_loop", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", type=bool, default=False)
    parser.add_argument("--is_causal", type=bool, default=False)

    # Dataset config
    parser.add_argument("--data_dir", type=str, default="data/sudoku")
    parser.add_argument("--maxlen", type=int, default=120)
    parser.add_argument("--maxdata", type=int, default=120)
    parser.add_argument("--maxans", type=int, default=30)
    parser.add_argument("--vocab", type=int, default=21)
    parser.add_argument("--num_range", type=int, default=100)

    # training config
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)

    args = parser.parse_args()
    return args
