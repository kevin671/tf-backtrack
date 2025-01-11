from typing import Dict, Any, Optional
import argparse

def parse_train_args(
    args: Optional[Dict[str, Any]] = None
):
    parser = argparse.ArgumentParser(description="train")

    # experiment config
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--dataset", type=str, required=True, choices=["sudoku"])

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
    
    # training config
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()
    return args
