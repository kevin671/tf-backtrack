import os
from parser import parse_train_args

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_scheduler, set_seed

from dataset.seq_align import getLoader
from model import LoopedTransformer
from trainer import Trainer


def main():
    args = parse_train_args()
    print(args)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.barrier()

    model = LoopedTransformer(args).cuda()
    if args.checkpoint_path:
        # TODO: load optimizer and scheduler
        model = LoopedTransformer.load_from_checkpoint(args.checkpoint_path).cuda()

    train_loader, test_loader = getLoader(args)  # TODO: Support different datasets
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type="cuda",
    )
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=args.epochs * len(train_loader),
    )

    # model = build_model(args)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    trainer = Trainer(model, train_loader, test_loader, optimizer, scheduler, args)
    trainer.train()


if __name__ == "__main__":
    main()
