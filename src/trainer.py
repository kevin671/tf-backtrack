import torch
import torch.distributed as dist
from torch import nn

import wandb


class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, scheduler, args):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        # self.device = args.device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # set up wandb
        if dist.get_rank() == 0:
            if args.checkpoint_path:
                # checkpoint_path: "/work/gg45/g45004/parallel-looped-tf/output/ED_60_Loop_100/wvs7wzm0/epoch_40.pt"
                run_id = args.checkpoint_path.split("/")[-2]
                wandb.init(
                    project="timestep",
                    config=args,
                    name=args.wandb_name,
                    id=run_id,
                    resume="must",
                )

            else:
                wandb.init(project="timestep", config=args, name=args.wandb_name)

    def train(self):
        model, args = self.model, self.args
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                # batch = {k: v.to(self.device) for k, v in batch.items()}
                # logits = model(batch["input_ids"])
                # loss = self.criterion(logits, batch["labels"])
                inputs, y = batch
                inputs, y = inputs.cuda(), y.cuda()
                logits = model(inputs)
                logits = logits[-1]
                loss = self.criterion(logits.transpose(1, 2), y)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()

                if i % args.log_interval == 0 and i > 0:
                    wandb.log({"loss": total_loss / args.log_interval})
                    total_loss = 0

            acc = 0.0
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(self.test_loader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    logits = model(batch["input_ids"])
                    acc += (logits.argmax(1) == batch["labels"]).float().mean().item()
            acc /= len(self.test_loader)
            wandb.log({"acc": acc})

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))
