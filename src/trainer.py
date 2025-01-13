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
                    project="backtrack",
                    config=args,
                    name=args.wandb_name,
                    id=run_id,
                    resume="must",
                )

            else:
                wandb.init(project="backtrack", config=args, name=args.wandb_name)

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
                if args.dataset_name in {"sudoku", "countdown", "sat"}:
                    logits = torch.stack(logits)
                    logits = logits.permute(1, 0, 2, 3)
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        y.unsqueeze(1).expand(-1, logits.size(1), -1).reshape(-1),
                    )
                else:
                    logit = logits[-1]  # (batch_size, seq_len, vocab_size)
                    loss = self.criterion(logit.transpose(1, 2), y)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()

                if (i + 1) % args.log_interval == 0 and dist.get_rank() == 0:
                    loss = total_loss / args.log_interval
                    print(f"Epoch {epoch} Iter {i} Loss {loss}")
                    wandb.log({"loss": loss})
                    wandb.log({"lr": self.scheduler.get_last_lr()[0]})
                    total_loss = 0

            Sum, correct = 0, 0
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(self.test_loader):
                    # batch = {k: v.to(self.device) for k, v in batch.items()}
                    # logits = model(batch["input_ids"])
                    # acc += (logits.argmax(1) == batch["labels"]).float().mean().item()
                    inputs, y = batch
                    inputs, y = inputs.cuda(), y.cuda()
                    logits = model(inputs)
                    logits = logits[-1]
                    Sum += torch.as_tensor(inputs.shape[0]).cuda()
                    truth = torch.where(y > 0, 1, 0)
                    predict = (
                        torch.where(torch.argmax(logits, dim=2) == y, 1, 0) * truth
                    )
                    correct += torch.sum(
                        torch.where(
                            torch.sum(truth, dim=1) == torch.sum(predict, dim=1), 1, 0
                        )
                    )

            dist.all_reduce(Sum)
            dist.all_reduce(correct)
            acc = correct / Sum
            if dist.get_rank() == 0:
                print(f"Epoch {epoch} Acc {acc}")
                wandb.log({"acc": acc})

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))
