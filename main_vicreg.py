# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import wandb


# import dataset
import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets

import augmentations as aug
from distributed import init_distributed_mode

import resnet


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
                        help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Model
    parser.add_argument("--arch", type=str, default="resnet50",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=2048,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    
    #wandb
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Enable WandB logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="vicreg",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="sd6701-new-york-university",
        help="WandB entity (team or username). If None, use your default account.",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="vicregr101-100k",
        help="WandB run name (optional)",
    )
    parser.add_argument(
        "--wandb-api-key",
        type=str,
        default="14abcf8b33d9a7f066dd1988891a00fec55f4030",  # <-- RECOMMENDED: don't hardcode here
        help="WandB API key (optional; better to set via env WANDB_API_KEY)",
    )

        # Local VICRegL-style loss
    parser.add_argument(
        "--use-local-loss",
        action="store_true",
        help="Enable local VICReg loss on feature maps (VICRegL-style)",
    )
    parser.add_argument(
        "--local-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the local VICReg loss term",
    )
    parser.add_argument(
        "--local-mlp",
        type=str,
        default=None,
        help="MLP spec for local projector (e.g. '2048-2048'); if None, reuse --mlp",
    )


    return parser


def main(args):
    torch.backends.cudnn.benchmark = True
    # init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    wandb_run = None   # <<< add this line

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

        if args.enable_wandb :
            if args.wandb_api_key:
                wandb.login(key=args.wandb_api_key)
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config=vars(args),
                dir=str(args.exp_dir),
            )

    transforms = aug.TrainTransform()

    dataset = datasets.ImageFolder(args.data_dir / "train", transforms)

    if args.world_size > 1:
      sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    else:
       sampler = torch.utils.data.RandomSampler(dataset)    
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    # ---------------------------
# FIX FOR CPU TRAINING
# ---------------------------
    if args.device == "cpu":
        device = torch.device("cpu")
        model = VICReg(args).to(device)
    else:
        model = VICReg(args).cuda(args.local_rank if args.local_rank != -1 else 0)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )



    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(
            args.exp_dir / "model.pth", 
            map_location="cpu",        
            weights_only=False
        )
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)
            # x = x.to("cpu")
            # y = y.to("cpu")

            lr = adjust_learning_rate(args, optimizer, loader, step)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    lr=lr,
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                
                if wandb_run is not None:
                    wandb.log(stats)
                last_logging = current_time

        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")
    if args.rank == 0:
        final_model = model.module if hasattr(model, "module") else model
        torch.save(final_model.backbone.state_dict(), args.exp_dir / f"{args.arch}.pth")
    if args.rank == 0 and wandb_run is not None:
        wandb_run.finish()





def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_epochs = max(1, int(0.1 * args.epochs))  # e.g. 10% of training
    warmup_steps = warmup_epochs * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(args, self.embedding)
        self.use_local = getattr(args, "use_local_loss", False)
        self.local_loss_weight = getattr(args, "local_loss_weight", 1.0)
        if self.use_local:
            local_mlp = args.local_mlp if args.local_mlp is not None else args.mlp
            self.local_projector = Projector(args, self.embedding, mlp_string=local_mlp)
        else:
            self.local_projector = None

        self.distributed = getattr(args, "distributed", False)

    def _vicreg_loss(self, x, y):
        """
        Core VICReg loss (invariance + variance + covariance) on two embeddings.
        x, y: [N, D]
        """
        # your original structure, but slightly safer numerics
        x = x.float()
        y = y.float()

        repr_loss = F.mse_loss(x, y)

        if self.distributed:
            x = torch.cat(FullGatherLayer.apply(x), dim=0)
            y = torch.cat(FullGatherLayer.apply(y), dim=0)

        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        std_x = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0, unbiased=False) + 1e-4)
        std_loss = (
            torch.mean(F.relu(1 - std_x)) / 2
            + torch.mean(F.relu(1 - std_y)) / 2
        )

        batch_size = x.size(0)
        denom = max(batch_size - 1, 1)  # avoid divide by 0

        cov_x = (x.T @ x) / denom
        cov_y = (y.T @ y) / denom
        cov_loss = (
            off_diagonal(cov_x).pow_(2).sum().div(self.num_features)
            + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        )

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss

    def forward(self, x, y):
        # ---- GLOBAL BRANCH ----
        if self.use_local:
            # get global pooled features + feature maps
            g_x, fmap_x = self.backbone(x, return_featmap=True)
            g_y, fmap_y = self.backbone(y, return_featmap=True)
        else:
            g_x = self.backbone(x)
            g_y = self.backbone(y)
            fmap_x = fmap_y = None

        z_x = self.projector(g_x)
        z_y = self.projector(g_y)

        global_loss = self._vicreg_loss(z_x, z_y)

        # ---- LOCAL BRANCH (optional, VICRegL-style) ----
        if self.use_local and self.local_projector is not None:
            # fmap_x, fmap_y: [B, C, H, W]
            B, C, H, W = fmap_x.shape

            # flatten spatial positions into "local samples"
            lx = fmap_x.permute(0, 2, 3, 1).reshape(B * H * W, C)
            ly = fmap_y.permute(0, 2, 3, 1).reshape(B * H * W, C)

            lx = self.local_projector(lx)
            ly = self.local_projector(ly)

            local_loss = self._vicreg_loss(lx, ly)
            loss = global_loss + self.local_loss_weight * local_loss
        else:
            loss = global_loss

        return loss

def Projector(args, embedding, mlp_string: str | None = None):
    if mlp_string is None:
        mlp_string = args.mlp
    mlp_spec = f"{embedding}-{mlp_string}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg-training-script', parents=[get_arguments()])
    args = parser.parse_args()
    args.distributed = False   # <<< ADD THIS LINE
    if not hasattr(args, 'rank'):
        args.rank = 0
    print(f"Running VICReg with args: {args}")
    if args.device == "cpu":
        args.distributed = False
        args.world_size = 1
        args.local_rank = -1
    main(args)
