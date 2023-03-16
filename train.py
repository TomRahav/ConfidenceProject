import os
import argparse
import yaml

import torch
import torchvision
import torchvision.transforms as T
from torch.optim import SGD
from torch_optimizer import Lookahead
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR
from loss import DistillationLoss, DistillationLossNoKD, DistillationLossDistribution
from resnet import resnet18
from classifier import ArgMaxClassifier
from training import DistillationTrainer
import wandb

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)

parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Confidence Project')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

# Training parameters
group = parser.add_argument_group('Training parameters')
group.add_argument('--seed', type=int, default=403, metavar='N',
                    help='used seed (default: 403 = 400 (ת) + 3 (ג)')
group.add_argument('--experiment', type=float, default=None, metavar='N',
                    help='used seed (default: 403 = 400 (ת) + 3 (ג)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
group.add_argument('--early-stopping', type=int, default=None, metavar='N',
                    help='number of epochs that need to passs without improvment before we stop')
group.add_argument('--dataset', type=str, default='cifar10', metavar='DATASET',
                    help='(default: "cifar10"')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=128, metavar='N',
                    help='Validation batch size override (default: 128)')

# Loss function parameters
group = parser.add_argument_group('Loss function parameters')
group.add_argument('--student-temp', type=float, default=1, metavar='M',
                    help='Student temprature')
group.add_argument('--teacher-temp', type=float, default=1, metavar='M',
                    help='Teacher temprature')
#group.add_argument('--teacher-loss', action='store_false', default=True,
#                    help='Add Teacher loss')
group.add_argument('--teacher-loss', default='kd', type=str,
                    help='options: kd, no_kd, distribution')
group.add_argument('--eta', default=0.5, type=float, help='hyperparameter for interpolating klDIV loss and cross-entropy loss')
# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
# group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
#                     help='Optimizer (default: "sgd")')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=1e-4,
                    help='weight decay (default: 2e-5)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
group.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--patience-epochs', type=int, default=2, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10)')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')


args, args_text = _parse_args()

mean = {
'cifar10': (0.4914, 0.4822, 0.4465),
'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
'cifar10': (0.2471, 0.2435, 0.2616),
'cifar100': (0.2675, 0.2565, 0.2761),
}

run = wandb.init(project="Confidence", entity="tom-rahav",
config={
    "epochs": args.epochs,
    "lr": args.lr,
    "momentum": args.momentum,
    "weight_decay": args.weight_decay,
    "decay_rate": args.decay_rate,
    "patience_epochs": args.patience_epochs,
    "early_stopping": args.early_stopping,
    "checkpoint_file": f'checkpoints/confidence_exp_{args.experiment}',
    "data_dir": os.path.expanduser('~/.pytorch-datasets'),
    "dataset": args.dataset,
    "seed": args.seed,
    "data_mean": mean[args.dataset],
    "data_std": std[args.dataset],
    "train_batch_size": args.batch_size,
    "test_batch_size": args.validation_batch_size,
    "student_temp": args.student_temp,
    "teacher_temp": args.teacher_temp,
    "teacher_loss": args.teacher_loss,
    "eta": args.eta,
    })

config = wandb.config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = config.seed
torch.manual_seed(seed)


# cifar10_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(config.data_mean, config.data_std),
            ]
        )

# Define datasets, currently if args.dataset is not default its cifar100
if config.dataset == 'cifar10':
    ds_train = torchvision.datasets.CIFAR10(root=config.data_dir, download=True, train=True, transform=transform)
    ds_test = torchvision.datasets.CIFAR10(root=config.data_dir, download=True, train=False, transform=transform)
elif config.dataset == 'cifar100':
    ds_train = torchvision.datasets.CIFAR100(root=config.data_dir, download=True, train=True, transform=transform)
    ds_test = torchvision.datasets.CIFAR100(root=config.data_dir, download=True, train=False, transform=transform)

# Define dataloaders

dl_train = torch.utils.data.DataLoader(ds_train, config.train_batch_size, shuffle=False)
dl_test = torch.utils.data.DataLoader(ds_test, config.test_batch_size, shuffle=False)

# Define models

# Untrained model (student)
student_model = resnet18(dataset=config.dataset)
student_classifier = ArgMaxClassifier(student_model)

# Pretrained model (teacher)
teacher_model = resnet18(dataset=config.dataset, pretrained=True)
teacher_classifier = ArgMaxClassifier(teacher_model)

wandb.watch(models=(student_classifier, teacher_classifier))

# Define optimizer and schedular
optimizer = SGD(params=student_classifier.parameters(), 
                            lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
optimizer = Lookahead(optimizer, k=5, alpha=0.5)
if args.sched == 'cosine':
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=10000)
elif args.sched == 'plateau':
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config.decay_rate,
        patience=config.patience_epochs)
else:
    scheduler = MultiStepLR(
        optimizer, 
        milestones=[75*(i+1) for i in range(int(config.epochs/75))],
        gamma=0.2) #learning rate decay

# Define loss function
if config.teacher_loss == 'kd':
    loss_fn = DistillationLoss(teacher_temp=config.teacher_temp, student_temp=config.student_temp, alpha = config.eta, beta = 1 - config.eta)
elif config.teacher_loss == 'no_kd':
    loss_fn = DistillationLossNoKD(student_temp=config.student_temp)
elif config.teacher_loss == 'distribution':
    loss_fn = DistillationLossDistribution(teacher_temp=config.teacher_temp, student_temp=config.student_temp, alpha = config.eta, beta = 1 - config.eta)

# Define trainer
trainer = DistillationTrainer(student_classifier, teacher_classifier, loss_fn, optimizer, device, teacher_temp=config.teacher_temp, student_temp=config.student_temp)

res = trainer.fit(dl_train, dl_test, config.epochs, run, checkpoints=config.checkpoint_file, early_stopping=config.early_stopping)


wandb.finish()
