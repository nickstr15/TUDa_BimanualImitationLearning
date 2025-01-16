import math
from typing import TypeAlias, Union, Iterable, Dict, Any

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from robomimic.config import Config

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

def optimizer_from_optim_params(optim_params: Config, thetas: ParamsT):
    """
    Helper function to return a torch Optimizer from the optim_params
    section of the config for a particular network.

    Args:
        optim_params (Config): optimizer parameters

        thetas (list): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.

    Returns:
        optimizer (torch.optim.Optimizer): optimizer
    """
    optimizer_type = optim_params.get("optimizer_type", "adam")
    lr = optim_params["learning_rate"]["initial"]
    betas = optim_params.get("betas", (0.9, 0.95))

    if optimizer_type == "adam":
        return optim.Adam(
            params=thetas,
            lr=lr,
            betas=betas,
        )
    elif optimizer_type == "adamw":
        return optim.AdamW(
            params=thetas,
            lr=lr,
            betas=betas,
        )
    else:
        raise ValueError("Invalid optimizer type: {}".format(optimizer_type))


def lr_scheduler_from_optim_params(scheduler_params, optimizer, num_epochs):
    """
    Helper function to return a LRScheduler from the optim_params
    section of the config for a particular network. Returns None
    if a scheduler is not needed.

    Args:
        scheduler_params (Config): scheduler parameters
        optimizer (torch.optim.Optimizer): optimizer corresponding to the scheduler
        num_epochs (int): total number of training epochs

    Returns:
        lr_scheduler (torch.optim.lr_scheduler or None): learning rate scheduler
    """
    scheduler_type = scheduler_params.get("scheduler_type", "cosine")
    warmup_epochs = scheduler_params.get("warmup_epochs", 0)
    epoch_schedule = scheduler_params.get("epoch_schedule", [])
    decay_factor = scheduler_params.get("decay_factor", 0.1)

    # linear scheduler
    if scheduler_type == "linear":
        if len(epoch_schedule) == 0:
            return None

        assert len(epoch_schedule) == 1
        end_epoch = epoch_schedule[0]

        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_epochs=warmup_epochs,
            end_epoch=end_epoch,
            decay_factor=decay_factor,
        )

    # multistep scheduler
    elif scheduler_type == "multistep":
        if len(epoch_schedule) <= 0:
            return None

        return get_multistep_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_epochs=warmup_epochs,
            milestones=epoch_schedule,
            gamma=decay_factor,
        )

    # cosine scheduler
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_episodes=warmup_epochs,
            num_epochs=num_epochs,
        )
    else:
        raise ValueError("Invalid LR scheduler scheduler_type: {}".format(scheduler_type))

def get_linear_schedule_with_warmup(optimizer, num_warmup_epochs, end_epoch,
                                    warmup_start_factor=0.1, decay_factor=0.0):
    """
    Returns a scheduler with a linear warmup followed by a linear decay.

    Args:
        optimizer (Optimizer): Optimizer to be scheduled.
        num_warmup_epochs (int): Number of warmup epochs.
        end_epoch (int): after this epoch the final decay factor is reached.
        warmup_start_factor (float): Initial learning rate factor during warmup (default: 0.1).
        decay_factor (float): Final learning rate factor during decay (default: 0.0).

    Returns:
        SequentialLR: A PyTorch SequentialLR scheduler.
    """
    # Warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, total_iters=num_warmup_epochs
    )

    # Linear decay scheduler
    decay_epochs = end_epoch - num_warmup_epochs
    decay_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=decay_factor, total_iters=decay_epochs
    )

    # Combine warmup and decay
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[num_warmup_epochs]
    )

    return scheduler

def get_multistep_schedule_with_warmup(optimizer, num_warmup_epochs, milestones, gamma=0.1, warmup_start_factor=0.1):
    """
    Returns a scheduler with a linear warmup followed by a MultiStepLR decay.

    Args:
        optimizer (Optimizer): Optimizer to be scheduled.
        num_warmup_epochs (int): Number of warmup epochs.
        milestones (list): List of epoch indices to decay the learning rate.
        gamma (float): Multiplicative factor of learning rate decay (default: 0.1).
        warmup_start_factor (float): Initial learning rate factor during warmup (default: 0.1).

    Returns:
        SequentialLR: A PyTorch SequentialLR scheduler.
    """
    # Warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, total_iters=num_warmup_epochs
    )

    # MultiStepLR decay scheduler
    decay_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )

    # Combine warmup and multistep decay
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[num_warmup_epochs]
    )

    return scheduler

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_episodes, num_epochs, num_cycles = 0.5, last_epoch: int = -1, warmup_start_factor=0.1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Copied code from HuggingFace's diffusers library:

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_episodes (`int`):
            The number of episodes for the warmup phase.
        num_epochs (`int`):
            The total number of training epochs.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        warmup_start_factor (`float`, *optional*, defaults to 0.1):
            The factor to start the warmup from.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_episode):
        if current_episode < num_warmup_episodes:
            return max(float(current_episode) / float(max(1, num_warmup_episodes)), warmup_start_factor)
        progress = float(current_episode - num_warmup_episodes) / float(max(1, num_epochs - num_warmup_episodes))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
