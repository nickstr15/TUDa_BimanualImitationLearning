from types import NotImplementedType
from typing import TypeAlias, Union, Iterable, Dict, Any

import torch
import torch.optim as optim
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
    betas = optim_params.get("betas", (0.9, 0.999))

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


def lr_scheduler_from_optim_params(scheduler_params, optimizer):
    """
    Helper function to return a LRScheduler from the optim_params
    section of the config for a particular network. Returns None
    if a scheduler is not needed.

    Args:
        scheduler_params (Config): scheduler parameters
        optimizer (torch.optim.Optimizer): optimizer corresponding to the scheduler

    Returns:
        lr_scheduler (torch.optim.lr_scheduler or None): learning rate scheduler
    """
    scheduler_type = scheduler_params.get("type", "cosine")
    warmup_steps = scheduler_params.get("warmup_steps", 0)
    epoch_schedule = scheduler_params.get("epoch_schedule", [])
    decay_factor = scheduler_params.get("decay_factor", 0.1)

    # linear scheduler
    if scheduler_type == "linear":
        if len(epoch_schedule) == 0:
            return None

        assert len(epoch_schedule) == 1
        end_epoch = epoch_schedule[0]
        return optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=decay_factor,
            total_iters=end_epoch,
        )

    # multistep scheduler
    elif scheduler_type == "multistep":
        if len(epoch_schedule) <= 0:
            return None

        return optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=epoch_schedule,
            gamma=decay_factor,
        )

    # cosine scheduler
    elif scheduler_type == "cosine":
        raise NotImplementedError("Cosine scheduler not implemented yet.")
        # TODO: which Cosine scheduler to use? What additional params? How to handle warmup?


    else:
        raise ValueError("Invalid LR scheduler scheduler_type: {}".format(scheduler_type))
