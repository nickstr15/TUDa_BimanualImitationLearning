import torch
from torch.nn.modules.batchnorm import _BatchNorm as BatchNorm

class EMAModel:
    """
    Exponential Moving Average (EMA) of model weights.

    This class maintains an exponentially moving average of the weights of a model during training.
    EMA is a technique often used to stabilize and improve performance by smoothing weight updates.
    """
    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999
    ):
        """
        Initialize the EMA model and its parameters.

        If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
        to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
        gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
        at 215.4k steps).

        Args:
            model (torch.nn.Module): The model to track using EMA.
            update_after_step (int): Number of steps after which EMA updates begin. Default: 0.
            inv_gamma (float): Inverse factor for EMA warmup. Default: 1.0.
            power (float): Exponential factor for EMA warmup. Default: 2/3.
            min_value (float): Minimum EMA decay rate. Default: 0.0.
            max_value (float): Maximum EMA decay rate. Default: 0.9999.
        """
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.

        Args:
            optimization_step (int): The current optimization step.

        Returns:
            float: The decay factor for the EMA.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        """
        Update the EMA model with weights from the given model.

        Args:
            new_model (torch.nn.Module): The model with the latest weights.
        """
        self.decay = self.get_decay(self.optimization_step)

        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                # Skip unsupported parameter types
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')

                if isinstance(module, BatchNorm):
                    # Skip BatchNorm layers (non-EMA updates for BatchNorm)
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    # Copy parameters that do not require gradients directly
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    # Apply EMA update rule for parameters
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

        self.optimization_step += 1
