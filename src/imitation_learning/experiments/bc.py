import numpy as np
import torch

from src.imitation_learning.networks.network import NetworkBase
from src.imitation_learning.policies.policy import PolicyBase
from src.imitation_learning.policies.bc import BehaviorCloning
from src.imitation_learning.experiments.experiment import ExperimentBase
from src.imitation_learning.core.helpers import get_loss_fn, get_optimizer_cls, get_network_cls


class BehaviorCloningExperiment(ExperimentBase):
    """
    Experiment class for behavior cloning.
    """
    def _setup_policy(self) -> PolicyBase:
        """
        Set up the algorithm.
        """
        assert self._config["policy"]["name"] == "BC", "Wrong algorithm name in config file."

        # Set up the model
        model = self._setup_model(self._config["policy"]["params"]["model"])
        optimizer = self._setup_optimizer(self._config["policy"]["params"]["optimizer"], model)
        criterion = self._setup_criterion(self._config["policy"]["params"]["loss"])

        return BehaviorCloning(model, optimizer, criterion)

    def _setup_model(self, model_config: dict):
        """
        Set up the model.
        """
        model : NetworkBase = get_network_cls(model_config["name"])
        input_dim = np.sum([np.prod(s) for s in self._input_sizes.values()])
        output_dim = np.prod(self._output_size)
        return model(
            input_dim=input_dim,
            output_dim=output_dim,
            **model_config["params"]
        )


    @staticmethod
    def _setup_optimizer(optimizer_config: dict, model: torch.nn.Module):
        """
        Set up the optimizer.
        """
        optimizer = get_optimizer_cls(optimizer_config["name"])
        return optimizer(model.parameters(), **optimizer_config["params"])

    @staticmethod
    def _setup_criterion(criterion_config: list):
        """
        Set up the criterion.
        """
        def composed_loss_fn(x, x_hat):
            x = x.reshape(x.size(0), -1)
            x_hat = x_hat.reshape(x_hat.size(0), -1)
            loss = 0
            for loss_cfg in criterion_config:
                loss_fn = get_loss_fn(loss_cfg["name"])()
                w = loss_cfg["weight"]
                v = loss_fn(x, x_hat)
                loss += w*v
            return loss

        return composed_loss_fn


if __name__ == "__main__":
    exp = BehaviorCloningExperiment(config_path="bc_two_arm_pick_place.yaml")
    exp.run()

    # exp.load_and_visualize_policy(
    #    "bc_mlp_two_arm_pick_place/2025-01-08_11-35-18/best_policy_e300.pt",
    #    num_episodes=5
    # )




