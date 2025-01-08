import numpy as np
import torch

from src.imitation_learning.policies.policy import PolicyBase


class BehaviorCloning(PolicyBase):
    """
    Behavior Cloning implementation using supervised learning.

    Args:
        model (torch.nn.Module): Neural network model to learn the policy.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        criterion: Loss function to optimize.
    """
    def __init__(self, model, optimizer, criterion):
        super().__init__(use_normalizer=True)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_on_batch(self, batch) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        predictions = self.predict_action(batch["observations"])

        loss = self.criterion(predictions, batch["actions"])
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def eval_on_batch(self, batch) -> float:
        self.model.eval()

        predictions = self.predict_action(batch["observations"])

        loss = self.criterion(predictions, batch["actions"])
        return loss.item()

    def predict_action(self, obs) -> torch.Tensor:
        obs = self._normalize_obs(obs)
        return self.model(obs)

    def save(self, path):
        # save model dict and normalizer dict
        model_state_dict = self.model.state_dict()

        d = {
            "model": model_state_dict
        }

        if self.normalizer is not None:
            normalizer_params_dict = self.normalizer.params_dict
            d["normalizer"] = normalizer_params_dict

        torch.save(d, path)

    def load(self, path):
        d = torch.load(path, weights_only=True)
        self.model.load_state_dict(d["model"])
        self.model.eval()

        if self.normalizer is not None:
            self.normalizer.load_state_dict(d["normalizer"])
            self.normalizer.eval()

