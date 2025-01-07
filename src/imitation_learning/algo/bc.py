import torch

from src.imitation_learning.algo.algorithm import AlgorithmBase


class BehaviorCloning(AlgorithmBase):
    """
    Behavior Cloning implementation using supervised learning.

    Args:
        model (torch.nn.Module): Neural network model to learn the policy.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        criterion: Loss function to optimize.
    """
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_on_batch(self, batch) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.model(batch["observations"])
        loss = self.criterion(predictions, batch["actions"])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_on_batch(self, batch) -> float:
        self.model.eval()
        predictions = self.model(batch["observations"])
        loss = self.criterion(predictions, batch["actions"])
        return loss.item()

    def call_policy(self, obs):
        return self.model(obs).detach().numpy()

