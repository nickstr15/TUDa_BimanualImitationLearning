import collections
import json
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import wandb
from tqdm import tqdm

from robosuite.environments import MujocoEnv

class AlgorithmBase(ABC):
    """
    Base class for all imitation learning algorithms.

    Must implement the following methods:
    - train_on_batch
    - eval_on_batch
    - call_policy
    - save_policy
    - load_policy

    - __init__ should be used to set up the algorithm.
    """
    def __init__(self):
        self.best_eval_loss = np.inf
        self.best_success_rate = -np.inf
        self.best_avg_steps = np.inf

    def train_loop(
            self,
            train_loader,
            test_loader,
            num_epochs=10,
            num_epochs_logging=1,
            num_episodes_eval=10,
            model_out_dir=None,
            logger=None,
            log_wandb=False,
            eval_env: MujocoEnv = None,
            obs_keys=None
    ):
        """
        Training loop for multiple epochs with evaluation and logging.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for evaluation data.
            num_epochs (int): Number of training epochs.
            num_epochs_logging (int): Frequency of logging and evaluation in epochs.
            num_episodes_eval (int): Number of episodes to evaluate in the environment.
            model_out_dir (str): Directory to save the best policy model.
            logger (Logger): Logger for training metrics.
            log_wandb (bool): Whether to log metrics to Weights & Biases.
            eval_env: Environment for additional evaluation metrics.
            obs_keys: List of observation keys to extract from the environment observation.
        """
        #initial evaluation
        epoch = 0
        step = 0
        eval_loss, eval_metrics = self.evaluate(
            test_loader, eval_env=eval_env, obs_keys=obs_keys, num_episodes=num_episodes_eval
        )

        if logger is not None:
            logger.info(f"Epoch: {epoch}, Step: {step}, Evaluation Loss: {eval_loss:.4f}, {json.dumps(eval_metrics)}")
        if log_wandb:
            wandb.log({"Evaluation Loss": eval_loss, **eval_metrics, "Epoch": epoch}, step=step)

        for epoch in tqdm(range(1, num_epochs + 1), desc=f"Training (For {num_epochs} Epochs)", unit="epoch", leave=False):
            step = self.train(train_loader, epoch, step, logger=logger, log_wandb=log_wandb)

            if epoch % num_epochs_logging == 0:
                eval_loss, eval_metrics = self.evaluate(
                    test_loader, eval_env=eval_env, obs_keys=obs_keys, num_episodes=num_episodes_eval
                )
                if logger is not None:
                    logger.info(f"Epoch: {epoch}, Step: {step}, Evaluation Loss: {eval_loss:.4f}, {json.dumps(eval_metrics)}")
                if log_wandb:
                    wandb.log({"Evaluation Loss": eval_loss, **eval_metrics, "Epoch": epoch}, step=step)

                if self.check_is_best_policy(eval_loss, success_rate=eval_metrics.get("Success Rate"), avg_steps=eval_metrics.get("Average Steps")):
                    if model_out_dir is not None:
                        self.save_policy(os.path.join(model_out_dir, f"policy_e{epoch}.pt"))

    def train(self, train_loader, epoch, step, logger=None, log_wandb=False) -> int:
        """
        Trains the policy model using supervised learning on state-action pairs for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.
            step (int): Current training step going into the training loop.
            logger (Logger): Logger for training metrics.
            log_wandb (bool): Whether to log training metrics to Weights & Biases.

        Returns:
            int: Training step after the epoch.
        """
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training (Single Epoch)", unit="batch", leave=False):

            batch_loss = self.train_on_batch(batch)
            epoch_loss += batch_loss

            step += 1

        avg_loss = epoch_loss / len(train_loader)

        if logger is not None:
            logger.info(f"Epoch: {epoch}, Step: {step}, Training Loss: {avg_loss:.4f}")
        if log_wandb:
            wandb.log({"Training Loss": avg_loss, "Epoch": epoch}, step=step)

        return step

    def evaluate(self, eval_loader, eval_env: MujocoEnv = None, obs_keys=None, num_episodes=10):
        """
        Evaluates the policy model on the test dataset and optionally in an environment.

        Args:
            eval_loader (DataLoader): DataLoader for evaluation data.
            eval_env: Environment for additional evaluation metrics.
            obs_keys: List of observation keys to extract from the environment observation.
            num_episodes (int): Number of episodes to evaluate in the environment.

        Returns:
            tuple: Average evaluation loss and a dictionary of evaluation metrics (Success Rate, Average Steps).
        """
        epoch_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluation", unit="batch", leave=False):
                batch_loss = self.eval_on_batch(batch)
                epoch_loss += batch_loss

        avg_loss = epoch_loss / len(eval_loader)

        eval_metrics = {}
        if eval_env is not None and num_episodes > 0:
            eval_metrics = self._run_in_env(eval_env, obs_keys, num_episodes)

        return avg_loss, eval_metrics

    def _run_in_env(self, env: MujocoEnv, obs_keys, num_episodes):
        """
        Runs the policy in the environment for a number of episodes and returns evaluation metrics.

        Args:
            env: Environment to run the policy in.
            obs_keys: List of observation keys to extract from the environment observation.
            num_episodes (int): Number of episodes to run the policy.

        Returns:
            dict: Evaluation metrics (Success Rate, Average Steps).
        """
        env.ignore_done = False
        successes, steps, total = 0, 0, 0
        for _ in tqdm(range(num_episodes), desc="Evaluation in Environment", unit="episode", leave=False):
            obs = env.reset()
            if obs_keys is not None:
                obs = collections.OrderedDict((k, v) for k, v in obs.items() if k in obs_keys)

            for k, v in obs.items():
                obs[k] = torch.from_numpy(v).type(torch.float32)

            done = False
            episode_steps = 0
            while not done:
                predicted_actions = self.call_policy(obs)
                if len(predicted_actions.shape) == 2 and predicted_actions.shape[0] == 1:
                    predicted_actions = predicted_actions.flatten()

                obs, reward, done, info = env.step(predicted_actions)
                if obs_keys is not None:
                    obs = collections.OrderedDict((k, v) for k, v in obs.items() if k in obs_keys)

                for k, v in obs.items():
                    obs[k] = torch.from_numpy(v).type(torch.float32)

                episode_steps += 1
                if done:
                    break

            if env._check_success():
                successes += 1

            if env.viewer is not None:
                env.viewer.close()
                env.viewer = None

            steps += episode_steps
            total += 1

        eval_metrics = {
            "Success Rate": successes / total,
            "Average Steps": steps / total,
        }

        return eval_metrics


    def check_is_best_policy(self, eval_loss, success_rate=None, avg_steps=None) -> bool:
        """
        Checks if the current policy is the best policy based on evaluation loss, success rate, and average steps.

        A policy is considered better than the current best policy if:
        1. Its success_rate is higher.
        2. If the success_rate is equal, its avg_steps is lower.
        3. If the success_rate and avg_steps are equal, its eval_loss is lower.

        Updates the best policy metrics if the current policy is better.

        If success_rate or avg_steps is not provided, the comparison will be based on the available metrics.

        Args:
            eval_loss (float): The evaluation loss of the current policy.
            success_rate (float, optional): The success rate of the current policy. Defaults to None.
            avg_steps (float, optional): The average number of steps for the current policy. Defaults to None.

        Returns:
            bool: True if the current policy is the best policy, False otherwise.
        """

        is_best = False

        if success_rate is None and avg_steps is None:
            is_best = eval_loss < self.best_eval_loss
        elif success_rate is None:
            is_best = avg_steps < self.best_avg_steps or (
                    avg_steps == self.best_avg_steps and eval_loss < self.best_eval_loss)
        elif avg_steps is None:
            is_best = success_rate > self.best_success_rate or (
                    success_rate == self.best_success_rate and eval_loss < self.best_eval_loss)
        else:
            if success_rate > self.best_success_rate:
                is_best = True
            elif success_rate == self.best_success_rate:
                if avg_steps < self.best_avg_steps:
                    is_best = True
                elif avg_steps == self.best_avg_steps:
                    if eval_loss < self.best_eval_loss:
                        is_best = True

        if is_best:
            self.best_eval_loss = eval_loss
            self.best_success_rate = success_rate if success_rate is not None else self.best_success_rate
            self.best_avg_steps = avg_steps if avg_steps is not None else self.best_avg_steps

        return is_best

    @abstractmethod
    def call_policy(self, obs):
        """
        Predicts an action given an observation.

        Args:
            obs (np.ndarray | OrderedDict): Observation.

        Returns:
            np.ndarray: Predicted action.
        """
        raise NotImplementedError

    @abstractmethod
    def train_on_batch(self, batch) -> float:
        """
        Trains the policy model on a batch of data.

        Args:
            batch (dict): Batch of data.

        Returns:
            float: Training loss.
        """
        raise NotImplementedError

    @abstractmethod
    def eval_on_batch(self, batch) -> float:
        """
        Evaluates the policy model on a batch of data.

        Args:
            batch (dict): Batch of data.

        Returns:
            float: Evaluation loss.
        """
        raise NotImplementedError

    @abstractmethod
    def save_policy(self, path):
        """
        Saves the policy model to a file.

        Args:
            path (str): Path to save the model.
        """
        raise NotImplementedError

    @abstractmethod
    def load_policy(self, path):
        """
        Loads the policy model from a file.

        Args:
            path (str): Path to load the model.
        """
        raise NotImplementedError

    def visualize_policy(self, _env, obs_keys, num_episodes):
        """
        Visualizes the policy in the environment for a number of episodes.

        Args:
            _env: Environment to visualize the policy in.
            obs_keys: List of observation keys to extract from the environment observation.
            num_episodes (int): Number of episodes to visualize.
        """
        _env.has_renderer = True
        self._run_in_env(_env, obs_keys, num_episodes)


