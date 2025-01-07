import collections
import json
from abc import ABC, abstractmethod

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

    - __init__ is optional, but can be used to set up the algorithm.
    """
    def train_loop(
            self,
            train_loader,
            test_loader,
            num_epochs=10,
            num_epochs_logging=1,
            num_episodes_eval=10,
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
            logger (Logger): Logger for training metrics.
            log_wandb (bool): Whether to log metrics to Weights & Biases.
            eval_env: Environment for additional evaluation metrics.
            obs_keys: List of observation keys to extract from the environment observation.
        """
        #initial evaluation
        epoch = 0
        eval_loss, eval_metrics = self.evaluate(
            test_loader, eval_env=eval_env, obs_keys=obs_keys, num_episodes=num_episodes_eval
        )
        if logger is not None:
            logger.info(f"Epoch: {epoch}, Evaluation Loss: {eval_loss:.4f}, {json.dumps(eval_metrics)}")
        if log_wandb:
            wandb.log({"Evaluation Loss": eval_loss, **eval_metrics, "Epoch": epoch})

        for epoch in tqdm(range(1, num_epochs + 1), desc=f"Training (For {num_epochs} Epochs)", unit="epoch", leave=False):
            self.train(train_loader, logger=logger, log_wandb=log_wandb)

            if epoch % num_epochs_logging == 0:
                eval_loss, eval_metrics = self.evaluate(
                    test_loader, eval_env=eval_env, obs_keys=obs_keys, num_episodes=num_episodes_eval
                )
                if logger is not None:
                    logger.info(f"Epoch: {epoch}, Evaluation Loss: {eval_loss:.4f}, {json.dumps(eval_metrics)}")
                if log_wandb:
                    wandb.log({"Evaluation Loss": eval_loss, **eval_metrics, "Epoch": epoch})

    def train(self, train_loader, logger=None, log_wandb=False):
        """
        Trains the policy model using supervised learning on state-action pairs for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            logger (Logger): Logger for training metrics.
            log_wandb (bool): Whether to log training metrics to Weights & Biases.
        """
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training (Single Epoch)", unit="batch", leave=False):
            batch_loss = self.train_on_batch(batch)
            epoch_loss += batch_loss

        avg_loss = epoch_loss / len(train_loader)

        if logger is not None:
            logger.info(f"Training Loss: {avg_loss:.4f}")
        if log_wandb:
            wandb.log({"Training Loss": avg_loss})

    def evaluate(self, eval_loader, eval_env: MujocoEnv = None, obs_keys=None, num_episodes=10):
        """
        Evaluates the policy model on the test dataset and optionally in an environment.

        Args:
            eval_loader (DataLoader): DataLoader for evaluation data.
            eval_env: Environment for additional evaluation metrics.
            obs_keys: List of observation keys to extract from the environment observation.
            num_episodes (int): Number of episodes to evaluate in the environment.

        Returns:
            tuple: Average evaluation loss and a dictionary of evaluation metrics.
        """
        epoch_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluation", unit="batch", leave=False):
                batch_loss = self.eval_on_batch(batch)
                epoch_loss += batch_loss

        avg_loss = epoch_loss / len(eval_loader)

        eval_metrics = {}
        if eval_env is not None and num_episodes > 0:
            eval_env.ignore_done = False
            successes, steps, total = 0, 0, 0
            for _ in tqdm(range(num_episodes), desc="Evaluation in Environment", unit="episode", leave=False):
                obs = eval_env.reset()
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

                    obs, reward, done, info = eval_env.step(predicted_actions)
                    if obs_keys is not None:
                        obs = collections.OrderedDict((k, v) for k, v in obs.items() if k in obs_keys)

                    for k, v in obs.items():
                        obs[k] = torch.from_numpy(v).type(torch.float32)

                    episode_steps += 1
                    if done:
                        break

                if eval_env._check_success():
                    successes += 1

                if eval_env.viewer is not None:
                    eval_env.viewer.close()
                    eval_env.viewer = None

                steps += episode_steps
                total += 1

            eval_metrics = {
                "Success Rate": successes / total,
                "Average Steps": steps / total,
            }

        return avg_loss, eval_metrics


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


