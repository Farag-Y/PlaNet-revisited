import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from torch import optim
from experience_replay import ExperienceReplay
from metrics import Metrics
from models.rssm import RSSM
from models.observation_model import ObservationModel
from models.encoder import Encoder
from models.reward_model import RewardModel
from models.planner import Planner

def model_wrapper(model, *inputs, trailing_dims=3):
    leading = inputs[0].shape[:-trailing_dims]
    reshaped = [obs.reshape(-1, *obs.shape[-trailing_dims:]) for obs in inputs]
    out = model(*reshaped)
    return out.view(*leading, *out.shape[1:])


def preprocess_frame(frame, size=64):
    frame = cv2.resize(frame, (size, size))        # (size, size, 3)
    frame = np.transpose(frame, (2, 0, 1))         # (3, size, size)
    return frame.astype(np.float32) / 255.0        # normalize to [0, 1]


def initialize_models(cfg: DictConfig, device: str, env):
    rssm = RSSM(
        state_size=cfg.state_size,
        hidden_size=cfg.hidden_size,
        belief_size=cfg.belief_size,
        action_size=env.action_size,
        obs_size=cfg.embedding_size,
        non_linearity=cfg.activation_function,
    ).to(device=device)
    decoder_model = ObservationModel(belief_size=cfg.belief_size, state_size=cfg.state_size, embedding_size=cfg.embedding_size)
    reward_model = RewardModel(belief_size=cfg.belief_size, state_size=cfg.state_size, hidden_size=cfg.hidden_size)
    encoder = Encoder(embedding_size=cfg.embedding_size)
    parameter_list = list(rssm.parameters()) + list(decoder_model.parameters()) + list(reward_model.parameters()) + list(encoder.parameters())
    adam_optim = optim.Adam(parameter_list, lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    min_action, max_action = env.action_range
    planner = Planner(
        action_size=env.action_size,
        planning_horizon=cfg.planning_horizon,
        optimisation_iters=cfg.optimisation_iters,
        candidates=cfg.candidates,
        top_candidates=cfg.top_candidates,
        transition_model=rssm,
        reward_model=reward_model,
        min_action=min_action,
        max_action=max_action,
    ).to(device=device)
    return rssm, decoder_model, reward_model, encoder, adam_optim, planner


def collect_observations(cfg: DictConfig, device: str, env, metrics: Metrics) -> ExperienceReplay:
    experience_replay = ExperienceReplay(
        cfg.experience_size,
        observation_size=0,
        image_shape=list(env.observation_size),
        action_size=env.action_size,
        device=device,
    )
    for s in range(1, cfg.seed_episodes + 1):
        observation = env.reset()
        done = False
        while not done:
            action = env.sample_random_action()
            next_obs, reward, done = env.step(action)
            experience_replay.append(observation, reward, action, done)
            observation = next_obs
        metrics.steps.append(env.t + metrics.last_step)
        metrics.episodes.append(s)
    env.close()
    return experience_replay


def load_checkpoint(cfg, device, rssm, decoder_model, reward_model, encoder, adam_optim) -> Metrics:
    state = torch.load(cfg.models, map_location=device)
    rssm.load_state_dict(state['rssm'])
    decoder_model.load_state_dict(state['decoder_model'])
    reward_model.load_state_dict(state['reward_model'])
    encoder.load_state_dict(state['encoder'])
    adam_optim.load_state_dict(state['adam_optim'])
    return Metrics.load(os.path.join(os.path.dirname(cfg.models), 'metrics.pt'))


def save_checkpoint(cfg, episode, rssm, decoder_model, reward_model, encoder, adam_optim, experience_replay, metrics, results_dir):
    checkpoint_dir = os.path.join(results_dir, f'checkpoint_{episode}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'rssm':          rssm.state_dict(),
        'decoder_model': decoder_model.state_dict(),
        'reward_model':  reward_model.state_dict(),
        'encoder':       encoder.state_dict(),
        'adam_optim':    adam_optim.state_dict(),
    }, os.path.join(checkpoint_dir, 'models.pt'))
    metrics.save(os.path.join(checkpoint_dir, 'metrics.pt'))
    if cfg.checkpoint_experience:
        experience_replay.save(os.path.join(checkpoint_dir, 'experience_replay.pt'))


def record_losses(metrics: Metrics, losses: list) -> None:
    kl_vals, obs_vals, rew_vals = zip(*losses)
    n = len(losses)
    metrics.kl_loss.append(sum(kl_vals) / n)
    metrics.observation_loss.append(sum(obs_vals) / n)
    metrics.reward_loss.append(sum(rew_vals) / n)


def plot_metrics(metrics: Metrics, results_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Training Metrics — Episode {metrics.last_episode}')

    if metrics.kl_loss:
        axes[0, 0].plot(metrics.kl_loss)
        axes[0, 0].set_title('KL Loss')
        axes[0, 0].set_xlabel('Episode')

    if metrics.observation_loss:
        axes[0, 1].plot(metrics.observation_loss)
        axes[0, 1].set_title('Observation Loss')
        axes[0, 1].set_xlabel('Episode')

    if metrics.reward_loss:
        axes[1, 0].plot(metrics.reward_loss)
        axes[1, 0].set_title('Reward Loss')
        axes[1, 0].set_xlabel('Episode')

    if metrics.train_rewards:
        axes[1, 1].plot(metrics.train_rewards)
        axes[1, 1].set_title('Episode Reward')
        axes[1, 1].set_xlabel('Episode')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics.png'))
    plt.close(fig)