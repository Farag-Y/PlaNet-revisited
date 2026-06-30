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
    decoder_model = ObservationModel(belief_size=cfg.belief_size, state_size=cfg.state_size, embedding_size=cfg.embedding_size).to(device=device)
    reward_model = RewardModel(belief_size=cfg.belief_size, state_size=cfg.state_size, hidden_size=cfg.hidden_size).to(device=device)
    encoder = Encoder(embedding_size=cfg.embedding_size).to(device=device)
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
        bit_depth=cfg.bit_depth,
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
        metrics.episodes.append(metrics.last_episode + 1)
    return experience_replay


def load_checkpoint(cfg, device, rssm, decoder_model, reward_model, encoder, adam_optim) -> Metrics:
    state = torch.load(cfg.models, map_location=device)
    rssm.load_state_dict(state['rssm'])
    decoder_model.load_state_dict(state['decoder_model'])
    reward_model.load_state_dict(state['reward_model'])
    encoder.load_state_dict(state['encoder'])
    adam_optim.load_state_dict(state['adam_optim'])
    return Metrics.load(os.path.join(os.path.dirname(cfg.models), 'metrics.pt'))


def save_checkpoint(cfg, episode, rssm, decoder_model, reward_model, encoder, adam_optim, metrics, results_dir, r2_prefix: str = ""):
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
    if getattr(cfg, 'r2_enabled', False):
        from cloud_storage import upload_checkpoint
        upload_checkpoint(cfg, checkpoint_dir, episode, r2_prefix)


def save_experience_replay(cfg, episode, experience_replay, results_dir, r2_prefix: str = ""):
    replay_path = os.path.join(results_dir, f'experience_replay_{episode}.pt')
    experience_replay.save(replay_path)
    if getattr(cfg, 'r2_enabled', False):
        from cloud_storage import upload_experience_replay
        upload_experience_replay(cfg, replay_path, episode, r2_prefix)


def record_losses(metrics: Metrics, losses: list) -> None:
    kl_vals, obs_vals, rew_vals, os_vals = zip(*losses)
    n = len(losses)
    metrics.kl_loss.append(sum(kl_vals) / n)
    metrics.observation_loss.append(sum(obs_vals) / n)
    metrics.reward_loss.append(sum(rew_vals) / n)
    metrics.overshooting_loss.append(sum(os_vals) / n)


def plot_metrics(metrics: Metrics, results_dir: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
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
        axes[0, 2].plot(metrics.reward_loss)
        axes[0, 2].set_title('Reward Loss')
        axes[0, 2].set_xlabel('Episode')

    if metrics.overshooting_loss:
        axes[1, 0].plot(metrics.overshooting_loss)
        axes[1, 0].set_title('Overshooting Loss')
        axes[1, 0].set_xlabel('Episode')

    if metrics.train_rewards:
        axes[1, 1].plot(metrics.train_rewards)
        axes[1, 1].set_title('Episode Reward')
        axes[1, 1].set_xlabel('Episode')

    if metrics.test_rewards:
        avg_test = [sum(ep) / len(ep) for ep in metrics.test_rewards]
        axes[1, 2].plot(metrics.test_episodes, avg_test)
        axes[1, 2].set_title('Avg Test Reward')
        axes[1, 2].set_xlabel('Episode')
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics.png'))
    plt.close(fig)


def write_video(frames: list, title: str, path: str, fps: int = 30) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        os.path.join(path, f'{title}.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h),
    )
    for frame in frames:
        writer.write(frame[:, :, ::-1])  # RGB → BGR for cv2
    writer.release()