import os
import torch
import hydra
from datetime import datetime
from omegaconf import DictConfig

from env_wrapper import Env
from experience_replay import ExperienceReplay
from metrics import Metrics
from models.rssm import RSSMOutput
from torch import nn
from torch.distributions import Normal
from tqdm import tqdm
from torch.nn import functional as F
from torch.distributions.kl import kl_divergence
from utils import (model_wrapper, initialize_models, collect_observations,
                   load_checkpoint, save_checkpoint, record_losses, plot_metrics)
'''
    TODO:
    6. Connecting everything together properly
    8. Latent overshooting
    9. Sampling data & losses
    10. Reloading data from a checkpoint
'''
def execute_one_run_with_planner(cfg:DictConfig,device:str,env,rssm,encoder,planner,action,observation,belief,state,explore):
    with torch.no_grad():
        encoded = encoder(observation.to(device))
        # Update posterior belief with current observation
        rssm_out = rssm(state, action.unsqueeze(0), belief, encoded.unsqueeze(0))
        belief = rssm_out.det_hidden_states[-1]
        state  = rssm_out.posterior_states[-1]
        action = planner(belief, state)
        if explore:
            action = (action + cfg.action_noise * torch.randn_like(action))
        action = action.clamp(planner.min_action, planner.max_action)
        next_obs, reward, done = env.step(action.cpu())
    return belief, state, action, next_obs, reward, done

def collect_with_planner(cfg:DictConfig,device:str,env,rssm,encoder,planner,experience_replay:ExperienceReplay,metrics:Metrics):
    belief = torch.zeros(1, cfg.belief_size, device=device)
    state  = torch.zeros(1, cfg.state_size,  device=device)
    action = torch.zeros(1, env.action_size, device=device)
    observation = env.reset()
    episode_reward = 0.0
    for _ in tqdm(range(cfg.max_episode_length // cfg.action_repeat)):
        belief, state, action, next_obs, reward, done = execute_one_run_with_planner(
            cfg, device, env, rssm, encoder, planner, action, observation, belief, state, explore=True)
        experience_replay.append(observation, reward, action.squeeze(0).cpu(), done)
        episode_reward += reward
        observation = next_obs
        if done:
            break
    metrics.steps.append(env.t + metrics.last_step)
    metrics.episodes.append(metrics.last_episode + 1)
    metrics.train_rewards.append(episode_reward)

def calculate_latent_overshooting(cfg, rssm, reward_model,
                                   actions,           # [chunk_size-1, B, action_size]  — full_actions[:-1]
                                   nonterminals,      # [chunk_size-1, B, 1]            — full_nonterminals[:-1]
                                   posterior_states,  # [chunk_size-1, B, state_size]
                                   posterior_means,   # [chunk_size-1, B, state_size]
                                   posterior_std_devs,# [chunk_size-1, B, state_size]
                                   rssm_beliefs,      # [chunk_size-1, B, belief_size]
                                   rewards,           # [chunk_size,   B]               — full rewards
                                   free_nats, device):
    if cfg.overshooting_kl_beta == 0:
        return torch.tensor(0.0, device=device)

    chunk_size = actions.shape[0] + 1
    B, state_size = posterior_states.shape[1], posterior_states.shape[2]

    beliefs_all     = rssm_beliefs        # [chunk_size-1, B, belief_size]
    post_states_all = posterior_states    # [chunk_size-1, B, state_size]
    post_means_all  = posterior_means     # [chunk_size-1, B, state_size]
    post_stds_all   = posterior_std_devs  # [chunk_size-1, B, state_size]

    overshooting_vars = []
    for t in range(1, chunk_size - 1):
        d = min(t + cfg.overshooting_distance, chunk_size - 1)
        t_ = t - 1                                      # index offset for belief/state (which include t=0)
        pad_len = cfg.overshooting_distance - (d - t)   # right-pad shorter sequences to uniform length
        seq_pad = (0, 0, 0, 0, 0, pad_len)              # pads dim-0 (time) of 3-D tensors

        overshooting_vars.append((
            F.pad(actions[t:d],                              seq_pad),           # [0] actions
            F.pad(nonterminals[t:d],                         seq_pad),           # [1] nonterminals
            F.pad(rewards[t:d],                              seq_pad[2:]),       # [2] rewards (2-D)
            beliefs_all[t_],                                                     # [3] starting belief
            post_states_all[t_].detach(),                                        # [4] starting state
            F.pad(post_means_all[t:d].detach(),              seq_pad),           # [5] target posterior means
            F.pad(post_stds_all[t:d].detach(),               seq_pad, value=1), # [6] target posterior stds (pad with 1, not 0)
            F.pad(torch.ones(d - t, B, state_size, device=device), seq_pad),    # [7] validity mask
        ))

    overshooting_vars = tuple(zip(*overshooting_vars))

    # Single batched RSSM call across all overshooting sequences
    prior_out = rssm(
        torch.cat(overshooting_vars[4], dim=0),   # prev_state:   [num_seqs*B, state_size]
        torch.cat(overshooting_vars[0], dim=1),   # actions:      [D, num_seqs*B, action_size]
        torch.cat(overshooting_vars[3], dim=0),   # prev_belief:  [num_seqs*B, belief_size]
        None,
        torch.cat(overshooting_vars[1], dim=1),   # nonterminals: [D, num_seqs*B, 1]
    )

    seq_mask    = torch.cat(overshooting_vars[7], dim=1)   # [D, num_seqs*B, state_size]
    target_means = torch.cat(overshooting_vars[5], dim=1)  # [D, num_seqs*B, state_size]
    target_stds  = torch.cat(overshooting_vars[6], dim=1)  # [D, num_seqs*B, state_size]

    # KL loss: mask out padded steps, then apply free-nats per (step, batch) cell
    kl = (kl_divergence(
        Normal(target_means, target_stds),
        Normal(prior_out.prior_means, prior_out.prior_std_devs),
    ) * seq_mask).sum(dim=2)                                # [D, num_seqs*B]

    kl_loss = (1 / cfg.overshooting_distance) * cfg.overshooting_kl_beta * \
        torch.max(kl, free_nats).mean(dim=(0, 1)) * (chunk_size - 1)

    total = kl_loss

    if cfg.overshooting_reward_scale != 0:
        target_rewards = torch.cat(overshooting_vars[2], dim=1)        # [D, num_seqs*B]
        pred_rewards   = model_wrapper(reward_model, prior_out.det_hidden_states, prior_out.prior_states, trailing_dims=1)
        reward_mask    = seq_mask[:, :, 0]                             # [D, num_seqs*B]
        reward_loss    = (1 / cfg.overshooting_distance) * cfg.overshooting_reward_scale * \
            F.mse_loss(pred_rewards * reward_mask, target_rewards, reduction='none').mean(dim=(0, 1)) * (chunk_size - 1)
        total = total + reward_loss

    return total

def train_world_model(runs:int,cfg:DictConfig,rssm,decoder_model,reward_model,encoder,adam_optim,experience_replay,metrics:Metrics,device):
    free_nats = torch.full((1,), cfg.free_nats, dtype=torch.float32, device=device)
    losses = []
    for _ in tqdm(range(runs)):
        init_belief = torch.zeros(cfg.batch_size, cfg.belief_size, device=device)
        init_state  = torch.zeros(cfg.batch_size, cfg.state_size,  device=device)
        obs, actions, rewards, nonterminals = experience_replay.sample(cfg.batch_size, cfg.chunk_size)
        encoded_obs = model_wrapper(encoder, obs[1:])
        rssm_output: RSSMOutput = rssm(init_state, actions[:-1], init_belief, encoded_obs, nonterminals[:-1])
        predicted_reward = model_wrapper(reward_model, rssm_output.det_hidden_states, rssm_output.posterior_states, trailing_dims=1)

        kl_div = kl_divergence(
            Normal(rssm_output.posterior_means, rssm_output.posterior_std_devs),
            Normal(rssm_output.prior_means,     rssm_output.prior_std_devs),
        ).sum(dim=-1)
        kl_loss = torch.max(kl_div, free_nats).mean()

        decoded_obs = model_wrapper(decoder_model, rssm_output.det_hidden_states, rssm_output.posterior_states, trailing_dims=1)
        obs_loss    = F.mse_loss(decoded_obs, obs[1:], reduction='none').sum((2, 3, 4)).mean()
        reward_loss = F.mse_loss(predicted_reward, rewards[:-1], reduction='none').mean()
        overshooting_loss = calculate_latent_overshooting(
            cfg, rssm, reward_model,
            actions[:-1], nonterminals[:-1],
            rssm_output.posterior_states,
            rssm_output.posterior_means,
            rssm_output.posterior_std_devs,
            rssm_output.det_hidden_states,
            rewards,        # full rewards tensor — function slices [t:d] internally
            free_nats,
            device,
        )
        adam_optim.zero_grad()
        (kl_loss + obs_loss + reward_loss + overshooting_loss).backward()
        nn.utils.clip_grad_norm_(adam_optim.param_groups[0]['params'], cfg.grad_clip_norm)
        adam_optim.step()
        losses.append([kl_loss.item(), obs_loss.item(), reward_loss.item(), overshooting_loss.item()])
    return losses

def train(cfg:DictConfig,rssm,decoder_model,reward_model,encoder,adam_optim,planner,experience_replay,metrics:Metrics,device,env,results_dir:str):
    for episode in tqdm(range(metrics.last_episode+1, cfg.episodes + 1), total=cfg.episodes, initial=metrics.last_episode):
        losses = train_world_model(cfg.collect_interval,cfg,rssm,decoder_model,reward_model,encoder,adam_optim,experience_replay,metrics,device)
        record_losses(metrics, losses)
        collect_with_planner(cfg,device,env,rssm,encoder,planner,experience_replay,metrics)
        plot_metrics(metrics, results_dir)
        if episode % cfg.checkpoint_interval == 0:
            save_checkpoint(cfg, episode, rssm, decoder_model, reward_model, encoder, adam_optim, experience_replay, metrics, results_dir)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = "cuda" if not cfg.disable_cuda and torch.cuda.is_available() else "cpu"
    env = Env(
        cfg.env,
        seed=cfg.seed,
        max_episode_length=cfg.max_episode_length,
        action_repeat=cfg.action_repeat,
        bit_depth=cfg.bit_depth,
    )

    rssm, decoder_model, reward_model, encoder, adam_optim, planner = initialize_models(cfg, device, env)
    metrics = load_checkpoint(cfg, device, rssm, decoder_model, reward_model, encoder, adam_optim) if cfg.models else Metrics()
    experience_replay = (ExperienceReplay.load(cfg.experience_replay_path, device)
                         if cfg.experience_replay_path
                         else collect_observations(cfg, device, env, metrics))

    results_dir = os.path.join(hydra.utils.get_original_cwd(), 'results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(results_dir, exist_ok=True)

    train(cfg, rssm, decoder_model, reward_model, encoder, adam_optim, planner, experience_replay, metrics, device, env, results_dir)


if __name__ == "__main__":
    main()
