import torch
import hydra
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
        reward_loss = F.mse_loss(predicted_reward, rewards[1:], reduction='none').mean()
        ##TODO: calculate latent overshooting
        adam_optim.zero_grad()
        (kl_loss + obs_loss + reward_loss).backward()
        nn.utils.clip_grad_norm_(adam_optim.param_groups[0]['params'], cfg.grad_clip_norm)
        adam_optim.step()
        losses.append([kl_loss.item(), obs_loss.item(), reward_loss.item()])
    return losses

def train(cfg:DictConfig,rssm,decoder_model,reward_model,encoder,adam_optim,planner,experience_replay,metrics:Metrics,device,env):
    for episode in tqdm(range(metrics.last_episode+1, cfg.episodes + 1), total=cfg.episodes, initial=metrics.last_episode):
        losses = train_world_model(cfg.collect_interval,cfg,rssm,decoder_model,reward_model,encoder,adam_optim,experience_replay,metrics,device)
        record_losses(metrics, losses)
        collect_with_planner(cfg,device,env,rssm,encoder,planner,experience_replay,metrics)
        plot_metrics(metrics)
        if episode % cfg.checkpoint_interval == 0:
            save_checkpoint(cfg, episode, rssm, decoder_model, reward_model, encoder, adam_optim, experience_replay, metrics)

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

    train(cfg, rssm, decoder_model, reward_model, encoder, adam_optim, planner, experience_replay, metrics, device, env)


if __name__ == "__main__":
    main()
