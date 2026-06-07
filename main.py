import os
import torch
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from env_wrapper import Env
from experience_replay import ExperienceReplay
from metrics import Metrics
from models.rssm import RSSM,RSSMOutput
from models.observation_model import ObservationModel
from models.encoder import Encoder
from models.reward_model import RewardModel
from models.planner import Planner
from torch import nn, optim
from torch.distributions import Normal
from tqdm import tqdm
from torch.nn import functional as F
from torch.distributions.kl import kl_divergence
from utils import model_wrapper

'''
    TODO:
    9. Sampling data & losses

    6. Connecting everything together properly
    8. Latent overshooting
    9.5 saving checkpoints
    10. Reloading data from a checkpoint
'''

def collect_observations(cfg:DictConfig,device:str,env,metrics:Metrics) -> ExperienceReplay:
    experience_replay = ExperienceReplay(
        cfg.experience_size,
        observation_size=0,                      # 0 = visual env (symbolic not used here)
        image_shape=list(env.observation_size),  # (3, 64, 64) from wrapper
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
            observation=next_obs
        metrics.steps.append(env.t + metrics.last_step)  # env.t already accounts for action_repeat
        metrics.episodes.append(s)
    env.close()
    return experience_replay

def initialize_models(cfg:DictConfig,device:str,env):
    rssm = RSSM(
        state_size=cfg.state_size,
        hidden_size=cfg.hidden_size,
        belief_size=cfg.belief_size,
        action_size=env.action_size,
        obs_size=cfg.embedding_size,    # encoder output dimension
        non_linearity=cfg.activation_function,
    ).to(device=device)
    decoder_model = ObservationModel(belief_size=cfg.belief_size,state_size=cfg.state_size,embedding_size=cfg.embedding_size)
    reward_model = RewardModel(belief_size=cfg.belief_size,state_size=cfg.state_size,hidden_size=cfg.hidden_size)
    encoder = Encoder(embedding_size=cfg.embedding_size)
    parameter_list = list(rssm.parameters()) + list(decoder_model.parameters()) + list(reward_model.parameters()) + list(encoder.parameters())
    adam_optim =  optim.Adam(parameter_list, lr = cfg.learning_rate , eps=cfg.adam_epsilon)#TODO: Implement learning rate scheduler ?
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

    return rssm,decoder_model,reward_model,encoder,adam_optim,planner

def execute_one_run_with_planner(cfg:DictConfig,device:str,env,rssm,encoder,planner,action,observation,belief,state,explore):
    with torch.no_grad():
        encoded = encoder(observation.to(device))
        # Update posterior belief with current observation
        rssm_out = rssm(state, action.unsqueeze(0), belief, encoded.unsqueeze(0))
        belief = rssm_out.det_hidden_states[-1]
        state  = rssm_out.posterior_states[-1]
        action = planner(belief, state)
        # Add exploration noise
        
        if explore:
            action = (action + cfg.action_noise * torch.randn_like(action))
        action=action.clamp(planner.min_action, planner.max_action)
        next_obs, reward, done = env.step(action.cpu())
    return belief,state,action,next_obs,reward,done

def collect_with_planner(cfg:DictConfig,device:str,env,rssm,encoder,planner,experience_replay:ExperienceReplay,metrics:Metrics):
    belief = torch.zeros(1, cfg.belief_size, device=device)
    state  = torch.zeros(1, cfg.state_size,  device=device)
    action=  torch.zeros(1, env.action_size, device=device)
    observation = env.reset()
    done = False
    episode_reward = 0.0
    for _ in tqdm(range(cfg.max_episode_length // cfg.action_repeat)):
        belief,state,action,next_obs,reward,done= execute_one_run_with_planner(cfg,device,env,rssm,encoder,planner,action,observation,belief,state,explore=True)
        experience_replay.append(observation, reward, action.squeeze(0).cpu(), done)
        episode_reward += reward
        observation = next_obs
        if done:
            break

    metrics.steps.append(env.t + metrics.last_step)
    metrics.episodes.append(metrics.last_episode + 1)
    metrics.train_rewards.append(episode_reward)

def execute_runs(runs:int,cfg:DictConfig,rssm,decoder_model,reward_model,encoder,adam_optim,experience_replay,metrics:Metrics,device):
    global_prior = Normal(torch.zeros(cfg.batch_size, cfg.state_size, device=device), torch.ones(cfg.batch_size, cfg.state_size, device=device))  # Global prior N(0, I)
    free_nats = torch.full((1, ), cfg.free_nats, dtype=torch.float32, device=device) 
    losses = []
    for _ in tqdm(range(runs)) :
        init_belief, init_state = torch.zeros(cfg.batch_size, cfg.belief_size, device=device), torch.zeros(cfg.batch_size, cfg.state_size, device=device)
        obs, actions, rewards, nonterminals = experience_replay.sample(cfg.batch_size,  cfg.chunk_size) 
        encoded_obs = model_wrapper(encoder,obs[1:])#TODO: a reshape
        rssm_output:RSSMOutput = rssm(init_state, actions[:-1], init_belief,encoded_obs, nonterminals[:-1])
        predicted_reward = model_wrapper(reward_model, rssm_output.det_hidden_states, rssm_output.posterior_states, trailing_dims=1)

        kl_div = kl_divergence(Normal(rssm_output.posterior_means,rssm_output.posterior_std_devs),Normal(rssm_output.prior_means,rssm_output.prior_std_devs)).sum(dim=-1)

        kl_loss=torch.max(kl_div,free_nats).mean()
        decoded_obs = model_wrapper(decoder_model, rssm_output.det_hidden_states, rssm_output.posterior_states, trailing_dims=1)
        obs_loss = F.mse_loss(decoded_obs,obs[1:],reduction='none').sum((2,3,4)).mean()#Reshape correctly
        reward_loss = F.mse_loss(predicted_reward,rewards[1:],reduction='none').mean()
        ##TODO: calculate latent overshooting
        ##TODO: ramping linear rates ? 
        adam_optim.zero_grad()
        (kl_loss+obs_loss+reward_loss).backward()
        nn.utils.clip_grad_norm_(adam_optim.param_groups[0]['params'], cfg.grad_clip_norm)
        adam_optim.step()
        losses.append([kl_loss.item(),obs_loss.item(),reward_loss.item()])
    return losses

def load_checkpoint(cfg, device, rssm, decoder_model, reward_model, encoder, adam_optim) -> Metrics:
    state = torch.load(cfg.models, map_location=device)
    rssm.load_state_dict(state['rssm'])
    decoder_model.load_state_dict(state['decoder_model'])
    reward_model.load_state_dict(state['reward_model'])
    encoder.load_state_dict(state['encoder'])
    adam_optim.load_state_dict(state['adam_optim'])
    return Metrics.load(os.path.join(os.path.dirname(cfg.models), 'metrics.pt'))

def save_checkpoint(cfg, episode, rssm, decoder_model, reward_model, encoder, adam_optim, experience_replay, metrics):
    checkpoint_dir = os.path.join(hydra.utils.get_original_cwd(), 'results', f'checkpoint_{episode}')
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

def plot_metrics(metrics: Metrics) -> None:
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
    plt.savefig('metrics.png')
    plt.close(fig)

def train(cfg:DictConfig,rssm,decoder_model,reward_model,encoder,adam_optim,planner,experience_replay,metrics:Metrics,device,env):
    for episode in tqdm(range(metrics.last_episode+1, cfg.episodes + 1), total=cfg.episodes, initial=metrics.last_episode):
        losses = execute_runs(cfg.collect_interval,cfg,rssm,decoder_model,reward_model,encoder,adam_optim,experience_replay,metrics,device)
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

    rssm,decoder_model,reward_model,encoder,adam_optim,planner = initialize_models(cfg,device,env)
    metrics = load_checkpoint(cfg,device,rssm,decoder_model,reward_model,encoder,adam_optim) if cfg.models else Metrics()
    experience_replay = (ExperienceReplay.load(cfg.experience_replay_path, device)
                         if cfg.experience_replay_path
                         else collect_observations(cfg,device,env,metrics))

    train(cfg,rssm,decoder_model,reward_model,encoder,adam_optim,planner,experience_replay,metrics,device,env)


if __name__ == "__main__":
    main()
