import torch
import hydra
from omegaconf import DictConfig

from env_wrapper import Env
from experience_replay import ExperienceReplay
from metrics import Metrics
from models.rssm import RSSM,RSSMOutput
from models.observation_model import ObservationModel
from models.encoder import Encoder
from models.reward_model import RewardModel
from torch import optim
from torch.distributions import Normal
from tqdm import tqdm
from torch.nn import functional as F
from torch.distributions.kl import kl_divergence
from utils import model_wrapper

'''
    TODO:
    X.Planner 
    6. Connecting everything together properly
    7. Second data collection round in the for loop
    8. Latent overshooting
    9. Sampling data & losses
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
    parameter_list = list(decoder_model.parameters()) + list(reward_model.parameters()) + list(encoder.parameters())
    adam_optim =  optim.Adam(parameter_list, lr = cfg.learning_rate , eps=cfg.adam_epsilon)#TODO: Implement learning rate scheduler ?
    #TODO: Still :  planner
    return rssm,decoder_model,reward_model,encoder,adam_optim

def execute_runs(runs:int,cfg:DictConfig,rssm,decoder_model,reward_model,encoder,adam_optim,experience_replay,metrics:Metrics,device):
    global_prior = Normal(torch.zeros(cfg.batch_size, cfg.state_size, device=device), torch.ones(cfg.batch_size, cfg.state_size, device=device))  # Global prior N(0, I)
    free_nats = torch.full((1, ), cfg.free_nats, dtype=torch.float32, device=device) 
    losses = []
    for _ in tqdm(range(runs)) :
        init_belief, init_state = torch.zeros(cfg.batch_size, cfg.belief_size, device=device), torch.zeros(cfg.batch_size, cfg.state_size, device=device)
        obs, actions, rewards, nonterminals = experience_replay.sample(cfg.batch_size,  cfg.chunk_size) 
        encoded_obs = model_wrapper(encoder,obs[1:])#TODO: a reshape
        rssm_output:RSSMOutput = rssm(init_state, actions[:-1], init_belief,encoded_obs, nonterminals[:-1])
        predicted_reward = model_wrapper(reward_model,(rssm_output.det_hidden_states,rssm_output.posterior_states))

        kl_div = kl_divergence(Normal(rssm_output.posterior_means,rssm_output.posterior_std_devs),Normal(rssm_output.prior_means,rssm_output.prior_std_devs))#TODO: reshape
        
        kl_loss=torch.max(kl_div,free_nats)#TODO:Mean or smth ?
        decoded_obs = decoder_model((rssm_output.det_hidden_states,rssm_output.posterior_states))#TODO: any reshaping ?
        obs_loss = F.mse_loss(rssm_output.decoded_obs,obs[1:])#Reshape correctly
        reward_loss = F.mse_loss(rssm_output,rewards)
        ##TODO: calculate latent overshooting
        ##TODO: ramping linear rates ? 
        adam_optim.zero_grad()
        (kl_loss+obs_loss+reward_loss).backward()
        adam_optim.step()
        losses.append([kl_loss.item(),obs_loss.item(),reward_loss.item()])
    return losses
def train(cfg:DictConfig,rssm,decoder_model,reward_model,encoder,adam_optim,experience_replay,metrics:Metrics,device):
    #TODO: Constants, need to be correctly identified
    # Allowed deviation in KL divergence
    for episode in tqdm(range(metrics.last_episode+1, cfg.episodes + 1), total=cfg.episodes, initial=metrics.last_episode):
        execute_runs(cfg.collect_interval,cfg,rssm,decoder_model,reward_model,encoder,adam_optim,experience_replay,metrics,device )
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

    metrics = Metrics()
    experience_replay=collect_observations(cfg,device,env,metrics)
    rssm,decoder_model,reward_model,encoder,adam_optim= initialize_models(cfg,device,env)
    train(cfg,rssm,decoder_model,reward_model,encoder,adam_optim,experience_replay,metrics,device)


if __name__ == "__main__":
    main()
