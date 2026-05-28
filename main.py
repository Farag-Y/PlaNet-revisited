import torch
import hydra
from omegaconf import DictConfig

from env_wrapper import Env
from experience_replay import ExperienceReplay
from models.rssm import RSSM


def collect_observations(env, seed_episodes, experience_replay: ExperienceReplay) -> ExperienceReplay:
    for _ in range(seed_episodes):
        observation = env.reset()          # tensor (1, 3, 64, 64), already preprocessed
        done = False
        while not done:
            action = env.sample_random_action()           # torch tensor
            observation, reward, done = env.step(action)  # (tensor, float, bool)
            experience_replay.append(observation, reward, action, done)
    env.close()
    return experience_replay


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

    experience_replay = ExperienceReplay(
        cfg.experience_size,
        observation_size=0,                      # 0 = visual env (symbolic not used here)
        image_shape=list(env.observation_size),  # (3, 64, 64) from wrapper
        action_size=env.action_size,
        device=device,
    )

    rssm = RSSM(
        state_size=cfg.state_size,
        hidden_size=cfg.hidden_size,
        belief_size=cfg.belief_size,
        action_size=env.action_size,
        obs_size=cfg.embedding_size,    # encoder output dimension
        non_linearity=cfg.activation_function,
    ).to(device=device)

    collect_observations(env, seed_episodes=cfg.seed_episodes, experience_replay=experience_replay)
    '''
    TODO:
    1. Implement RSSM
    2. Implement Obs Model
    3. Action Model
    4. Reward model
    4. KL Divergence
    5. Losses
    '''
    rssm = RSSM(
        state_size=cfg.state_size,
        hidden_size=cfg.hidden_size,
        belief_size=cfg.belief_size,
        action_size=env.action_size,
        obs_size=cfg.embedding_size,    # encoder output dimension
        non_linearity=cfg.activation_function,
    ).to(device=device)


if __name__ == "__main__":
    main()
