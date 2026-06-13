import abc
import cv2
import numpy as np
import torch


def preprocess_observation_(observation, bit_depth):
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))


def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


class BaseEnv(abc.ABC):
    def _images_to_observation(self, images, bit_depth):
        images = torch.tensor(cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)
        preprocess_observation_(images, bit_depth)
        return images.unsqueeze(dim=0)

    @abc.abstractmethod
    def reset(self): ...

    @abc.abstractmethod
    def step(self, action): ...

    @abc.abstractmethod
    def render(self): ...

    @abc.abstractmethod
    def close(self): ...

    @property
    @abc.abstractmethod
    def observation_size(self): ...

    @property
    @abc.abstractmethod
    def action_size(self): ...

    @property
    @abc.abstractmethod
    def action_range(self): ...

    @abc.abstractmethod
    def sample_random_action(self): ...


class GymEnv(BaseEnv):
    def __init__(self, env, seed, max_episode_length, action_repeat, bit_depth):
        import gymnasium as gym
        gym.logger.min_level = gym.logger.ERROR
        self._env = gym.make(env, render_mode='rgb_array')
        self._seed = seed
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0
        self._env.reset(seed=self._seed)
        self._seed = None
        return self._images_to_observation(self._env.render(), self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for _ in range(self.action_repeat):
            _, reward_k, terminated, truncated, _ = self._env.step(action)
            reward += reward_k
            self.t += 1
            done = terminated or truncated or self.t == self.max_episode_length
            if done:
                break
        observation = self._images_to_observation(self._env.render(), self.bit_depth)
        return observation, float(reward), done

    def render(self):
        frame = self._env.render()
        if frame is not None:
            cv2.imshow('screen', frame[:, :, ::-1])
            cv2.waitKey(1)

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    @property
    def action_range(self):
        return float(self._env.action_space.low[0]), float(self._env.action_space.high[0])

    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())


class DMControlEnv(BaseEnv):
    def __init__(self, env, seed, max_episode_length, action_repeat, bit_depth):
        from dm_control import suite
        domain, *task_parts = env.split('-')
        task = '_'.join(task_parts)
        self._env = suite.load(domain, task, task_kwargs={'random': seed})
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0
        self._env.reset()
        return self._images_to_observation(self._env.physics.render(64, 64, camera_id=0), self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for _ in range(self.action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            self.t += 1
            done = time_step.last() or self.t == self.max_episode_length
            if done:
                break
        observation = self._images_to_observation(self._env.physics.render(64, 64, camera_id=0), self.bit_depth)
        return observation, float(reward), done

    def render(self):
        frame = self._env.physics.render(240, 320, camera_id=0)
        if frame is not None:
            cv2.imshow('screen', frame[:, :, ::-1])
            cv2.waitKey(1)

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]

    @property
    def action_range(self):
        spec = self._env.action_spec()
        return float(spec.minimum[0]), float(spec.maximum[0])

    def sample_random_action(self):
        spec = self._env.action_spec()
        action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
        return torch.from_numpy(action.astype(np.float32))


# pip install gymnasium[classic-control]
GYM_ENVS_CLASSIC = [
    'Pendulum-v1',
    'MountainCarContinuous-v0',
]

# pip install gymnasium[box2d]  (also requires: pip install swig)
GYM_ENVS_BOX2D = [
    'BipedalWalker-v3',
    'BipedalWalkerHardcore-v3',
    'CarRacing-v3',
]

# pip install gymnasium[mujoco]
GYM_ENVS_MUJOCO = [
    'Ant-v5',
    'HalfCheetah-v5',
    'Hopper-v5',
    'Humanoid-v5',
    'HumanoidStandup-v5',
    'InvertedDoublePendulum-v5',
    'InvertedPendulum-v5',
    'Pusher-v5',
    'Reacher-v5',
    'Swimmer-v5',
    'Walker2d-v5',
]

GYM_ENVS = GYM_ENVS_CLASSIC + GYM_ENVS_BOX2D + GYM_ENVS_MUJOCO

DMCONTROL_ENVS = [
    'cartpole-balance', 'cartpole-balance-sparse',
    'cartpole-swingup', 'cartpole-swingup-sparse',
    'finger-spin', 'finger-turn-easy', 'finger-turn-hard',
    'cheetah-run',
    'reacher-easy', 'reacher-hard',
    'cup-catch',
    'walker-stand', 'walker-walk', 'walker-run',
    'hopper-stand', 'hopper-hop',
    'humanoid-stand', 'humanoid-walk', 'humanoid-run',
]


def Env(env, seed, max_episode_length, action_repeat, bit_depth):
    if env in GYM_ENVS:
        return GymEnv(env, seed, max_episode_length, action_repeat, bit_depth)
    elif env in DMCONTROL_ENVS:
        return DMControlEnv(env, seed, max_episode_length, action_repeat, bit_depth)
    else:
        raise ValueError(f"Unknown environment: '{env}'. Must be one of GYM_ENVS or DMCONTROL_ENVS.")
