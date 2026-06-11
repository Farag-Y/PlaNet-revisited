import numpy as np
from typing import List
import torch

from env_wrapper import preprocess_observation_, postprocess_observation


class ExperienceReplay():
    def __init__(self, experience_size, observation_size, image_shape, action_size, bit_depth, device):
        self.device = device
        self.bit_depth = bit_depth
        #TODO: Observation size will only be used in symbolic envs
        # Observations are stored quantised as uint8 (4x less memory than float32);
        # dequantisation noise is re-sampled on every batch in _get_batch
        self.observations = np.empty((experience_size, image_shape[0], image_shape[1], image_shape[2]), dtype=np.uint8)
        self.actions = np.empty((experience_size, action_size), dtype=np.float32)
        self.rewards = np.empty((experience_size,), dtype=np.float32)
        self.non_terminals = np.empty((experience_size, 1), dtype=np.float32)
        self.idx, self.steps, self.episodes = 0, 0, 0
        self.full = False
        self.size = experience_size

    def append(self, observation, reward, action, done):
        self.observations[self.idx] = postprocess_observation(observation.numpy(), self.bit_depth)
        self.rewards[self.idx] = reward
        self.actions[self.idx] = action
        self.non_terminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1
        self.episodes += (1 if done else 0)

    def _get_indexes(self, batch_size: int, batch_length: int) -> List[int]:
        # Guard: buffer must have enough transitions to form a sequence
        available = self.size if self.full else self.idx
        if available < batch_length:
            raise ValueError(f"Not enough transitions ({available}) to sample batch_length={batch_length}")
        batches = []
        for i in range(batch_size):
            valid_idx = False
            while not valid_idx:
                max_size = self.idx - batch_length + 1 if not self.full else self.size
                idx = np.random.randint(0, max_size)
                idxs = np.arange(idx, idx + batch_length) % self.size
                # Exclude sequences that overlap the write pointer at any position
                valid_idx = self.idx not in idxs
            batches.append(idxs)
        return batches

    def _get_batch(self, idxs, batch_size, batch_length):
        # Stack list of per-sample index arrays into shape (batch_size, batch_length)
        stacked = np.stack(idxs, axis=0)
        obs = torch.as_tensor(self.observations[stacked].astype(np.float32))
        preprocess_observation_(obs, self.bit_depth)
        obs = obs.to(self.device).transpose(0, 1)
        acts = torch.as_tensor(self.actions[stacked]).to(self.device).transpose(0, 1)
        rewards = torch.as_tensor(self.rewards[stacked]).to(self.device).transpose(0, 1)
        non_terminals = torch.as_tensor(self.non_terminals[stacked]).to(self.device).transpose(0, 1)
        return obs, acts, rewards, non_terminals

    def sample(self, batch_size, batch_length):
        batch_idxs = self._get_indexes(batch_size, batch_length)
        batches = self._get_batch(batch_idxs, batch_size, batch_length)
        return batches

    def save(self, path: str) -> None:
        torch.save({
            'observations': self.observations, 'actions': self.actions,
            'rewards': self.rewards, 'non_terminals': self.non_terminals,
            'idx': self.idx, 'steps': self.steps,
            'episodes': self.episodes, 'full': self.full, 'size': self.size,
            'bit_depth': self.bit_depth,
        }, path)

    @classmethod
    def load(cls, path: str, device: str) -> 'ExperienceReplay':
        import numpy.core.multiarray
        import numpy.dtypes
        safe = [
            numpy.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            numpy.dtypes.UInt8DType,
            numpy.dtypes.Float32DType,
        ]
        with torch.serialization.safe_globals(safe):
            data = torch.load(path, map_location='cpu', weights_only=True)
        instance = cls.__new__(cls)
        instance.device        = device
        instance.bit_depth     = data.get('bit_depth', 5)
        instance.observations  = data['observations']
        # Buffers saved before the switch to uint8 storage hold preprocessed float32 frames
        if instance.observations.dtype != np.uint8:
            instance.observations = postprocess_observation(instance.observations, instance.bit_depth)
        instance.actions       = data['actions']
        instance.rewards       = data['rewards']
        instance.non_terminals = data['non_terminals']
        instance.idx           = data['idx']
        instance.steps         = data['steps']
        instance.episodes      = data['episodes']
        instance.full          = data['full']
        instance.size          = data['size']
        return instance
