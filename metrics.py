from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class Metrics:
    steps:            List[int]         = field(default_factory=list)
    episodes:         List[int]         = field(default_factory=list)
    train_rewards:    List[float]       = field(default_factory=list)
    test_episodes:    List[int]         = field(default_factory=list)
    test_rewards:     List[List[float]] = field(default_factory=list)
    observation_loss: List[tuple]       = field(default_factory=list)
    reward_loss:      List[tuple]       = field(default_factory=list)
    kl_loss:          List[tuple]       = field(default_factory=list)

    @property
    def last_episode(self) -> int:
        return self.episodes[-1] if self.episodes else 0

    @property
    def last_step(self) -> int:
        return self.steps[-1] if self.steps else 0

    def save(self, path: str) -> None:
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> "Metrics":
        return torch.load(path)
