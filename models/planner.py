from math import inf
import torch

from torch import nn

class Planner(nn.Module):
  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, min_action=-inf, max_action=inf):
        super().__init__()
        self.transition_model, self.reward_model = transition_model, reward_model
        self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates

  def forward(self, belief, state):
    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)

    for _ in range(self.optimisation_iters):
      # Sample [H, B, candidates, action_size] then clamp to valid range
      actions = (action_mean + action_std_dev * torch.randn(
        self.planning_horizon, B, self.candidates, self.action_size, device=belief.device
      )).clamp(self.min_action, self.max_action)
      # Flatten batch and candidates for RSSM rollout: [H, B*C, action_size]
      actions_flat = actions.view(self.planning_horizon, B * self.candidates, self.action_size)

      with torch.no_grad():
        rssm_output = self.transition_model(state, actions_flat, belief)
        beliefs_seq = rssm_output.det_hidden_states  # [H, B*C, H_dim]
        states_seq  = rssm_output.prior_states        # [H, B*C, Z_dim]

      # Predict reward at each step, sum over horizon -> [B*C]
      rewards = self.reward_model(
        beliefs_seq.view(-1, H),
        states_seq.view(-1, Z),
      ).view(self.planning_horizon, B * self.candidates).sum(dim=0)

      # Refit distribution from top-k candidates
      rewards = rewards.view(B, self.candidates)
      _, top_idx = rewards.topk(self.top_candidates, dim=1, largest=True, sorted=False)
      actions = actions.view(self.planning_horizon, B, self.candidates, self.action_size)
      top_actions = actions[:, torch.arange(B, device=belief.device).unsqueeze(1), top_idx, :]  # [H, B, top_k, action_size]
      action_mean    = top_actions.mean(dim=2, keepdim=True)
      action_std_dev = top_actions.std(dim=2, keepdim=True)

    return action_mean[0].squeeze(dim=1)  
