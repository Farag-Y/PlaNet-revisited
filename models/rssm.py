import torch
from torch import nn
from torch.nn import functional as F

MIN_STD_DEV = 0.1


class RSSM(nn.Module):
    def __init__(self, state_size, hidden_size, belief_size, action_size, obs_size, non_linearity='relu',std_dev_fn="softplus"):
        super().__init__()
        self.act_fn = getattr(F, non_linearity)
        self.std_dev_fn = getattr(F, std_dev_fn)
        self.min_std_dev = MIN_STD_DEV
        self.fc_embed_state_action     = nn.Linear(state_size + action_size, belief_size)
        self.rnn                       = nn.GRUCell(input_size=belief_size, hidden_size=belief_size)
        self.fc_embed_belief_prior     = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior            = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + obs_size, hidden_size)
        self.fc_state_posterior        = nn.Linear(hidden_size, 2 * state_size)

    def forward(self, prev_state, actions, prev_belief, observations=None, nonterminals=None):
        #TODO: What to do with non terminals
        sequence_length = actions.shape[0] +1
        (det_hidden_states, prior_states, prior_means, prior_std_devs,
            posterior_states, posterior_means, posterior_std_devs) = (
                [[torch.empty(0)] * sequence_length for _ in range(7)]
            )
        #Belifes is the detemnistic hidden state
        det_hidden_states[0],prior_states[0],posterior_states[0] = prev_belief,prev_state,prev_state #TODO: Why here does prior and posterior share the same state ?
        for t in range(actions.shape[0]):
            prev_state = posterior_states[t] if observations else prior_states[t]
            prev_state = prev_state if nonterminals is None else prev_state*nonterminals[t]
            hidden_input = self.act_fn(self.fc_embed_state_action(torch.concat((prev_state,actions),dim=1)))## Why dimension 1 ?
            det_hidden_states[t+1]= self.rnn(hidden_input,det_hidden_states[t])

            ## Prior
            hidden_prior = self.act_fn(self.fc_embed_belief_prior(det_hidden_states[t+1]))
            prior_means[t+1],prior_std_devs[t+1] = torch.chunk(self.fc_state_prior(hidden_prior),2,dim=1)
            prior_std_devs[t+1] = self.std_dev_fn(prior_std_devs[t+1]) + self.min_std_dev
            ##Reparam trick
            prior_states = prior_means[t+1] + prior_std_devs[t+1]*torch.randn_like(prior_std_devs[t+1]) #TODO: rand like Std_dev here is correct ?
            ##Posterior TODO: Posterior.
            if observations:
                hidden_posterior = self.act_fn(self.fc_embed_belief_posterior(torch.concat((det_hidden_states[t+1],observations[t]),dim=1)))
                posterior_means[t+1],posterior_std_devs[t+1] = torch.chunk(self.fc_state_prior(hidden_posterior),2,dim=1)
                posterior_std_devs[t+1] = self.std_dev_fn(posterior_std_devs[t+1])+self.min_std_dev
                #Reparam trick
                posterior_states = posterior_means[t+1] + posterior_std_devs[t+1]*torch.randn_like(posterior_std_devs[t+1]) #TODO: rand like Std_dev here is correct ?
                # Return new hidden states
        hidden = [torch.stack(det_hidden_states[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
        return hidden

