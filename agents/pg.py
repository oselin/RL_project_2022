###import sys, os
###sys.path.insert(0, os.path.abspath(".."))
###import torch
###import torch.nn.functional as F
###from torch import nn
###from torch.distributions import Normal
###import numpy as np
###from common import helper as h
###
###
#### Use CUDA for storing tensors / calculations if it's available
###device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###
#### Initialisation function for neural network layers
###def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
###    torch.nn.init.orthogonal_(layer.weight, std)
###    torch.nn.init.constant_(layer.bias, bias_const)
###    return layer
###
#### This class defines the neural network policy
###class Policy(nn.Module):
###    def __init__(self, state_dim, action_dim):
###        super(Policy, self).__init__()
###
###        # Initialise a neural network with two hidden layers (64 neurons per layer)
###        self.actor_mean = nn.Sequential(
###            layer_init(nn.Linear(state_dim, 64)), nn.Tanh(),
###            layer_init(nn.Linear(64, 64)), nn.Tanh(),
###            layer_init(nn.Linear(64, action_dim), std=0.01),
###        )
###
###        # TODO: Task 1: Implement actor_logstd as a torch tensor
###        # TODO: Task 2: Implement actor_logstd as a learnable parameter
###        # Use log of std to make sure std doesn't become negative during training
###        self.actor_logstd = 0
###        #self.actor_logstd = torch.zeros(1, action_dim, device=device)
###        #self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
###
###    # Do a forward pass to map state to action
###    def forward(self, state):
###        # Get mean of a Normal distribution (the output of the neural network)
###        action_mean = self.actor_mean(state)
###
###        # Make sure action_logstd matches dimension of action_mean
###        action_logstd = self.actor_logstd.expand_as(action_mean)
###
###        # Exponentiate the log std to get actual std
###        action_std = torch.exp(action_logstd)
###
###        # TODO: Task 1: Create a Normal distribution with mean of 'action_mean' and standard deviation of 'action_logstd', and return the distribution
###        probs = Normal(action_mean, action_std)
###
###        return probs
###
###
#### Class for the Policy Gradient algorithm
###class PG(object):
###    def __init__(self, state_dim, action_dim, lr, gamma):
###
###        # Define the neural network policy
###        self.policy = Policy(state_dim, action_dim).to(device)
###
###        # Create an optimizer
###        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
###
###        # Set discount factor value
###        self.gamma = gamma
###
###        # Simple buffers for action probabilities and rewards
###        self.action_probs = []
###        self.rewards = []
###
###    def update(self,):
###
###        # Prepare dataset used to update policy
###        action_probs = torch.stack(self.action_probs, dim=0) \
###                .to(device).squeeze(-1) # shape: [batch_size,]
###        rewards = torch.stack(self.rewards, dim=0).to(device).squeeze(-1) # shape [batch_size,]
###        self.action_probs, self.rewards = [], [] # clean buffers
###        
###        # TODO: Task 1: Implement the policy gradient
###        ########## Your code starts here. ##########
###        # Hints:
###        #   1. compute discounted rewards (use the discount_rewards function offered in common.helper)
###        #   2. compute the policy gradient loss
###        #   3. update the parameters (backpropagate gradients, do the optimizer step, empty optimizer gradients afterwards so that gradients don't accumulate over updates)
###        
###        # pass
###
###        # Compute discounted rewards 
###        discounted_rewards = h.discount_rewards(rewards, self.gamma)
###        discounted_rewards -= torch.mean(discounted_rewards)
###        discounted_rewards /= torch.std(discounted_rewards)
###
###        # Calculate the PG loss
###        weighted_probs = -action_probs * discounted_rewards 
###        loss = torch.mean(weighted_probs)  
###
###        # Backprop gradients
###        loss.backward()  
###
###        # Do the optimizer step 
###        self.optimizer.step() 
###        self.optimizer.zero_grad() 
###        ########## Your code ends here. ##########
###
###        # if you want to log something in wandb, you can put them inside the {}, otherwise, just leave it empty.
###        return {'logstd': self.policy.actor_logstd.cpu().detach().numpy()}
###
###
###    def get_action(self, observation, evaluation=False):
###        """Return action and logprob of this action."""
###
###        # Add batch dimension if necessary
###        if observation.ndim == 1: observation = observation[None]
###
###        # Convert observation to a torch tensor
###        x = torch.from_numpy(observation).float().to(device)
###
###        # TODO: Task 1: Calculate action and its log_prob
###        ########## Your code starts here. ###########
###        # Hint: 
###        #   1. when evaluation=True, return mean, otherwise return samples from the distribution created in self.policy.forward() function.
###        #   2. notice the shape of action and act_logprob.
###        
###        # action = 0
###        # act_logprob = 0
###
###        # Pass state x through the policy network (T1)
###        dist = self.policy.forward(x)  
###        # Return mean if evaluation, else sample from the distribution
###        if evaluation:
###            action = dist.mean 
###        else:
###            action = dist.sample() 
###
###        # Calculate the log probability of the action (T1)
###        act_logprob = dist.log_prob(action).sum(-1) 
###        ########## Your code ends here. ##########
###        
###        if observation.ndim == 1: action = action[0]
###
###        return action, act_logprob
###
###    def record(self, action_prob, reward):
###        """ Store agent's and env's outcomes to update the agent."""
###        self.action_probs.append(action_prob)
###        self.rewards.append(torch.tensor([reward]))
###
###    def save(self, filepath):
###        torch.save(self.policy.state_dict(), filepath)
###
###    def load(self, filepath):
###        self.policy.load_state_dict(torch.load(filepath))




from symbol import parameters
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np
from common import helper as h


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Actor-critic agent
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        # TODO: Task 1: Implement actor_logstd as a learnable parameter
        # Use log of std to make sure std doesn't become negative during training
        actor_std = torch.ones(action_dim)
        self.actor_logstd = torch.nn.Parameter(torch.log(actor_std))


    def forward(self, state):
        # Get mean of a Normal distribution (the output of the neural network)
        action_mean = self.actor_mean(state)

        # Make sure action_logstd matches dimension of action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)

        # Exponentiate the log std to get actual std
        action_std = torch.exp(action_logstd)

        # TODO: Task 1: Create a Normal distribution with mean of 'action_mean' and standard deviation of 'action_logstd', and return the distribution
        probs = Normal(action_mean, action_std)


        return probs

class Value(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.value = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1)))
    
    def forward(self, x):
        return self.value(x).squeeze(1) # output shape [batch,]


class PG(object):
    def __init__(self, state_dim, action_dim, lr, gamma, grad_clip_norm = 10):
        self.policy = Policy(state_dim, action_dim).to(device)
        self.value = Value(state_dim).to(device)
        self.optimizer = torch.optim.Adam(list(self.policy.parameters())+ list(self.value.parameters()), 
                                         lr=lr,)

        self.gamma = gamma
        self.grad_clip_norm = grad_clip_norm

        # a simple buffer
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.next_states = []


    def update(self,):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(device).squeeze(-1)
        dones = torch.stack(self.dones, dim=0).to(device).squeeze(-1)
        # clear buffer
        self.states, self.action_probs, self.rewards, self.dones, self.next_states = [], [], [], [], []

        # TODO: Task 1
        ########## Your code starts here. ##########
        # Hints: 1. calculate the TD target as well as the MSE loss between the predicted value and the TD target
        #        2. calculate the policy loss (similar to ex5) with advantage calculated from the value function. Normalise
        #           the advantage to zero mean and unit variance.
        #        3. update parameters of the policy and the value function jointly
        current_value = self.value(states)
        next_value = self.value(next_states)
        next_value[dones == 1] = 0

        TD_target = rewards + self.gamma*next_value
        val_loss = torch.mean(torch.square(TD_target.detach() - current_value))

        advantage_fun = TD_target - current_value
        #zero mean, unitary std
        mean, std = torch.mean(advantage_fun), torch.std(advantage_fun)
        advantage_fun  = (advantage_fun-mean)/std
        policy_loss = torch.mean(-action_probs * advantage_fun.detach())

        total_loss = val_loss + policy_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.value.parameters()) + list(self.policy.parameters()), self.grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        ########## Your code ends here. ##########

        # if you want to log something in wandb, you can put them inside the {}, otherwise, just leave it empty.
        return {}


    def get_action(self, observation, evaluation=False):
        """Return action (np.ndarray) and logprob (torch.Tensor) of this action."""
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        x = torch.from_numpy(observation).float().to(device)

        # TODO: Task 1
        ########## Your code starts here. ##########
        # Hints: 1. the self.policy returns a normal distribution, check the PyTorch document to see 
        #           how to calculate the log_prob of an action and how to sample.
        #        2. if evaluation, return mean, otherwise, return a sample
        #        3. the returned action and the act_logprob should be the torch.Tensors.
        #            Please always make sure the shape of variables is as you expected.
        
        action_dist = self.policy.forward(x)
        
        if evaluation: action = action_dist.mean
        else:          action = action_dist.sample()

        act_logprob = action_dist.log_prob(action).sum(axis=1)      

        ########## Your code ends here. ###########

        return action, act_logprob

    def record(self, observation, action_prob, reward, done, next_observation):
        self.states.append(torch.tensor(observation, dtype=torch.float32))
        self.action_probs.append(action_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.dones.append(torch.tensor([done], dtype=torch.float32))
        self.next_states.append(torch.tensor(next_observation, dtype=torch.float32))

    # You can implement these if needed, following the previous exercises.
    def load(self, filepath):
        pass
    
    def save(self, filepath):
        pass