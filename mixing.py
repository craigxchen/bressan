### TODO: 1-hot encode actions, limit to up/right movement only, 

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment import Grid

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Mixer(nn.Module):
    def __init__(self, grid_size=4, kernel_size=2, n_filters=8, n_latent_var=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_grid = grid_size**2
        
        # # 1 input image channel, n_filters output channels
        self.conv1 = nn.Conv2d(1, n_filters, kernel_size, 
                               padding=(kernel_size-1), padding_mode='circular') # TODO: ADD PERIODIC PADDING
        self.relu1 = nn.LeakyReLU()
        # self.pool1 = nn.MaxPool2d((2,2))

        # # flatten and then feed into linears
        self.fltn = nn.Flatten()
        self.fc1 = nn.Linear(n_filters * (grid_size + 2*(kernel_size-1) - kernel_size + 1)**2, n_latent_var)
        
        # self.fltn = nn.Flatten()
        # self.fc1 = nn.Linear(grid_size**2, n_latent_var)
        self.relu2 = nn.LeakyReLU()
        self.fc2 = nn.Linear(n_latent_var, n_latent_var)
        self.relu3 = nn.LeakyReLU()
        # self.fc3 = nn.Linear(n_latent_var, (self.num_grid + 2)) # four possible directions to move, n^2 points
        self.fc3 = nn.Linear(n_latent_var, self.num_grid//2) # two moves from each point
        
        # self.deltas = [-grid_size, 1, grid_size, -1] # up, right, down, left

    def forward(self, x):
        # x = self.pool1(self.relu1(self.conv1(x)))
        x = self.relu1(self.conv1(x))
        
        x = self.fltn(x)

        x = self.relu2(self.fc1(x))
        x = self.relu3(self.fc2(x))
        x = self.fc3(x)

        # s, d = torch.split(x, [self.grid_size**2, 2], dim=1)
        # return nn.functional.softmax(s, dim=1), nn.functional.softmax(d, dim=1)
        return nn.functional.softmax(x, dim=1)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.costs = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.costs[:]
        del self.is_terminals[:]

# use different algorithm? actor-critic?
class PG:
    def __init__(self, grid_size=4, kernel_size=2, n_filters=8, n_latent_var=64, 
                 lr=3e-4, gamma=0.99):
        self.gamma = gamma
        self.eps_clip = 0.2
        
        self.policy = nn.Sequential(nn.Conv2d(1, n_filters, kernel_size, padding=(kernel_size-1), padding_mode='circular'),
                                    nn.LeakyReLU(),
                                    nn.Flatten(),
                                    nn.Linear(n_filters * (grid_size + 2*(kernel_size-1) - kernel_size + 1)**2, n_latent_var),
                                    nn.LeakyReLU(),
                                    nn.Linear(n_latent_var, n_latent_var//2),
                                    nn.LeakyReLU(),
                                    nn.Linear(n_latent_var//2, grid_size**2 // 2),
                                    nn.Softmax(dim=1)
                                    ).to(device)
        
        self.value = nn.Sequential(nn.Conv2d(1, n_filters, kernel_size, padding=(kernel_size-1), padding_mode='circular'),
                                    nn.LeakyReLU(),
                                    nn.Flatten(),
                                    nn.Linear(n_filters * (grid_size + 2*(kernel_size-1) - kernel_size + 1)**2, n_latent_var),
                                    nn.LeakyReLU(),
                                    nn.Linear(n_latent_var, 1)
                                    ).to(device)
        
        self.pol_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.val_optimizer = torch.optim.Adam(self.value.parameters(), lr=3*lr)
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):        
        probs = self.policy(state)
        dist = Categorical(probs=probs)
        
        action = dist.sample()
        
        if memory is not None:
            logprob = dist.log_prob(action)
            memory.logprobs.append(logprob)
            memory.states.append(state)
            memory.actions.append(action)
        else:
            pass
        
        return action.item()

    def evaluate(self, states, actions):
        action_probs = self.policy(states)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        
        state_value = self.value(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def update_actor(self, memory):
        # Monte Carlo estimate of state costs:

        costs = []
        discounted_cost = torch.zeros(1)
        for cost, is_terminal in zip(reversed(memory.costs), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_cost = torch.zeros(1)
            discounted_cost = cost + (self.gamma * discounted_cost)
            costs.insert(0,discounted_cost)
            
        # normalizing
        costs = torch.tensor(costs).to(device)
        costs = (costs - costs.mean()) / (costs.std() + 1e-5)
            
        old_states = torch.vstack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Evaluating old actions and values :
        logprobs, state_values, _ = self.evaluate(old_states, old_actions)
        
        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(logprobs - old_logprobs.detach())
            
        # Finding Surrogate Loss:
        advantages = costs - state_values.detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
        actor_loss = torch.min(surr1, surr2)
        
        # take gradient step
        self.pol_optimizer.zero_grad()
        actor_loss.mean().backward()
        self.pol_optimizer.step()


        # take gradient step
        self.pol_optimizer.step()
        return
    
    def update_critic(self, memory):
        states = torch.vstack(memory.states).to(device).detach()
        costs = []
        discounted_cost = torch.zeros(1)
        for state, cost, is_terminal in zip(reversed(memory.states), reversed(memory.costs), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_cost = torch.zeros(1)
            discounted_cost = cost + (self.gamma * discounted_cost)
            costs.append(discounted_cost)

        costs = torch.tensor(costs).to(device).reshape(-1,1)
        
        state_values = self.value(states)
        
        critic_loss = 0.5*self.MseLoss(state_values, costs)
        
        # take gradient step
        self.val_optimizer.zero_grad()
        critic_loss.mean().backward()
        self.val_optimizer.step()
        return


def simulate(policy):
    state, _, done = env.reset()
    state_hist = [state]
    t=0
    while (not done) and (t<16):
        t+=1
        action = policy.select_action(torch.from_numpy(state).float()[None,None], None)
        state, _, done = env.step(action)
        state_hist.append(state.copy())
    return state_hist

############ Training parameters ############

n = 4 # grid size

log_interval = 100  # print avg cost in the interval
max_episodes = 100000  # max training episodes
exp_length = 10
random_seed = 1

#############################################

env = Grid(n)
model = PG(grid_size=n, kernel_size=3, n_filters=16, n_latent_var=128, lr=1e-3)
memory = Memory()

# logging variables
running_cost = 0

# training loops
for ep in range(1,max_episodes):

    k = np.random.randint(1,5)
    if k == 4:
        state, _, done = env.reset()
    else:
        state, _, done = env.scramble_reset(k)
    
    ep_cost = 0
    state = torch.tensor(state).float()[None,None]
    i=0
    while i < exp_length:
        action = model.select_action(state, memory) 

        # Change cost function?
        next_state, cost, done = env.step(action)
    
        # Saving cost and is_terminals:
        memory.costs.append(cost)
        memory.is_terminals.append(done)

        ep_cost += cost.item()
        
        state = torch.tensor(next_state).float()[None,None]
        i += 1
        if done:
            break
    
    running_cost += ep_cost/(i)

    model.update_actor(memory)
    model.update_critic(memory)
    memory.clear_memory()
    
    if ep%log_interval == 0:
        print(f'episode {ep}')
        print(f'\t running average exp cost: {running_cost/log_interval:.2f}')
        running_cost = 0

# %%

# import plotly.graph_objects as go

# # Create figure
# fig = go.Figure()

# state_seq = simulate(model)

# # Add traces, one for each slider step
# for step in np.arange(0, 5, 0.1):
#     fig.add_trace(
#         go.imshow(
#             state_seq[i],
#             visible=False,
#             ))

# # Make 10th trace visible
# fig.data[10].visible = True

# # Create and add slider
# steps = []
# for i in range(len(fig.data)):
#     step = dict(
#         method="update",
#         args=[{"visible": [False] * len(fig.data)},
#               {"title": "Slider switched to step: " + str(i)}],  # layout attribute
#     )
#     step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
#     steps.append(step)

# sliders = [dict(
#     active=10,
#     currentvalue={"prefix": "Frequency: "},
#     pad={"t": 50},
#     steps=steps
# )]

# fig.update_layout(
#     sliders=sliders
# )

# fig.show()