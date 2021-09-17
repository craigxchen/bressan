import numpy as np
import torch
from torch.distributions import Categorical

from environment import Grid
from ppo import PPO

# Hyperparameters
grid_size   = 4
T_horizon   = 10
N_iterations= 100000
random_seed = None
if random_seed is not None:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    

env = Grid(grid_size)
model = PPO(grid_size=grid_size, kernel_size=3, n_filters=32, n_latent_var=128, lr=3e-4, 
            gamma=0.99, lmbda=0.9, beta=0.4, eps_clip=0.2, k_epoch=1)
model.train()

# uncomment to load trained model on 4x4 grid
# model = PPO(grid_size=4, kernel_size=3, n_filters=32, n_latent_var=128, lr=3e-4, 
#             gamma=0.99, lmbda=0.9, beta=0.4, eps_clip=0.2, k_epoch=1)
# model.load_state_dict(torch.load('bressan.pt'))
# model.eval()

# %% Training loop

score = 0.0
exp_len = 0.0
print_interval = 1000

for n_epi in range(N_iterations):
    # k = np.random.randint(1,5)
    # if k == 4:
    #     s, done = env.reset()
    # else:
    #     s, done = env.scramble_reset(k)

    s, done = env.reset()
        
    ep_cost = 0
    for t in range(T_horizon):
        prob = model.pi(torch.from_numpy(s).float()[None,None])
        m = Categorical(prob)
        ent = m.entropy()
        a = m.sample().item()
        s_prime, r, done = env.step(a)

        model.put_data((s[None], a, r, s_prime[None], prob[0][a].item(), ent, done))
        s = s_prime

        ep_cost += r
        if done:
            break

    score += ep_cost/(t+1)
    exp_len += t+1

    model.update()

    if n_epi%print_interval==0 and n_epi!=0:
        print(f"episode {n_epi}")
        print(f"\t avg cost: {score/print_interval:.2f}, avg exp len: {exp_len/print_interval:.2f}") # optimal cost is < 3
        if score/print_interval <= 3:
            break 
        score = 0.0
        exp_len = 0.0

# %% Simulate learned policy

def simulate(policy):
    state, done = env.reset()
    state_hist = [state.copy()]
    t=0
    while (not done) and (t<16):
        t+=1
        probs = policy(torch.from_numpy(state).float()[None,None])
        dist = Categorical(probs=probs)
        
        action = dist.sample().item()
        
        state, _, done = env.step(action)
        state_hist.append(state.copy())
    return state_hist

import plotly.express as px
# from plotly.offline import plot

state_seq = np.array(simulate(model.pi))

fig = px.imshow(state_seq, animation_frame=0)

fig["layout"].pop("updatemenus") # optional, drop animation buttons
fig.show()
        
        