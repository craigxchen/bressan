import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from environment import Grid

#Hyperparameters
learning_rate = 3e-4
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 10

device = torch.device("cpu")

class PPO(nn.Module):
    def __init__(self, grid_size=4, kernel_size=2, n_filters=8, n_latent_var=64, 
                 lr=3e-4, gamma=0.99):
        super(PPO, self).__init__()
        self.data = []
        
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
        
        self.pol_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.val_optimizer = optim.Adam(self.value.parameters(), lr=learning_rate)

    def pi(self, x):
        return self.policy(x)
    
    def v(self, x):
        return self.value(x)
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        
        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            actor_loss = torch.min(surr1, surr2) 
            critic_loss = F.mse_loss(self.v(s), td_target.detach())

            self.pol_optimizer.zero_grad()
            actor_loss.mean().backward()
            self.pol_optimizer.step()
            
            self.val_optimizer.zero_grad()
            critic_loss.mean().backward()
            self.val_optimizer.step()
            
        return
        

env = Grid(4)
model = PPO()
score = 0.0
print_interval = 1000

for n_epi in range(1000000):
    k = np.random.randint(1,5)
    if k == 4:
        s, _, done = env.reset()
    else:
        s, _, done = env.scramble_reset(k)
        
    for t in range(T_horizon):
        prob = model.pi(torch.from_numpy(s).float()[None,None])
        m = Categorical(prob)
        a = m.sample().item()
        s_prime, r, done = env.step(a)

        model.put_data((s[None], a, r, s_prime[None], prob[0][a].item(), done))
        s = s_prime

        score += r
        if done:
            break

    model.train_net()

    if n_epi%print_interval==0 and n_epi!=0:
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
        score = 0.0