import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cpu")

class PPO(nn.Module):
    def __init__(self, grid_size=4, kernel_size=2, n_filters=8, n_latent_var=64, 
                 lr=3e-4, gamma=0.99, lmbda=0.9, beta=0.4, eps_clip=0.2, k_epoch=1):
        super(PPO, self).__init__()
        self.data = []
        self.gamma = gamma
        self.lr = lr
        self.lmbda = lmbda
        self.beta = beta
        self.eps_clip = eps_clip
        self.k_epoch = k_epoch
        
        self.shared = nn.Sequential(nn.Conv2d(1, n_filters, kernel_size, padding=(kernel_size-1), padding_mode='circular'),
                                    nn.LeakyReLU(),
                                    nn.Flatten(),
                                    nn.Linear(n_filters * (grid_size + 2*(kernel_size-1) - kernel_size + 1)**2, n_latent_var),
                                    nn.LeakyReLU(),
                                    )
        
        # self.shared = nn.Sequential(nn.Flatten(),
        #                             nn.Linear(grid_size**2, n_latent_var),
        #                             nn.LeakyReLU(),)
        
        self.policy = nn.Sequential(nn.Linear(n_latent_var, n_latent_var),
                                    nn.LeakyReLU(),
                                    nn.Linear(n_latent_var, grid_size**2 * 2),
                                    nn.Softmax(dim=1)
                                    ).to(device)
        
        self.value = nn.Sequential(nn.Linear(n_latent_var, 1),
                                   ).to(device)
        
        # self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.pol_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.val_optimizer = optim.Adam(self.value.parameters(), lr=lr)

    def pi(self, x):
        return self.policy(self.shared(x))
    
    def v(self, x):
        return self.value(self.shared(x))
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, ent_lst, done_lst = [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, ent, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            ent_lst.append(ent)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a, entropy = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst), \
                                          torch.tensor(ent_lst, dtype=torch.float)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, entropy
        
    def update(self):
        s, a, r, s_prime, done_mask, prob_a, dist_entropy = self.make_batch()
        
        for i in range(self.k_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            actor_loss = torch.min(surr1, surr2) - self.beta * dist_entropy
            critic_loss = nn.functional.mse_loss(self.v(s), td_target.detach())
            # total_loss = actor_loss + critic_loss

            # self.optimizer.zero_grad()
            # total_loss.mean().backward()
            # self.optimizer.step()

            self.pol_optimizer.zero_grad()
            actor_loss.mean().backward()
            self.pol_optimizer.step()
            
            self.val_optimizer.zero_grad()
            critic_loss.mean().backward()
            self.val_optimizer.step()
            
        return