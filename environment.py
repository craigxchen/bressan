import numpy as np

class Grid:
    def __init__(self, n=4):
        assert n%2 == 0
        self.n = n
        
        self.start = np.array([[1]*(n//2) + [0]*(n//2) for _ in range(n)])
        
        # checkerboards are the most mixed
        cb1 = np.zeros((n,n), dtype=int)
        cb1[1::2, ::2] = 1
        cb1[::2, 1::2] = 1
        
        cb2 = 1-cb1
        self.final = [cb1,cb2]
        
        # tracking agent info        
        self.curr_state = self.start.copy()
        
    def step(self, action): 
        # action will be inputted as a number in range(self.n**2 * 2)
        assert action in list(range(self.n**2 * 2))
        
        point = action//2 
        
        if action%2 == 0: # move up
            swap = (point - self.n)%(self.n**2)
        else: # move right
            if point%self.n == (self.n-1):
                swap = (point - (self.n-1))
            else:
                swap = point + 1
        
        (x1,y1),(x2,y2) = ((point//self.n, point%self.n), 
                           (swap//self.n, swap%self.n))
        
        temp1 = self.curr_state[x1][y1]
        
        self.curr_state[x1][y1] = self.curr_state[x2][y2]
        self.curr_state[x2][y2] = temp1
        
        done = np.array_equal(self.curr_state, self.final[0]) or np.array_equal(self.curr_state, self.final[1]) 
        
        return (self.curr_state, self.get_cost(self.curr_state), done)
    
    def reset(self):
        self.curr_state = self.start.copy()
        return self.curr_state, False
    
    def rand_reset(self):
        temp = np.ones(self.n**2)
        temp[:(self.n**2)//2] = 0
        np.random.shuffle(temp)
        self.curr_state = temp.reshape(self.n,self.n)
        
        done = np.array_equal(self.curr_state, self.final[0]) or np.array_equal(self.curr_state, self.final[1])
        
        return self.curr_state, done
    
    def scramble_reset(self, k):
        # take k random moves from goal state
        b = np.random.binomial(1,1/2)
        goal = self.final[b].copy() # random goal (assuming k ~ unif(1,8))
        
        for i in range(k):
            point_index = np.random.randint(0,self.n**2)
            index_delta = np.random.choice([-self.n, 1, self.n, -1])
            if (point_index%self.n == 0) and (index_delta == -1):
                swap_index = point_index + self.n-1
            else:
                swap_index = (point_index + index_delta)%(self.n**2)
            
            (x1,y1), (x2,y2) = ((point_index//self.n, point_index%self.n), 
                                (swap_index//self.n, swap_index%self.n))
            
            temp1 = goal[x1][y1]
            
            goal[x1][y1] = goal[x2][y2]
            goal[x2][y2] = temp1
            
        self.curr_state = goal
        return self.curr_state, False

    def get_cost(self, state):
        return min(self.L1_cost(state, self.final[0]),
                   self.L1_cost(state, self.final[1]))
    
    @staticmethod
    def L1_cost(x,y):
        return np.sum(np.abs(x-y))