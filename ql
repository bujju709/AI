import numpy as np 
import random 
 
class GridEnvironment: 
    def _init_(self, size, goal): 
        self.size = size 
        self.goal = goal 
        self.state = (0, 0) 
        self.path = [] 
 
    def reset(self): 
        self.state = (0, 0) 
        self.path = [self.state] 
        return self.state 
 
    def step(self, action): 
        x, y = self.state 
        if action == 0:  # Up 
            x = max(0, x - 1) 
        elif action == 1:  # Down 
            x = min(self.size - 1, x + 1) 
        elif action == 2:  # Left 
            y = max(0, y - 1) 
        elif action == 3:  # Right 
            y = min(self.size - 1, y + 1) 
 
        self.state = (x, y) 
        self.path.append(self.state) 
        reward = 1 if self.state == self.goal else -0.1 
        done = self.state == self.goal 
        return self.state, reward, done 
 
    def render(self): 
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)] 
        for x, y in self.path: 
            grid[x][y] = '*' 
        x, y = self.state 
        grid[x][y] = 'A' 
        gx, gy = self.goal 
        grid[gx][gy] = 'G' 
        for row in grid: 
            print(' '.join(row)) 
        print() 
 
    def get_state_space(self): 
        return self.size * self.size 
 
    def get_action_space(self): 
        return 4  # Up, Down, Left, Right 
 
def q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1): 
    q_table = np.zeros((env.get_state_space(), env.get_action_space())) 
 
    def state_to_index(state): 
        return state[0] * env.size + state[1] 
 
    for episode in range(episodes): 
        state = env.reset() 
        done = False 
 
        while not done: 
            state_idx = state_to_index(state) 
            if random.uniform(0, 1) < epsilon: 
                action = random.randint(0, 3) 
            else: 
                action = np.argmax(q_table[state_idx]) 
 
            next_state, reward, done = env.step(action) 
            next_state_idx = state_to_index(next_state) 
 
            q_table[state_idx, action] += alpha * (reward + gamma * np.max(q_table[next_state_idx]) - 
q_table[state_idx, action]) 
            state = next_state 
 
    return q_table 
 
def analyze_performance(q_table, env): 
    state = env.reset() 
    steps = 0 
    done = False 
 
    def state_to_index(state): 
        return state[0] * env.size + state[1] 
 
    while not done and steps < 100: 
        env.render() 
        state_idx = state_to_index(state) 
        action = np.argmax(q_table[state_idx]) 
        state, _, done = env.step(action) 
        steps += 1 
    env.render() 
    print(f"Steps taken to reach goal: {steps}") 
 
env = GridEnvironment(size=5, goal=(4, 4)) 
q_table = q_learning(env) 
analyze_performance(q_table, env)
