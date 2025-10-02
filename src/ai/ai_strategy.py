import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class StrategyPolicyNet(nn.Module):
    def __init__(self, input_dim=100, output_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class AI_STRATEGY:
    def __init__(self, input_dim=100, output_dim=20, lr=1e-3):
        self.model = StrategyPolicyNet(input_dim, output_dim)
        self.target_model = StrategyPolicyNet(input_dim, output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = 0.99
        self.epsilon = 0.1

    def select_action(self, state_vector):
        if random.random() < self.epsilon:
            return random.randint(0, 19)
        with torch.no_grad():
            x = torch.tensor(state_vector, dtype=torch.float32)
            q_values = self.model(x)
            return q_values.argmax().item()

    def store_experience(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def train(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        r = torch.tensor(r, dtype=torch.float32)
        s_next = torch.tensor(s_next, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
        q_next = self.target_model(s_next).max(1)[0]
        target = r + self.gamma * q_next * (1 - done)
        loss = nn.functional.mse_loss(q, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())