import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class OrderPolicyNet(nn.Module):
    def __init__(self, input_dim=76, output_dim=19):  # 19人のスタメンスコアを評価して返す。各ポジションで、スコア最大の選手をスタメンに。
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)  # 出力は
        )

    def forward(self, x):
        return self.net(x)

class AI_ORDER:
    def __init__(self, input_dim=76, output_dim=19, lr=1e-3):
        self.model = OrderPolicyNet(input_dim, output_dim)
        self.buffer = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = 0.95
        self.epsilon = 0.2

    def generate_lineup(self, team_vector):
        lineup=[]
        position_numbers={"C":[],"1B":[],"2B":[],"SS":[],"3B":[],"LF":[],"CF":[],"RF":[]}
        numbers=[]
        for player_vector in team_vector:
            numbers.append(player_vector[5])
            for pos,num in position_numbers.items():
                if(pos==player_vector[4]):
                    position_numbers[pos].append(player_vector[5])
        if random.random() < self.epsilon:
            for pos,num in position_numbers.items():
                lineup.append(random.choice(num))
            possible_DH = [item for item in numbers if item not in lineup]
            DH=random.choice(possible_DH)
            lineup.append(DH)
            random.shuffle(lineup)
            return lineup
        else:
            lineup_with_score=[]
            with torch.no_grad():
                x=[]
                n=[]
                for player_vector in team_vector:
                    x.append(player_vector[0])
                    x.append(player_vector[1])
                    x.append(player_vector[2])
                    x.append(player_vector[3])
                    n.append(player_vector[5])
                x = torch.tensor(x, dtype=torch.float32)
                scores = self.model(x)
                for pos,num in position_numbers.items():
                    max_score=-100
                    select=None
                    for number in num:
                        if(max_score<scores[n.index(number)]):
                            select=number
                        max_score=max(max_score,scores[n.index(number)])
                    lineup_with_score.append([select,max_score])
                lineup_temp=[]
                for i in lineup_with_score:
                    lineup_temp.append(i[0])
                possible_DH = [item for item in numbers if item not in lineup_temp]
                max_dh=-100
                DH=None
                for pdh in possible_DH:
                    if(max_dh<scores[n.index(pdh)]):
                        DH=[pdh,scores[n.index(pdh)]]
                        max_dh=scores[n.index(pdh)]
                lineup_with_score.append(DH)
                lineup_with_score = sorted(lineup_with_score, key=lambda x: x[1])
                for i in lineup_with_score:
                    lineup.append(i[0])
                return lineup

    def store_experience(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def train(self, batch_size=32):
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
        q_next = self.model(s_next).max(1)[0]
        target = r + self.gamma * q_next * (1 - done)
        loss = nn.functional.mse_loss(q, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()