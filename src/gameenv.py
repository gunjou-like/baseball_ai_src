from gamestate import GameState
from ai.ai_strategy import AI_STRATEGY
from team import Team

class BaseballEnv:
    def __init__(self, team_a:Team, team_b:Team):
        self.state = GameState(self.team_a, self.team_b)
        self.done = False

    def reset(self):
        self.state = GameState(self.team_a, self.team_b)
        self.done = False
        return self.state.to_vector()

    def step(self, action):
        reward = self.state.apply_action(action)
        next_state = self.state.to_vector()
        self.done = self.state.is_terminal()
        return next_state, reward, self.done