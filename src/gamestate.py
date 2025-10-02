class GameState:
    def __init__(self, team_a, team_b):
        self.inning = 1
        self.half = "top"
        self.outs = 0
        self.bases = [None, None, None]
        self.score = { "A": 0, "B": 0 }
        self.team_a = team_a
        self.team_b = team_b
        self.current_batter= team_a.team_data[str(team_a.lineup[0])] #dict
        self.current_pitcher = team_b.current_pitcher() #dict

    def apply_action(self, action):
        # 例: 代打、盗塁、投手交代などを処理
        reward = self.resolve_play(action)
        self.update_state()
        return reward

    def is_terminal(self):
        return self.inning > 9 and self.half == "bottom" and self.score["A"] != self.score["B"]

    def to_vector(self):
        # 状態をAIモデルに渡すベクトル形式に変換
        return encode_game_state(self)
    
    def update_state(self):
        return None
    