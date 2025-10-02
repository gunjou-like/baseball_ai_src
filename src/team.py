import random
class Team:
    def __init__(self, team_data,team_name):
        self.name = team_name
        self.team_data=team_data
        players=[]
        for id, player in team_data.items():
            players.append[player]
        self.players = players # 全選手データdictのりすと
        self.lineup = []                    # スタメン選手
        self.batting_order = []             # 打順（インデックスの並び）
        self.relievers = []
        self.SPs=[]
        for player in players:
            if(players["position"] in ["SP"]):
                self.SPs.append(player)
        for player in players:
            if(players["position"] in ["RP","CP"]):
                self.relievers.append(player)
        self.current_pitcher = None

    def set_lineup(self, indices):
        """スタメン選手をインデックスで指定"""
        self.lineup = indices
        self.batting_order=self.lineup

    def get_batter(self, index):
        """打順インデックスから打者を取得"""
        if not self.lineup:
            raise ValueError("Lineup not set")
        batter_index = self.batting_order[index % len(self.batting_order)]
        return self.lineup[batter_index]

    def select_SP(self):
        """先発投手を選出（ランダム）"""
        if not self.SPs:
            raise ValueError("No starting pitchers available")
        self.current_pitcher = random.choice(self.SPs)
        return self.current_pitcher

    def get_player_vector(self, player):
        """選手ベクトルを生成（AI_ORDERやAI_STRATEGY用）"""
        position_list = ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
        hitting = player["ability"]["hitting"] / 100
        power = player["ability"]["power"] / 100
        speed = player["ability"]["speed"] / 100
        defence = player["ability"]["defence"] / 100
        fatigue = player.get("fatigue", 0) / 100
        injury_flag = 1 if player.get("injury", False) else 0
        condition_score = player.get("condition", 50) / 100

        pos_index = position_list.index(player["position"])
        position_onehot = [0] * len(position_list)
        position_onehot[pos_index] = 1

        return [
            hitting, power, speed, defence,
            fatigue, injury_flag, condition_score
        ] + position_onehot
    

def encode_team(team_data):
    #ok
    #野手限定
    position_list = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]

    player_vectors = []
    for index, player in team_data.items():
        p_or_b=False
        for pos in position_list:
            if(player["position"]==pos):
                p_or_b=True
        if p_or_b == False:
            continue
        # 能力値の正規化（0〜1）
        fatigue = player["status"]["fatigue"]/100
        injury = player["status"]["injury"]
        condition = player["status"]["condition"]
        status_r=(1-fatigue)*(1-injury/4)
        if(condition=="A"):
            status_r*=1.2
        elif(condition=="C"):
            status_r*=0.8
        hitting =min(player["ability"]["hitting"]*status_r / 100,1)
        power =min(player["ability"]["power"]*status_r/ 100 ,1)
        speed = min(player["ability"]["speed"]*status_r / 100,1)
        defence = player["ability"]["defence"] / 100
    
        
        # ポジション one-hot
        pos_index = position_list.index(player["position"])
        position_onehot = [0] * len(position_list)
        position_onehot[pos_index] = 1

        # 選手ベクトル結合
        vector = [
            hitting, power, speed, defence,
        ] + player["position"] +player["number"]
        #[0.4,0.3,0.5,0.2,"SS",背番号]

        player_vectors.append(vector)

    # チーム統計（平均値など）
    # avg_vector = ...
    # injured_count = ...
    # fatigue_avg = ...
    #team_features = avg_vector[:7] + [injured_count / len(team_data["players"]), fatigue_avg / 100]

    return player_vectors      # 個別ベクトル（必要ならAI_ORDERで使える）