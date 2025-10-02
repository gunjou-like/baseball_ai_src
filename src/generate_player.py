import random
import json

name_json_path = "./name.json"

def create_default_record(injury,batting_ability):
    batting_score=0
    for ability in batting_ability:
        batting_score+=batting_ability[ability]
    if(batting_score>300):
        PA=443*(1-injury/4)
    else:
        PA=443*batting_score/300*(1-injury/4)
    PA=round(PA)
    AVG = round((0.320-0.100)*batting_ability["hitting"]/100+0.100,3)
    HR = round(AVG*200*batting_ability["power"]*PA/(100*443))
    SB = round(50*batting_ability["speed"]*PA/(100*443))
    return {"PA":PA, "AVG":AVG, "HR":HR, "SB":SB}

def estimate_era(power, control, breaking):
    base = 5.00 - (0.015 * power + 0.01 * control + 0.01 * breaking)
    noise = random.uniform(-0.3, 0.3)
    return round(max(1.50, base + noise), 2)

def estimate_results(role, ip, era):
    if role == "SP":
        wins = int(ip / 15 * (4.5 - era) / 3.0)
        losses = int(ip / 25 * (era - 2.5) / 2.0)
        return [wins, losses, 0, 0]
    elif role == "RP":
        holds = int(ip / 2.5 * (4.5 - era) / 3.0)
        return [0, 0, holds, 0]
    elif role == "CP":
        saves = int(ip / 1.5 * (4.5 - era) / 3.0)
        return [0, 0, 0, saves]

def create_default_pitcher_record(injury,position,pitching_ability):
    pitching_score=0
    for ability in pitching_ability:
        pitching_score+=pitching_ability[ability]
    if position=="SP":
        IP = round(random.uniform(120, 180) * (pitching_ability["stamina"] / 100), 1)*(1-injury/4)
    elif position=="RP":
        IP  = round(random.uniform(40, 80) * (pitching_ability["stamina"] / 100), 1)*(1-injury/4)
    else:
        IP = round(random.uniform(30, 60) * (pitching_ability["stamina"] / 100), 1)*(1-injury/4)
    ERA=estimate_era(pitching_ability["power"],pitching_ability["control"],pitching_ability["curveball"])
    results= estimate_results(position,IP,ERA)#勝、敗、ホールドポイント、セーブ
    return {"IP":IP, "ERA":ERA, "results":results}

class PLAYER_GENERATOR():
    def __init__(self, number,name_json_path):
        self.number = number #背番号
        self.position = ["C","1B","2B","SS","3B","LF","CF","RF"]
        self.pitcher_position = ["SP","RP","CP"]
        self.min_salary = 5000000
        self.max_salary = 500000000
        self.hidden_batting_ability = ["hitting","power","speed","defence"]
        self.hidden_piching_ability = ["power", "control","curveball","stamina"]
        self.batting_record = ["PA", "AVG", "HR", "SB"]
        self.pitching_record = []

        with open(name_json_path, 'r', encoding='utf-8') as f:
            self.name_data = json.load(f)




    def generate_batter(self):
        player_data = {}
        player_data["number"] = self.number
        family_name = random.choice(self.name_data["Family_name"])
        given_name = random.choice(self.name_data["Given_name"])
        player_data["name"] = family_name+" "+given_name
        player_data["age"] = random.randint(18,40)
        if random.randint(0,100)<35:
            player_data["hitting_rl"]="L"
        else:
            player_data["hitting_rl"]="R"
        if random.randint(0,100)<20:
            player_data["pitching_rl"]="L"
        else:
            player_data["pitching_rl"]="R"
        player_data["position"] = random.choice(self.position)
        ability_list = self.hidden_batting_ability
        ability_score = 0 #年俸は打撃能力の合計スコア依存で決める、年俸決定ロジックは後から拡張
        player_data["ability"] = {}
        for ability in ability_list:
            player_data["ability"][ability] = random.randint(1,90)
            ability_score+=player_data["ability"][ability]
        if(ability_score<100):
            max_salary = round((self.max_salary-self.min_salary)*2/10 + self.min_salary)
            min_salary = self.min_salary
        elif(ability_score<200):
            max_salary = round((self.max_salary-self.min_salary)*5/10 + self.min_salary)
            min_salary = round((self.max_salary-self.min_salary)*2/10 + self.min_salary)
        else:
            max_salary = self.max_salary
            min_salary = round((self.max_salary-self.min_salary)*5/10 + self.min_salary)
        player_data["salary"] = random.randint(round(min_salary/1000000),round(max_salary/1000000))*1000000
        player_data["status"]={}
        player_data["status"]["condition"]=random.choice(["A","B","C"])
        player_data["status"]["fatigue"]=0
        player_data["status"]["injury"]=random.choices([0,1,2,3], weights=[70, 15, 10, 5], k=1)[0]
        if(player_data["status"]["injury"]>0):
            player_data["status"]["condition"]="C"
        player_data["record"] = create_default_record(player_data["status"]["injury"],player_data["ability"])
        return player_data
    
    def generate_pitcher(self):
        player_data = {}
        player_data["number"] = self.number
        family_name = random.choice(self.name_data["Family_name"])
        given_name = random.choice(self.name_data["Given_name"])
        player_data["name"] = family_name+" "+given_name
        player_data["age"] = random.randint(18,40)
        if random.randint(0,100)<35:
            player_data["hitting_rl"]="L"
        else:
            player_data["hitting_rl"]="R"
        if random.randint(0,100)<20:
            player_data["pitching_rl"]="L"
        else:
            player_data["pitching_rl"]="R"
        player_data["position"] = random.choices(self.pitcher_position,weights=[6, 5, 2])[0]
        ability_list = self.hidden_piching_ability
        ability_score = 0 #年俸は投手能力の合計スコア依存で決める、年俸決定ロジックは後から拡張
        player_data["ability"] = {}
        for ability in ability_list:
            player_data["ability"][ability] = random.randint(1,90)
            ability_score+=player_data["ability"][ability]
        if(ability_score<100):
            max_salary = round((self.max_salary-self.min_salary)*2/10 + self.min_salary)
            min_salary = self.min_salary
        elif(ability_score<200):
            max_salary = round((self.max_salary-self.min_salary)*5/10 + self.min_salary)
            min_salary = round((self.max_salary-self.min_salary)*2/10 + self.min_salary)
        else:
            max_salary = self.max_salary
            min_salary = round((self.max_salary-self.min_salary)*5/10 + self.min_salary)
        player_data["salary"] = random.randint(round(min_salary/1000000),round(max_salary/1000000))*1000000
        player_data["status"]={}
        player_data["status"]["condition"]=random.choice(["A","B","C"])
        player_data["status"]["fatigue"]=0
        player_data["status"]["injury"]=random.choices([0,1,2,3], weights=[70, 15, 10, 5], k=1)[0]
        if(player_data["status"]["injury"]>0):
            player_data["status"]["condition"]="C"
        player_data["record"] = create_default_pitcher_record(player_data["status"]["injury"],player_data["position"],player_data["ability"])
        return player_data
    
