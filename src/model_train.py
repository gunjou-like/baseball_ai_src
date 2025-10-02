import json
from ai.ai_strategy import AI_STRATEGY
from ai.ai_order import AI_ORDER
from gameenv import BaseballEnv
from team import Team, encode_team  # チームベクトル化関数（要実装）

# AIインスタンス生成
ai_strategy_a = AI_STRATEGY()
ai_strategy_b = AI_STRATEGY()
ai_order_a = AI_ORDER()
ai_order_b = AI_ORDER()

# チームデータ読み込み
with open("./data/team_data_team_A.json", 'r', encoding='utf-8') as f:
    team_a_data = json.load(f)
with open("./data/team_data_team_B.json", 'r', encoding='utf-8') as f:
    team_b_data = json.load(f)

for episode in range(10000):
    # チームベクトル化（AI_ORDER用）
    team_vec_a = encode_team(team_a_data)
    team_vec_b = encode_team(team_b_data)

    # スタメン選定（AI_ORDERによる）
    lineup_indices_a = ai_order_a.generate_lineup(team_vec_a)
    lineup_indices_b = ai_order_b.generate_lineup(team_vec_b)

    # チーム構築
    team_a = Team(team_a_data,"team_A")
    team_b = Team(team_b_data,"team_B")
    team_a.set_lineup(lineup_indices_a)
    team_a.select_SP()
    team_b.set_lineup(lineup_indices_b)
    team_b.select_SP()


    # 試合環境生成
    env = BaseballEnv(team_a, team_b)
    state = env.reset()

    # 試合進行（AI_STRATEGYによる逐次判断）
    while not env.done:
        if env.state.half == "top":
            action = ai_strategy_a.select_action(state)
        else:
            action = ai_strategy_b.select_action(state)

        next_state, reward, done = env.step(action)

        if env.state.half == "top":
            ai_strategy_a.store_experience(state, action, reward, next_state, done)
            ai_strategy_a.train()
        else:
            ai_strategy_b.store_experience(state, action, reward, next_state, done)
            ai_strategy_b.train()

        state = next_state

    # 試合終了後：スタメン選定への報酬反映（例：勝利 +10、得点 +1/点）
    final_reward_a = env.get_team_reward("A")  # 要実装：勝利・得点などから報酬算出
    final_reward_b = env.get_team_reward("B")

    ai_order_a.store_experience(team_vec_a["team_features"], lineup_indices_a, final_reward_a, team_vec_a["team_features"], True)
    ai_order_b.store_experience(team_vec_b["team_features"], lineup_indices_b, final_reward_b, team_vec_b["team_features"], True)

    ai_order_a.train()
    ai_order_b.train()

    # ターゲットネット同期（定期）
    if episode % 10 == 0:
        ai_strategy_a.update_target()
        ai_strategy_b.update_target()