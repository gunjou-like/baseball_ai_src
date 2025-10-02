"""
チームのデータを生成。
"""
import argparse
import os
import json
from generate_player import *

def insert_str_into_filename(path_str: str, insert_str: str) -> str:
    """
    ファイルパスのファイル名に指定文字列を挿入する。
    例: logs/data.json + "_backup" → logs/data_backup.json
    """
    # ディレクトリとファイル名に分割
    dir_name, file_name = os.path.split(path_str)

    # 拡張子とベース名に分割
    base_name, ext = os.path.splitext(file_name)

    # 挿入後のファイル名を生成
    new_file_name = f"{base_name}_{insert_str}{ext}"

    # 新しいパスを結合
    new_path = os.path.join(dir_name, new_file_name)
    return new_path



class TEAM_GENERATOR():
    def __init__(self,team_name="team_A",name_json_path="./name.json"):
        self.fund = 3000000000
        self.num_player = 31
        self.team_name = team_name
        self.name_json_path=name_json_path
    
    def generate_team(self):
        team_data={}
        for num in range(self.num_player):
            id=str(num)
            player_generator=PLAYER_GENERATOR(num,self.name_json_path)
            if(random.randint(0,10)<4):
                data=player_generator.generate_pitcher()
            else:
                data=player_generator.generate_batter()
            team_data[id]=data
        return team_data

def main():
    parser = argparse.ArgumentParser(description="チーム生成スクリプト")
    parser.add_argument("--team_name",type=str,default="team_A",help="チーム名")
    parser.add_argument("--output_json_path",type=str,default="./src/ai/data/team_data.json",help="output_file")
    parser.add_argument("--name_json_path",type=str,default="./name.json",help="name_file")
    
    args=parser.parse_args()
    print("generating team...")
    team_generator=TEAM_GENERATOR(args.team_name,args.name_json_path)
    team_data=team_generator.generate_team()
    output_path=args.output_json_path
    output_path=insert_str_into_filename(output_path,args.team_name)
    with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(team_data, f, ensure_ascii=False, indent=2)
    print("team data saved.")


if __name__ == "__main__":
    main()