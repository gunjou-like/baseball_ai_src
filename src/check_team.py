import json
import argparse

def main():
    positions={"SP":0,"RP":0,"CP":0,"C":0,"1B":0,"2B":0,"SS":0,"3B":0,"LF":0,"CF":0,"RF":0}

    parser = argparse.ArgumentParser(description="チームチェックスクリプト")
    parser.add_argument("team_data_path",type=str,help="team_data_file")

    args=parser.parse_args()
    with open(args.team_data_path, 'r', encoding='utf-8') as f:
        data=json.load(f)
    
    print("loaded")

    for key,value in data.items():
        #print(value["position"])
        for p,num in positions.items():
            if(p==value["position"]):
                positions[p]+=1

    print(positions)
if __name__=="__main__":
    main()