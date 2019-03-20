import pandas as pd

pergame = pd.read_csv("./data/nba_2017.csv")
extra_cols = ["MP", "PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "AST%", "STL%", 
              "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS", "WS/48", "OBPM", "DBPM", "BPM", "VORP"]
advanced = pd.read_csv("./data/nba_2017_advanced.csv", usecols=extra_cols)
combined = pd.concat([pergame, advanced], axis=1)
combined.to_csv('./data/nba_2017_test_combined.csv', index=False)

