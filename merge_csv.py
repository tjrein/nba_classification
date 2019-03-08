import pandas as pd

a = pd.read_csv("./nba_1982.csv")

extra_cols = ["MP", "PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "AST%", "STL%", 
              "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS", "WS/48", "OBPM", "DBPM", "BPM", "VORP"]

b = pd.read_csv("./nba_1982_advanced.csv", usecols=extra_cols)


c = pd.concat([a, b], axis=1)
c.to_csv('./nba_1982_combined.csv', index=False)

