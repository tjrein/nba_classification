import numpy as np
import requests
import csv

def main():
    names = []
    player_stats = []
    targets_1 = []
    targets_2 = []
    targets_3 = []
    targets_4 = []

    with open('./nba_pergame.csv', newline='') as csvfile:
        nbareader = csv.reader(csvfile)
        next(nbareader)

        for row in nbareader:
            name = row[1].split("\\")[0]
            position = row[2]
            stats = row[5:]
            if name not in names:
                names.append(name)
                stats = ['0.0' if stat is '' else stat for stat in stats]
                player_stats.append(stats)

                #original_positions
                targets_1.append(position)

                # G, F, C
                if position in ["PG", "SG"]:
                    targets_2.append("G")
                elif position in ["SF", "PF"]:
                    targets_2.append("F")
                else:
                    targets_2.append("C")

                # Point Guard, Wing, Big
                if position in ["PG"]:
                    targets_3.append("PG")
                elif position in ["SG", "SF"]:
                    targets_3.append("W")
                elif position in ["PF", "C"]:
                    targets_3.append("B")

                #Frontcourt, Backcourt
                if position in ["PG", "SG"]:
                    targets_4.append("BC")
                else:
                    targets_4.append("FC")

    player_stats = np.array(player_stats)
    player_stats = player_stats.astype(np.float)

    return (names, player_stats, targets_1, targets_2, targets_3, targets_4)

if __name__ == "__main__":
    main()
