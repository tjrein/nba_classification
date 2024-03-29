import numpy as np
import csv

def filter_low_std(data):
    std = data.std(axis=0, ddof=1)
    remove = [ i for i, val in enumerate(std) if val == 0 ]
    data = np.delete(data, remove, 1)

    return data

def standardize_data(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0, ddof=1)

    return (matrix - mean) / std

def read_data():
    names = []
    player_stats = []
    targets_1 = []
    targets_2 = []
    targets_3 = []
    targets_4 = []

    with open('./data/nba_2017_combined.csv', newline='') as csvfile:
        nbareader = csv.reader(csvfile)

        next(nbareader)

        superstars = ["LeBron James", "Kawhi Leonard", "Kevin Durant", "Giannis Antetokounmpo", "Steph Curry", "Russell Westbrook", "James Harden"]

        for row in nbareader:
            name = row[1].split("\\")[0]
            position = row[2].split('-')[0]
            games_played = row[5]
            games_started = row[6]
            minutes_played = row[30]

            stats = row[5:]

            if name not in names and int(minutes_played) >= 500:
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

                if position in ["PG"]:
                    targets_4.append("PG")
                elif position in ["SG", "SF", "PF"]:
                    targets_4.append("W")
                elif position in ["C"]:
                    targets_4.append("C")


    player_stats = np.array(player_stats)
    player_stats = player_stats.astype(np.float)
    remove = [0, 1, 2, 3, 25]
    player_stats = np.delete(player_stats, remove, 1)

    return (names, player_stats, targets_1, targets_2, targets_3, targets_4)
