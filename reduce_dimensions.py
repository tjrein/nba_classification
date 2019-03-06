import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from data import read_data

def get_t1_colors(position):
    color = {
        "PG": 'r',
        "SG": 'b',
        "SF": 'y',
        'PF': 'c',
        'C': 'g'
    }[position]
    
    return color

def get_t2_colors(position):
    color = {
        'G': 'r',
        'F': 'b',
        'C': 'g'
    }[position]

    return color

def get_t3_colors(position):
    color = {
        "PG": 'r',
        "W": 'b',
        "B": 'g'
    }[position]

    return color

def get_t4_colors(position):
    color = {
        "BC": 'r',
        "FC": 'b'
    }

    return color

def main():
    names, stats, t1, t2, t3, t4 = read_data()

    
    LDA
    lda = LDA(n_components=2)
    lda.fit(stats, t3)
    test = lda.transform(stats)
    
    #pca = PCA(n_components=2)
    #pca.fit(stats)
    #test = pca.transform(stats)

    ax = plt.subplot(111)
    for i, obs in enumerate(test):
        ax.scatter(obs[0], obs[1], c=get_t3_colors(t3[i]))

    #plt.plot([1, 2, 3, 4])
    plt.show()



if __name__ == "__main__":
    main()

