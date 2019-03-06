import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from data import read_data, standardize_data
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture as GMM


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
    }[position]

    return color

def main():
    names, stats, t1, t2, t3, t4 = read_data()
    stats = standardize_data(stats)
    
    lda = LDA(n_components=2)
    lda.fit(stats, t3)
    test = lda.transform(stats)
    
    #pca = PCA(n_components=2)
    #pca.fit(test)
    #test = pca.transform(test)
    
    ax = plt.subplot(311)
    for i, obs in enumerate(test):
        ax.scatter(obs[0], obs[1], c=get_t3_colors(t3[i]), s=2)

        if names[i] in ["Russell Westbrook","Ben Simmons", "LeBron James", "Kawhi Leonard"]:
            ax.annotate(names[i], (obs[0], obs[1]))

    ax = plt.subplot(312)
    kmeans = KMeans(3, random_state=0)
    labels = kmeans.fit_predict(test)
    plt.scatter(test[:,0], test[:,1], c=labels, cmap='viridis', s=2)

    ax = plt.subplot(313)
    gmm = GMM(n_components=3).fit(test)
    gmm_labels = gmm.predict(test)
    probs = gmm.predict_proba(test)
    size = 10 * probs.max(1) ** 2
    print(size)
    plt.scatter(test[:,0], test[:,1], c=gmm_labels, cmap='viridis', s=size)
    silhouette_avg = silhouette_score(test, gmm_labels)
    print(silhouette_avg)

    plt.show()



if __name__ == "__main__":
    main()

