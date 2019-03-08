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
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering as AGGC
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

def get_statistical_profile(stats, groups):
    group = groups[3]
    mean_stats = stats.mean(axis=0)
    print(mean_stats)
    #print(group)

    stats = [obj['stats'] for obj in group]
    group_stats = np.vstack(stats)
    mean_group = group_stats.mean(axis=0)

    fig = plt.figure(2)

def get_t1_colors(position):
    color = {
        "SS": 'm',
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
        "PG": 'r',
        "W": 'b',
        "PF": 'c',
        "C": 'g'
    }[position]

    return color

def compute_bic(test):
    bic = []
    lowest_bic = np.infty
    n_components_range = range(6, 16)
    cv_types = ['spherical', 'tied', 'diag', 'full']

    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = GMM(random_state=0, n_components=n_components, covariance_type=cv_type)
            gmm.fit(test)
            bic.append(gmm.bic(test))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gm = gmm


    print("best_gm", best_gm)
    print("bic", bic)

def compute_agg_score(test):
    for i in range(2, 21):
        ward = AGGC(n_clusters=i, linkage='ward').fit(test)
        ward_labels = ward.labels_
        ward_avg = silhouette_score(test, ward_labels)
        print("Ward avg", i, ward_avg)

def compute_silhouette_gmm(test):
    for i in range(2, 21):
        gmm = GMM(n_components=i, random_state=0).fit(test)
        gmm_labels = gmm.predict(test)
        probs = gmm.predict_proba(test)
        size = 10 * probs.max(1) ** 2
        gmm_avg = silhouette_score(test, gmm_labels)
        print("Silhouette score: ", i, gmm_avg)

def main():
    names, stats, t1, t2, t3, t4 = read_data()
    #stats = standardize_data(stats)

    pca = PCA(n_components=6)
    pca.fit(stats)
    pcs = pca.transform(stats)
    
    #mds = MDS(n_components=5, random_state=0)
    #test = mds.fit_transform(stats)

    #lda = LDA(n_components=2, solver='eigen')
    #lda.fit(stats, t2)
    #test = lda.transform(stats)

    #mds = MDS(n_components=2, random_state=0)
    #pcs = mds.fit_transform(stats)

    #isomap = Isomap(n_neighbors=8)
    #pcs = isomap.fit_transform(stats)

    lda = LDA(n_components=2, solver='eigen')
    lda.fit(pcs, t1)
    test = lda.transform(pcs)

    #tsne = TSNE(random_state=0)
    #test = tsne.fit_transform(stats)
    
    ax = plt.subplot(211)
    for i, obs in enumerate(test):
        ax.scatter(obs[0], obs[1], c=get_t1_colors(t1[i]), s=5)

        #if names[i] in ["Giannis Antetokounmpo", "Kevin Durant", "Jarrett Jack", "LeBron James"]:
           # ax.annotate(names[i], (obs[0], obs[1]))
            

    #ax = plt.subplot(312)
    #kmeans = KMeans(8, random_state=0)
    #labels = kmeans.fit_predict(test)
    #plt.scatter(test[:,0], test[:,1], c=labels, cmap='viridis', s=2)

    #ward = AGGC(n_clusters=10, linkage='ward').fit(test)
    #ward_labels = ward.labels_
    #plt.scatter(test[:,0], test[:,1], c=ward_labels, cmap='viridis', s=2)
    #ward_avg = silhouette_score(test, ward_labels)

    #compute_silhouette_gmm(test)
    #compute_agg_score(test)
    compute_bic(test)

    ax = plt.subplot(212)
    gmm = GMM(n_components=6, random_state=0, covariance_type='spherical').fit(test)
    gmm_labels = gmm.predict(test)
    probs = gmm.predict_proba(test)
    size = 10 * probs.max(1) ** 2
    plt.scatter(test[:,0], test[:,1], c=gmm_labels, cmap='viridis', s=size)
    #gmm_avg = silhouette_score(test, gmm_labels)
    #sample_silhouette_values = silhouette_samples(test, ward_labels)

    #print("ward", ward_avg)
    #print("gmm", gmm_avg)

    groups = {}
    for i, obs in enumerate(test):
        label = gmm_labels[i]

        if not label in groups:
            groups[label] = []

        groups[label].append({'name': names[i], 'index': i })

    for key, val in groups.items():
        print("Group", key)
        print("\n")
        print(val)


    #get_statistical_profile(stats, groups)

    plt.show()

if __name__ == "__main__":
    main()

