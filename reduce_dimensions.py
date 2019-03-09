import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
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

np.set_printoptions(suppress=True)

def get_statistical_profile(orig_stats, groups):
    group = groups[1]
    diff_group = groups[6]
    mean_stats = orig_stats.mean(axis=0)

    indices = [ obj['index'] for obj in group]
    diff_indices = [ obj['index'] for obj in diff_group]

    stats = [ orig_stats[index] for index in indices]
    diff_stats = [ orig_stats[j] for j in diff_indices]

    group_stats = np.vstack(stats)
    diff_group_stats = np.vstack(diff_stats)

    mean_group = group_stats.mean(axis=0)
    diff_mean_group = diff_group_stats.mean(axis=0)
    
    index = np.arange(21)

    bar_width = 0.35

    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)

    ax.bar(index, mean_group[0:21], bar_width, color='b', label='Group Avgs') 
    ax.bar(index + bar_width, diff_mean_group[0:21], bar_width, color='r', label='Leage Avgs') 

    ax.set_xticks(index + bar_width / 2.0)
    xticklabels = [
        'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'EFG'
        'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PPG'
    ]

    ax.set_xticklabels(xticklabels)
    fig2.tight_layout()

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
    #fig2 = plt.figure(2)
    lowest_bic = np.infty
    n_components_range = range(7, 16)
    cv_types = ['spherical', 'tied', 'diag', 'full']

    for i, cv_type in enumerate(cv_types):
        subplot = int('41' + str(i + 1))
        #ax = fig2.add_subplot(subplot)
        #ax.set_title(cv_type)
        #plt_bic = []

        for n_components in n_components_range:
            gmm = GMM(random_state=0, n_components=n_components, covariance_type=cv_type)
            gmm.fit(test)

            bic.append(gmm.aic(test))
            #plt_bic.append(gmm.aic(test))

            str_val = str(int(gmm.aic(test)))
            #ax.annotate(str_val, (n_components, gmm.aic(test)))

            if bic[-1] <= lowest_bic:
                lowest_bic = bic[-1]
                best_gm = gmm

        #ax.plot(n_components_range, plt_bic)
        #ax.annotate(plt_bic, (n_components_range, plt_bic))

    print("bic", bic)
    print("lowest bic", lowest_bic)
    print(best_gm)

    return best_gm

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
    orig_stats = stats
    stats = standardize_data(stats)

    fig1 = plt.figure(1)

    pca = PCA(n_components=0.95, svd_solver='full')
    pca.fit(stats)
    pcs = pca.transform(stats)

    #pca = PCA(n_components=2)
    #pca.fit(stats)
    #test = pca.transform(stats)
    
    #mds = MDS(n_components=2, random_state=0)
    #test = mds.fit_transform(stats)

    #lda = LDA(n_components=2, solver='eigen')
    #lda.fit(stats, t2)
    #test = lda.transform(stats)

    #mds = MDS(n_components=2, random_state=0)
    #test = mds.fit_transform(stats)

    #isomap = Isomap(n_neighbors=5)
    #pcs = isomap.fit_transform(stats)

    lda = LDA(n_components=2, solver='eigen', shrinkage='auto')
    lda.fit(pcs, t1)
    test = lda.transform(pcs)


    #tsne = TSNE(random_state=0)
    #test = tsne.fit_transform(stats)
    
    ax = fig1.add_subplot(211)
    for i, obs in enumerate(test):
        ax.scatter(obs[0], obs[1], c=get_t1_colors(t1[i]), s=5)

        if names[i] in ["Giannis Antetokounmpo", "Kevin Durant", "James Harden", "LeBron James"]:
            ax.annotate(names[i], (obs[0], obs[1]))
            
    suggested_gmm = compute_bic(test)

    ax = fig1.add_subplot(212)
    #gmm = compute_bic(test)
    #gmm = GMM(covariance_type='full', n_components=13).fit(test)
    gmm = suggested_gmm
    gmm_labels = gmm.predict(test)
    probs = gmm.predict_proba(test)
    size = 20 * probs.max(1) ** 2
    ax.scatter(test[:,0], test[:,1], c=gmm_labels, cmap='viridis', s=size)

    #compute_agg_score(test)
    #ax = plt.subplot(313)
    #ward = AGGC(n_clusters=7, linkage='ward').fit(test)
    #ward_labels = ward.labels_
    #plt.scatter(test[:,0], test[:,1], c=ward_labels, cmap='viridis', s=2)

    labels = gmm_labels

    groups = {}
    for i, obs in enumerate(test):
        label = labels[i]

        if not label in groups:
            groups[label] = []

        groups[label].append({'name': names[i], 'index': i })

    for key, val in groups.items():
        print("Group", key)
        print("\n")
        print(val)

    get_statistical_profile(orig_stats, groups)
    plt.show()

if __name__ == "__main__":
    main()

