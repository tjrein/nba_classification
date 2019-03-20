import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from  matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from data import read_data, standardize_data
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture as GMM

np.set_printoptions(suppress=True)

def get_statistical_profile(orig_stats, group):
    mean_stats = orig_stats.mean(axis=0)
    indices = [ obj['index'] for obj in group]
    stats = [ orig_stats[index] for index in indices]
    group_stats = np.vstack(stats)
    mean_group = group_stats.mean(axis=0)
    index = np.arange(19)
    bar_width = 0.35

    fig2 = plt.figure(figsize=(7, 3.5))
    ax = fig2.add_subplot(111)

    ax.bar(index, mean_group[2:21], bar_width, color='g', label='Group Avgs') 
    ax.bar(index + bar_width, mean_stats[2:21], bar_width, color='y', label='Leage Avgs') 

    ax.set_xticks(index + bar_width / 2.0)
    xticklabels = [
        '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'EFG%',
        'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PPG'
    ]

    ax.set_xticklabels(xticklabels)
    ax.legend()
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

def get_labels():
    red = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="red", label="PG")
    blue = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="blue", label="SG")
    yellow = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="yellow", label="SF")
    cyan = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="cyan", label="PF")
    green = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="green", label="C")

    return [ red, blue, yellow, cyan, green]

def compute_aic(test):
    aic = []
    lowest_aic = np.infty
    n_components_range = range(5, 16)
    cv_types = ['spherical', 'tied', 'diag', 'full']

    for i, cv_type in enumerate(cv_types):
        for n_components in n_components_range:
            gmm = GMM(random_state=0, n_components=n_components, covariance_type=cv_type)
            gmm.fit(test)
            aic.append(gmm.aic(test))
            if aic[-1] <= lowest_aic:
                lowest_aic = aic[-1]
                best_gm = gmm
    print(best_gm)
    return best_gm

def compute_silhouette_gmm(test):
    for i in range(2, 21):
        gmm = GMM(n_components=i, random_state=0).fit(test)
        gmm_labels = gmm.predict(test)
        probs = gmm.predict_proba(test)
        size = 10 * probs.max(1) ** 2
        gmm_avg = silhouette_score(test, gmm_labels)
        print("Silhouette score: ", i, gmm_avg)

def perform_lda(stats, classes):
    lda = LDA(n_components=2, solver='eigen', shrinkage='auto')
    lda.fit(stats, classes)
    result = lda.transform(stats)
    return result

def perform_pca(stats):
    pca = PCA(n_components=2)
    pca.fit(stats)
    result = pca.transform(stats)
    return result

def perform_pca_plus_lda(stats, classes):
    pca = PCA(n_components=0.95, svd_solver='full')
    pcs = pca.fit_transform(stats)
    result = perform_lda(pcs, classes)
    return result

def get_gmm(data): 
    gmm = compute_aic(data) 
    clusters = gmm.predict(data)
    probs = gmm.predict_proba(data)
    return (clusters, probs)


def assign_groups(clusters, data, names):
    groups = {}
    for i, obs in enumerate(data):
        cluster = clusters[i]

        if not cluster in groups:
            groups[cluster] = []

        groups[cluster].append({'name': names[i], 'index': i })

    return groups

def plot_results(data, t1, names):
    fig1 = plt.figure(figsize=(3.5, 3.5))
    ax = fig1.add_subplot(111)
    for i, obs in enumerate(data):
        ax.scatter(obs[0], obs[1], c=get_t1_colors(t1[i]), s=5, label=t1[i])

     #Uncomment to annotate star players
     #   if names[i] in ["Giannis Antetokounmpo", "Kevin Durant", "James Harden", "LeBron James"]:
     #       ax.annotate(names[i], (obs[0], obs[1]))

    handles = get_labels()
    ax.legend(handles=handles, bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

    fig2 = plt.figure(figsize=(3.5, 3.5))
    ax = fig2.add_subplot(111)
    clusters, probs = get_gmm(data)
    size = 20 * probs.max(1) ** 2
    ax.scatter(data[:,0], data[:,1], c=clusters, cmap='viridis', s=size)

