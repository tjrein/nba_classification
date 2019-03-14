import numpy as np
import matplotlib.pyplot as plt
from reduce_dimensions import perform_pca_plus_lda, plot_results
from data import read_data, standardize_data

def main():
    names, stats, t1, t2, t3, t4 = read_data()
    raw_stats = stats
    stats = standardize_data(stats)

    results = perform_pca_plus_lda(stats, t1)
    plot_results(results, t1)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
