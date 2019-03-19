import numpy as np
import matplotlib.pyplot as plt
from reduce_dimensions import perform_lda, plot_results
from data import read_data, standardize_data

def main():
    names, stats, t1, t2, t3, t4 = read_data()
    raw_stats = stats
    stats = standardize_data(stats)

    results = perform_lda(stats, t1)
    plot_results(results, t1)

    plt.tight_layout()
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
