import numpy as np
import sys
import pprint
from reduce_dimensions import get_gmm, perform_pca, perform_lda, perform_pca_plus_lda
from data import read_data, standardize_data

np.set_printoptions(precision=3, suppress=True)

def function_wrapper(method):
    func = {
        'pca': perform_pca,
        'lda': perform_lda,
        '+': perform_pca_plus_lda
    }[method]

    return func

def main():
    args = sys.argv
    names, stats, t1, t2, t3, t4 = read_data()
    stats = standardize_data(stats)
    lc_names = [ name.lower() for name in names ]

    method = perform_pca
    results = []

    if len(args) > 1:
        if args[1] in ['lda', '+']:
            method = function_wrapper(args[1])
            results = method(stats, t1)

    if not len(results):
        results = method(stats)

    clusters, probs = get_gmm(results)

    if len(args) > 2:
        name_arg = args[2].lower()
        index = lc_names.index(name_arg)
        names = [names[index]]
        probs = [probs[index]]
    
    for i, name in enumerate(names):
        test = np.array2string(probs[i], formatter={'float_kind':lambda x: "%.3f" % x})
        print('\n' + name + ': ' + test + '\n')

if __name__ == '__main__':
    main()
