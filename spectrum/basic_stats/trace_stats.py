import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ortho_group
from itertools import product
from tqdm import trange

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process parameters for orthogonal matrices.')
    parser.add_argument('-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('-d', type=int, required=True, help='Dimension of the orthogonal matrices')
    parser.add_argument('-n', type=int, required=True, help='Number of IID orthogonal matrices')
    parser.add_argument('-L', type=int, required=True, help='Length of the word')
    parser.add_argument('-s', type=int, default=1000, help='Number of samples for averaging')
    return parser

def generate_orthogonal_matrices(d, n):
    return ortho_group.rvs(dim=d, size=n)

def create_word(matrices, indices):
    result = np.eye(matrices[0].shape[0])
    for idx in indices:
        result = result @ matrices[idx]
    return result

def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    m1_trace=0
    m2_trace=0
    #all_words = product(range(args.n), repeat=args.L)
    
    m1_trace = np.zeros(args.n**(args.L*2))
    m2_trace = np.zeros(args.n**(args.L*2))


    for _ in trange(args.s):
        matrices = generate_orthogonal_matrices(args.d, args.n)    
        idx=0
        iter_a = product(range(args.n), repeat=args.L)
        #iter_a = [(0, 0)]
        for indices_a in iter_a:          
            word_a = create_word(matrices, indices_a)
            iter_b = product(range(args.n), repeat=args.L)
            #iter_b = [(1, 1)]
            for indices_b in iter_b:
                word_b = create_word(matrices, indices_b)
                #print(indices_a, indices_b)
                tr = np.trace(word_a @ word_b)
                m1_trace[idx] += tr 
                m2_trace[idx] += tr**2 
                idx+=1


    m1_trace /= args.s
    m2_trace /= args.s



    phi1 = m1_trace / args.d
    phi2 = m2_trace  - m1_trace**2
    


    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 10))
    
    plt.scatter(phi1, phi2, alpha=0.5)
    plt.xlabel(r'$E(\text{Tr}(U))/d$')
    plt.ylabel(r'$\text{Var}(\text{Tr}(U))$')
    plt.title(f'Trace Statistics for d={args.d}, n={args.n}, L={args.L}')
    
    plt.tight_layout()
    plt.savefig(f'trace_statistics_d{args.d}_n{args.n}_L{args.L}.png', dpi=600)
    plt.savefig(f'trace_statistics_d{args.d}_n{args.n}_L{args.L}.pdf', dpi=600)
    plt.show()

if __name__ == "__main__":
    main()