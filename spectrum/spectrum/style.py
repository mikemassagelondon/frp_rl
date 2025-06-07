import matplotlib.pyplot as plt

def set_paper_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 24,
        'axes.titlesize': 28,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.figsize': (16, 16)
    })

