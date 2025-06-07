from orthogonal_curve import plot_orthogonal_circle_in_unit
import numpy as np
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

fig, ax = plt.subplots(figsize=(10, 10))
set_paper_style()

# 論文用の見やすい2色を設定
color1 = '#1f77b4'  # 青系
color2 = '#ff7f0e'  # オレンジ系

ax.plot([-1,1], [0,0], color=color2)
ax.plot([0,0], [-1,1], color=color1)

def _get_pair(base, diff):
    A = np.array([np.cos(base-diff), np.sin(base-diff)])
    B = np.array([np.cos(base+diff), np.sin(base+diff)])
    return A, B    

def _draw(fig, ax, m):
    base = np.pi/int(m)
    diff = np.pi/int(3*m)
    for n in range(2*m):
        A, B = _get_pair(n*base, diff)
        color = color1 if n % 2 == 0 else color2
        fig, ax = plot_orthogonal_circle_in_unit(fig, ax, A, B, arc_color=color, reverse_direction=False)


depth = 7
m=2
for ell in range(depth):
    _draw(fig, ax, m)
    m*=3


# プロットだけを表示するための設定
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.axis('off')  # 軸を非表示にする

# 保存時にDPIを指定できるようにする
dpi = 300  # 任意のDPI値を設定

plt.savefig("free_hyperbolic.png", dpi=dpi, bbox_inches='tight', pad_inches=0)
plt.savefig("free_hyperbolic.pdf", dpi=dpi, bbox_inches='tight', pad_inches=0)