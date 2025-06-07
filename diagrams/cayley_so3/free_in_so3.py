#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


# 二元生成自由群を生成
def generate_group_with_edges(initial_point, L, color_map, rep):
    """Generate vertices and edges along given representation of the free group F_2.
    
    Args:
        initial_point (X): The point for the identity.
        L (Int): The depth of the Cayley graph
        color_map (dict): 
        rep (dict[X to X]): The representation of generators (and inverse) of F_2.

    Returns:
        group (List[Str]): The list of generated words
        edges (List[X], List[X], color):  The tuple of (start pt, end pt, color) 
        points (List[X]): The list of vertices
    """
    
    group = {''}  # 恒等変換
    edges = []  # エッジ情報を保持
    current_level = {''}
    points = {'' : initial_point}  # 各語に対応する点（ポアンカレ円盤上）

    for _ in range(L):
        next_level = set()
        for word in current_level:
            z_start = points[word]
            for gen in ['A', 'B', 'a', 'b']:
                #if len(word) ==0 and gen != "A": continue
                if not (len(word) > 0 and ((word[-1] == 'A' and gen == 'a') or 
                                           (word[-1] == 'a' and gen == 'A') or 
                                           (word[-1] == 'B' and gen == 'b') or 
                                           (word[-1] == 'b' and gen == 'B'))):
                    new_word = word + gen
                    z_end = rep[gen](z_start)
                    next_level.add(new_word)
                    group.add(new_word)
                    points[new_word] = z_end
                    # エッジ情報を追加 (始点, 終点, 色)
                    edges.append((z_start, z_end, color_map[gen]))
        current_level = next_level
    
    return group, edges, points


def rotation_matrix_z(angle: float) -> np.ndarray:
    """
    z軸まわりに angle 回転させる3x3行列を返す
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [ c, -s,  0],
        [ s,  c,  0],
        [ 0,  0,  1]
    ])

def rotation_matrix_x(angle: float) -> np.ndarray:
    """
    x軸まわりに angle 回転させる3x3行列を返す
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ])

def apply_rotation(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    3x3回転行列 R を、ベクトル v (shape=(3,)) に適用した結果を返す
    """
    return R @ v

def is_reduced(word: list) -> bool:
    """
    与えられた (文字列) のリスト word が自由群上の既約語か判定する。
    例えば、末尾が a と a^-1 のような連続はだめ。
    """
    for i in range(len(word) - 1):
        # 生成元 a と a^-1 が隣り合ってはいけない
        if (word[i] == 'a' and word[i+1] == 'A') or (word[i] == 'A' and word[i+1] == 'a'):
            return False
        # 生成元 b と b^-1 が隣り合ってはいけない
        if (word[i] == 'b' and word[i+1] == 'B') or (word[i] == 'B' and word[i+1] == 'b'):
            return False
    return True

def slerp(c1: np.ndarray, c2: np.ndarray, num_points: int = 20) -> np.ndarray:
    """
    Spherical Linear Interpolation (Slerp) で c1, c2 (ともに単位ベクトル) の間を補間し、
    球面上の大円弧の座標列を返す。
    num_points は補間点数。
    """
    # 内積から角度を求める
    dot = np.dot(c1, c2)
    # 数値誤差で ±1 を超えることがあるのでクリップ
    dot = max(-1.0, min(1.0, dot))
    omega = np.arccos(dot)

    if abs(omega) < 1e-12:
        # ほぼ同じ方向の場合は、ほぼ同じ点として扱う
        return np.array([c1 for _ in range(num_points)])
    
    t_list = np.linspace(0, 1, num_points)
    points = []
    for t in t_list:
        # (sin((1-t) * omega)/sin(omega)) * c1 + (sin(t * omega)/sin(omega)) * c2
        part1 = np.sin((1 - t)*omega)/np.sin(omega) * c1
        part2 = np.sin(t*omega)/np.sin(omega) * c2
        p = part1 + part2
        # 念のため正規化
        p = p / np.linalg.norm(p)
        points.append(p)
    return np.array(points)

def set_aspect_equal_3d(ax):
    """
    x, y, z 軸のスケールを同一にする。
    Matplotlib バージョンによっては ax.set_box_aspect([1,1,1]) が使えるが、
    ここでは後方互換のため手動で設定する。
    """
    # 各軸の min, max を取得
    ranges = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    # それぞれの軸の中心と長さを取得
    centers = np.mean(ranges, axis=1)
    max_range = 0.5 * np.max(ranges[:,1] - ranges[:,0])

    # 軸を同じ大きさに設定
    ax.set_xlim3d(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim3d(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim3d(centers[2] - max_range, centers[2] + max_range)




def main():
    parser = argparse.ArgumentParser(
        description="二元生成自由群による球面上のケーリーグラフを描画し、PNGで出力するスクリプト (測地線版)")
    parser.add_argument("--L", type=int, default=3,
                        help="生成する既約語の最大長 (default: 3)")
    parser.add_argument("--outfile", type=str, default="free_group_cayley_geodesic.png",
                        help="出力PNGファイル名 (default: free_group_cayley_geodesic.png)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="出力PNGのdpi (default: 300)")
    args = parser.parse_args()

    L = args.L
    outfile = args.outfile
    dpi = args.dpi

    # cos(theta) = 1/3 をみたす角 theta
    theta = math.acos(1.0 / 3.0)

    # 回転行列の定義: a, b, A, B
    R_a  = rotation_matrix_z(theta)
    R_ai = rotation_matrix_z(-theta)  # A
    R_b  = rotation_matrix_x(theta)
    R_bi = rotation_matrix_x(-theta)  # B

    # 生成元 -> 回転行列 の辞書
    rotation_dict = {
        'a': lambda x: R_a @ x,
        'A': lambda x: R_ai @ x,
        'b': lambda x: R_b @ x,
        'B': lambda x: R_bi @ x
    }

    # エッジの色付け用に seaborn のカラーパレットを利用 (4色)
    palette = sns.color_palette("Set2", 4)
    color_map = {
        'a':      palette[0],
        'A': palette[1],
        'b':      palette[2],
        'B': palette[3]
    }


    group, edges, points=generate_group_with_edges(
        initial_point=np.array([0,1,0]), 
        L=L, 
        color_map=color_map, 
        rep=rotation_dict)    


    # ここから描画
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_title("Cayley graph on S^2 (geodesic edges, up to length = {})".format(L))

    # 頂点を描く (球面上の点)
    for p in points.values():
        ax.scatter(p[0], p[1], p[2], color="black", s=4)

    # エッジを測地線 (大円弧) として描画
    for start, end, color in edges:
        # c1, c2 の間を Slerp で補間して大円弧を描く
        arc_points = slerp(start, end, num_points=20)
        ax.plot(arc_points[:,0], arc_points[:,1], arc_points[:,2],
                color=color, linewidth=2)

    # 軸のアスペクト比を同じにする
    set_aspect_equal_3d(ax)

    # 軸ラベルなど調整
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # 好みで枠を消す場合は以下を使用:
    # ax.set_axis_off()

    # カメラ視点を変更する (elev=, azim=)
    ax.view_init(elev=45, azim=45)

    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved to {outfile} with dpi={dpi}")

if __name__ == "__main__":
    main()
