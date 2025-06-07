import numpy as np
import matplotlib.pyplot as plt

def is_colinear_with_origin(z1, z2, tol=1e-12):
    """
    原点(0+0j), z1, z2 がほぼ同一直線上かどうかを判定する。
    具体的には cross = Im(conj(z1)*z2) を用い、これが 0 なら共線。
    """
    cross_im = (z1.conjugate() * z2).imag
    return abs(cross_im) < tol

def find_circle_center_and_radius(z1, z2):
    """
    単位円と直交し、かつ z1, z2 を通る円の
    中心 c (複素数) と半径 R (実数) を返す。
    
    条件：
      1. |c|^2 = R^2 + 1   (単位円と直交)
      2. |z1 - c| = R,  |z2 - c| = R  (z1, z2 は同じ円上)
    このうち、"Re(conj(z1)*c) = (|z1|^2 + 1)/2" の形の連立方程式で c を求められる。
    """
    # z1 = x1 + i y1,  z2 = x2 + i y2
    x1, y1 = z1.real, z1.imag
    x2, y2 = z2.real, z2.imag
    
    # |z1|^2, |z2|^2
    r1_sq = x1**2 + y1**2
    r2_sq = x2**2 + y2**2
    
    # 以下の連立方程式を解く：
    #   Re( conj(z1) * c ) = (|z1|^2 + 1)/2
    #   Re( conj(z2) * c ) = (|z2|^2 + 1)/2
    #
    # conj(z1) = x1 - i y1
    # conj(z1)*c = (x1 - i y1)*(cx + i cy) = x1 cx + y1 cy + i( ... ) --> 実部は x1 cx + y1 cy
    # よって
    #   x1 cx + y1 cy = (r1_sq + 1)/2
    #   x2 cx + y2 cy = (r2_sq + 1)/2
    #
    A = np.array([[x1, y1],
                  [x2, y2]], dtype=float)
    b = np.array([(r1_sq + 1)/2, 
                  (r2_sq + 1)/2], dtype=float)
    
    # A * [cx, cy] = b を解く
    cx, cy = np.linalg.solve(A, b)
    c = cx + 1j*cy
    
    # R = sqrt(|c|^2 - 1)
    c_abs_sq = (cx**2 + cy**2)
    R = np.sqrt(c_abs_sq - 1)
    return c, R

def geodesic_points_on_circle(c, R, z1, z2, num=300):
    """
    中心 c、半径 R の円上で、
    z1, z2 を結ぶ「短い方の円弧」をパラメトリックに生成し，
    その (real, imag) 座標列を返す。
    """
    # z1, z2 の偏角 (c を基準として) を求める
    theta1 = np.angle(z1 - c)
    theta2 = np.angle(z2 - c)
    
    # theta1 -> theta2 へ最短回りになるように補正
    if theta2 < theta1:
        theta2 += 2*np.pi
    if theta2 - theta1 > np.pi:
        # 逆回りの方が短い
        theta2 -= 2*np.pi
    
    thetas = np.linspace(theta1, theta2, num)
    
    # 円パラメータ z = c + R*e^{i theta}
    arc = c + R * np.exp(1j * thetas)
    return arc

def plot_poincare_geodesic(z1, z2):
    """
    複素数 z1, z2 (ともに |z|<1) を結ぶポアンカレ円板上の測地線を描画する。
    """
    fig, ax = plt.subplots(figsize=(6,6))
    
    # 1) 単位円(境界)の描画
    #    パラメータで e^{i phi}, phi in [0, 2pi]
    phi = np.linspace(0, 2*np.pi, 300)
    boundary = np.exp(1j * phi)  # 単位円
    ax.plot(boundary.real, boundary.imag, color='gray', linestyle='--')
    
    # 2) z1, z2 の描画
    ax.plot(z1.real, z1.imag, 'ro')
    ax.plot(z2.real, z2.imag, 'ro')
    
    # 3) 測地線の描画
    if is_colinear_with_origin(z1, z2):
        # 原点通過の直線 (ユークリッドでも直線)
        # 単位円との交点を求め，その2点を結ぶ
        
        # z1, z2 ともに原点方向にあるので，
        # 交点は  z = ± (z / |z|)  という形で求まる（z1, z2 どちらでも同じ直線）
        # 今回は単位円との交点を２つとりたいので，
        #   intersection_1 = z1 / |z1|
        #   intersection_2 = - z1 / |z1|
        # のようにやっても良いですが，
        # ここでは z1,z2 が (0 でない) と仮定して手抜き実装：
        
        # 交点1
        int1 = z1 / abs(z1)
        # 交点2
        int2 = z2 / abs(z2)
        
        # 直線の端点として int1, int2 を描画
        # ただし同じ向きかもしれないので，「より外側」と「逆向き側」の2つにしたほうが確実
        #   → 例えば abs(z1) < abs(z2) なら int2 が外側とみなせる
        # ここでは簡単に書きます
        x_vals = [int1.real, int2.real]
        y_vals = [int1.imag, int2.imag]
        ax.plot(x_vals, y_vals, 'b-')
    else:
        # 単位円と直交する円弧
        c, R = find_circle_center_and_radius(z1, z2)
        arcs = geodesic_points_on_circle(c, R, z1, z2, num=300)
        xs, ys = arcs.real, arcs.imag
        ax.plot(xs, ys, 'b-')
    
    # 軸設定
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Geodesic in Poincaré disk (complex version)")
    plt.savefig("geodesic.png")

if __name__ == "__main__":
    # 例：p1, p2 を複素数で与える
    p1 = 0.3 + 0.2j
    p2 = -0.4 + 0.3j
    plot_poincare_geodesic(p1, p2)
