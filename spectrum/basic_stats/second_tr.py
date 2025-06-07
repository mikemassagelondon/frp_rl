import numpy as np

def random_orthogonal_matrix(d):
    """
    d x d のランダム実直交行列（Haar分布）を返す関数。
    ガウス分布からサンプリングした行列 X に対して QR 分解を行い、
    R の対角成分の符号を吸収して一様な直交行列を生成します。
    """
    # 標準正規分布から d x d の行列を生成
    X = np.random.normal(loc=0.0, scale=1.0, size=(d, d))
    
    # QR 分解
    Q, R = np.linalg.qr(X)
    
    # R の対角成分の符号を Q に吸収させることで一様な（Haar分布の）直交行列を得る
    # （R の対角成分が 0 にならないことはほぼ確実だが、一応チェックしても良い）
    diag_sign = np.sign(np.diag(R))
    Q = Q * diag_sign
    
    return Q

def simulate_trace_variance(d, num_samples=100000):
    """
    d x d の Haar 直交行列を多数サンプリングし、そのトレースの分散を推定する。
    
    Parameters
    ----------
    d : int
        行列のサイズ
    num_samples : int
        サンプリング数（大きいほど精度が上がるが、計算時間がかかる）
        
    Returns
    -------
    float
        トレースの分散の推定値
    """
    traces = np.zeros(num_samples)
    for i in range(num_samples):
        O = random_orthogonal_matrix(d)
        traces[i] = np.trace(O )
    
    # np.var は標本分散(不偏分散)ではなく母分散（デフォルトは ddof=0）を計算する点に注意
    return np.var(traces, ddof=1)  # 不偏分散を使うなら ddof=1

if __name__ == "__main__":
    d = 200
    num_samples = 2000
    var_tr = simulate_trace_variance(d, num_samples)
    print(f"Dimension d={d}, Sample Size={num_samples}")
    print(f"Estimated Variance of Tr(O): {var_tr}")
