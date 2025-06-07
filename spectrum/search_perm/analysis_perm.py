import numpy as np
from collections import defaultdict, Counter

def analyze_permutations(perms):
    """Analyze common patterns in a list of permutations"""
    perms = [tuple(p) for p in perms]
    n = len(perms[0])
    
    # 各位置での要素の出現頻度
    position_counts = [Counter(p[i] for p in perms) for i in range(n)]
    
    # 隣接ペアの分析
    adjacent_pairs = defaultdict(int)
    for perm in perms:
        for i in range(n-1):
            adjacent_pairs[(perm[i], perm[i+1])] += 1
    
    # 各要素の相対位置関係
    relative_positions = defaultdict(list)
    for perm in perms:
        for i in range(n):
            for j in range(i+1, n):
                relative_positions[(perm[i], perm[j])].append(j-i)
    
    # 固定位置の検出
    fixed_positions = {
        i: val for i, counts in enumerate(position_counts)
        for val, count in counts.items()
        if count == len(perms)
    }
    
    return {
        'position_counts': position_counts,
        'adjacent_pairs': dict(adjacent_pairs),
        'relative_positions': dict(relative_positions),
        'fixed_positions': fixed_positions
    }

# 入力データ
permutations = [
(6, 4, 1, 2, 5, 3, 7, 0),
(6, 1, 4, 2, 5, 7, 0, 3),
(6, 1, 4, 2, 5, 3, 7, 0),
(6, 4, 1, 2, 0, 7, 5, 3),
(6, 1, 4, 2, 0, 3, 5, 7),
(6, 4, 1, 2, 0, 3, 7, 5),
(6, 1, 4, 2, 0, 7, 3, 5),
(6, 4, 1, 2, 3, 5, 7, 0),
(6, 1, 4, 2, 5, 7, 3, 0),
(6, 4, 1, 2, 3, 0, 7, 5),
(6, 4, 1, 2, 3, 5, 0, 7),
(6, 1, 4, 2, 3, 5, 7, 0),
(6, 4, 1, 2, 5, 3, 0, 7),
(6, 4, 1, 2, 5, 7, 3, 0),
(6, 1, 4, 2, 5, 3, 0, 7),
(6, 4, 1, 2, 5, 7, 0, 3),
(6, 1, 4, 2, 0, 5, 7, 3),
(6, 1, 4, 2, 3, 7, 5, 0),
(6, 1, 4, 2, 0, 5, 3, 7),
(6, 4, 1, 2, 3, 7, 0, 5),
(6, 1, 4, 2, 0, 3, 7, 5),
(6, 1, 4, 2, 5, 0, 3, 7),
(6, 4, 1, 2, 0, 3, 5, 7),
(6, 4, 1, 2, 0, 5, 7, 3),
(6, 4, 1, 2, 3, 7, 5, 0),
(6, 1, 4, 2, 5, 0, 7, 3),
]

# 分析実行
analysis = analyze_permutations(permutations)

# 結果出力
print("Fixed Positions Analysis:")
for pos, val in analysis['fixed_positions'].items():
    print(f"Position {pos}: Always contains {val}")

print("\nCommon Patterns:")
print("1. Start of permutation:", analysis['position_counts'][0])
print("2. Second position frequencies:", analysis['position_counts'][1])
print("3. Third position frequencies:", analysis['position_counts'][2])
print("4. Fourth position frequencies:", analysis['position_counts'][3])