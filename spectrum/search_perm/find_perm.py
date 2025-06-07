from typing import List, Tuple
import itertools

def apply_transposition(perm: List[int], i: int, j: int) -> List[int]:
    """Apply a transposition (swap) at positions i and j"""
    perm = perm.copy()
    perm[i], perm[j] = perm[j], perm[i]
    return perm

def check_prefix_match(perm: List[int], target: List[int], prefix_length: int) -> bool:
    """Check if the first prefix_length elements match the target"""
    return all(perm[i] == target[i] for i in range(prefix_length))

def find_minimal_transpositions(target_prefix: List[int], length: int = 8) -> List[Tuple[int, int]]:
    """
    Find minimal sequence of transpositions to achieve target prefix
    Args:
        target_prefix: Target values for the prefix
        length: Total length of permutation
    Returns:
        List of transpositions (i,j) to apply
    """
    # 初期順列
    initial = list(range(length))
    
    # 探索済みの状態を記録
    visited = {tuple(initial)}
    
    # BFS用のキュー
    # (current_perm, transpositions)
    queue = [(initial, [])]
    
    while queue:
        current_perm, trans_sequence = queue.pop(0)
        
        # 目標の接頭辞と一致するか確認
        if check_prefix_match(current_perm, target_prefix, len(target_prefix)):
            return trans_sequence
        
        # 可能なすべての互換を試す
        for i in range(length):
            for j in range(i+1, length):
                next_perm = apply_transposition(current_perm, i, j)
                next_perm_tuple = tuple(next_perm)
                
                if next_perm_tuple not in visited:
                    visited.add(next_perm_tuple)
                    queue.append((next_perm, trans_sequence + [(i,j)]))
    
    return []  # 解が見つからない場合

def print_transposition_sequence(initial: List[int], transpositions: List[Tuple[int, int]]):
    """Print the sequence of transpositions and resulting permutations"""
    current = initial.copy()
    print(f"Initial: {current}")
    
    for i, (a, b) in enumerate(transpositions, 1):
        current = apply_transposition(current, a, b)
        print(f"After transposition {i} ({a},{b}): {current}")

# 目標の接頭辞: (6,1,4,2)
target = [6, 1, 4, 2]
initial = list(range(8))  # [0,1,2,3,4,5,6,7]

# 最小互換列を求める
trans_sequence = find_minimal_transpositions(target)

# 結果を表示
print(f"Found minimal sequence of {len(trans_sequence)} transpositions:")
print_transposition_sequence(initial, trans_sequence)

# 検証
final = initial.copy()
for i, j in trans_sequence:
    final = apply_transposition(final, i, j)
print("\nVerification:")
print(f"Target prefix: {target}")
print(f"Achieved prefix: {final[:len(target)]}")
print(f"Full permutation: {final}")