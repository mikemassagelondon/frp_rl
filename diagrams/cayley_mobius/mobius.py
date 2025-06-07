#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from geodesic import find_circle_center_and_radius, geodesic_points_on_circle
from geodesic import is_colinear_with_origin


def double(z, r, fn):
    """Apply the given Möbius transformation twice.
    
    Args:
        z (complex): The input point
        r (float): The parameter for the Möbius transformation
        fn (function): The Möbius transformation function
        
    Returns:
        complex: The result of applying the transformation twice
    """
    for _ in range(2):
        z = fn(z, r)
    return z
    
# Möbius transformation functions
def mobius_A(z, r):
    """Apply the Möbius transformation A.
    
    Args:
        z (complex): The input point
        r (float): The parameter for the transformation
        
    Returns:
        complex: The transformed point
    """
    return (z - r) / (1 - r * z)

def mobius_B(z, r):
    """Apply the Möbius transformation B.
    
    Args:
        z (complex): The input point
        r (float): The parameter for the transformation
        
    Returns:
        complex: The transformed point
    """
    return (z - 1j * r) / (1 + 1j * r * z)

def inverse_A(z, r):
    """Apply the inverse of Möbius transformation A.
    
    Args:
        z (complex): The input point
        r (float): The parameter for the transformation
        
    Returns:
        complex: The transformed point
    """
    out = (z + r) / (1 + r * z)
    return out

def inverse_B(z, r):
    """Apply the inverse of Möbius transformation B.
    
    Args:
        z (complex): The input point
        r (float): The parameter for the transformation
        
    Returns:
        complex: The transformed point
    """
    out = (z + 1j * r) / (1 - 1j * r * z)
    return out

# Generate the free group with two generators
def generate_group_with_edges(r, L, colors):
    """Generate vertices and edges for the free group F_2 on the Poincaré disk.
    
    Args:
        r (float): The parameter for the Möbius transformations
        L (int): The depth of the Cayley graph
        colors (dict): Mapping from generators to colors
        
    Returns:
        group (set): The set of generated words
        edges (list): List of tuples (start point, end point, color)
        points (dict): Mapping from words to points on the Poincaré disk
    """
    transformations = {'A': lambda z: mobius_A(z, r), 
                       'B': lambda z: mobius_B(z, r),
                       'a': lambda z: inverse_A(z, r),
                       'b': lambda z: inverse_B(z, r)}
    
    group = {''}  # Identity transformation
    edges = []  # Edge information
    current_level = {''}
    points = {'' : 0}  # Points on the Poincaré disk corresponding to each word
    
    for _ in range(L):
        next_level = set()
        for word in current_level:
            z_start = points[word]
            for gen in ['A', 'B', 'a', 'b']:
                # Avoid creating reducible words (e.g., Aa, aA, Bb, bB)
                if not (len(word) > 0 and ((word[-1] == 'A' and gen == 'a') or 
                                           (word[-1] == 'a' and gen == 'A') or 
                                           (word[-1] == 'B' and gen == 'b') or 
                                           (word[-1] == 'b' and gen == 'B'))):
                    new_word = word + gen
                    z_end = transformations[gen](z_start)
                    next_level.add(new_word)
                    group.add(new_word)
                    points[new_word] = z_end
                    # Add edge information (start, end, color)
                    edges.append((z_start, z_end, colors[gen]))
        current_level = next_level
    
    return group, edges, points

def poincare_distance(z1, z2):
    """Calculate the hyperbolic distance between two points on the Poincaré disk.
    
    Args:
        z1 (complex): First point
        z2 (complex): Second point
        
    Returns:
        float: The hyperbolic distance between z1 and z2
    """
    numerator = abs(z1 - z2) ** 2
    denominator = (1 - abs(z1) ** 2) * (1 - abs(z2) ** 2)
    return np.arccosh(1 + 2 * numerator / denominator)

def closest_pairs(points, k):
    """Find the k closest pairs of points based on Poincaré distance.
    
    Args:
        points (dict): Mapping from words to points on the Poincaré disk
        k (int): Number of closest pairs to return
        
    Returns:
        tuple: (k closest pairs with distances, furthest pair with distance)
    """
    point_items = list(points.items())  # Convert to list of (word, point) pairs
    distances = []
    
    for i in range(len(point_items)):
        for j in range(i + 1, len(point_items)):
            word1, z1 = point_items[i]
            word2, z2 = point_items[j]
            distance = poincare_distance(z1, z2)
            distances.append((distance, (word1, word2)))
    
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    # Return the k closest pairs and the furthest pair
    return distances[:k], distances[-1]

def geodesic_poincare(z1, z2, n=100):
    """Return a sequence of points along the geodesic from z1 to z2 on the Poincaré disk.
    
    Args:
        z1 (complex): Starting point
        z2 (complex): Ending point
        n (int, optional): Number of points to generate. Defaults to 100.
        
    Returns:
        list: Sequence of points along the geodesic
    """
    if is_colinear_with_origin(z1, z2):
        zs = [z1, z2]
    else:
        center, R = find_circle_center_and_radius(z1, z2)
        zs = geodesic_points_on_circle(center, R, z1, z2, num=n)
    return zs


def plot_cayley_graph_with_edges(points, edges, show_edges=True, edge_alpha=0.8):
    """Plot the Cayley graph on the Poincaré disk.
    
    Args:
        points (dict): Mapping from words to points on the Poincaré disk
        edges (list): List of tuples (start point, end point, color)
        show_edges (bool, optional): Whether to show edges. Defaults to True.
        edge_alpha (float, optional): Alpha value for edges. Defaults to 0.8.
        
    Returns:
        tuple: (figure, axis) for the plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw edges
    if show_edges:
        for z_start, z_end, color in edges:
            # Get points along the geodesic from z_start to z_end
            geodesic_pts = geodesic_poincare(z_start, z_end, n=100)
            # Extract real and imaginary parts
            gx = [z.real for z in geodesic_pts]
            gy = [z.imag for z in geodesic_pts]
            ax.plot(gx, gy, color=color, lw=edge_alpha)

    # Draw vertices
    x, y = [z.real for z in points.values()], [z.imag for z in points.values()]
    ax.scatter(x, y, color='black', s=2, label='Vertices')
    return fig, ax


def main():
    """Main function to generate and plot the Cayley graph on the Poincaré disk."""
    parser = argparse.ArgumentParser(
        description="Generate and plot the Cayley graph of the free group F_2 on the Poincaré disk")
    parser.add_argument("--L", type=int, default=2,
                        help="Maximum length of reduced words (default: 2)")
    parser.add_argument("--outfile", type=str, default="mobius_cayley.pdf",
                        help="Output PDF file name (default: mobius_cayley.pdf)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output PDF DPI (default: 300)")
    parser.add_argument("--edge-alpha", type=float, default=0.5,
                        help="Alpha value for edges (default: 0.5)")
    parser.add_argument("--show-edges", action="store_true", default=True,
                        help="Show edges in the plot (default: True)")
    parser.add_argument("--print-distances", action="store_true", default=True,
                        help="Print distances between pairs of points (default: True)")
    parser.add_argument("--k", type=int, default=4*3**3,
                        help="Number of closest pairs to output (default: 4*3^3)")
    args = parser.parse_args()

    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Parameter settings
    r = 1/math.sqrt(2)  # Maximum possible value
    print(f"Mobius Param r = {r}")
    L = args.L
    k = args.k
    edge_alpha = args.edge_alpha
    show_edges = args.show_edges
    print_distances = args.print_distances

    # Use Seaborn color palette
    palette = sns.color_palette("husl", 4)  # 4 colors
    colors = {'A': palette[0], 'a': palette[1], 'B': palette[2], 'b': palette[3]}

    # Generate the free group and plot the graph
    group, edges, points = generate_group_with_edges(r, L, colors)
    fig, ax = plot_cayley_graph_with_edges(points, edges, show_edges, edge_alpha)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect('equal', adjustable='datalim')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(args.outfile, dpi=args.dpi, format='pdf', bbox_inches='tight')
    print(f"Saved to {args.outfile} with dpi={args.dpi}")

    if print_distances:
        # Sort and output pairs by distance
        closest_pairs_result, far_pair_result = closest_pairs(points, k)
        for i, (distance, (word1, word2)) in enumerate(closest_pairs_result, 1):
            print(f"{i}: Distance = {distance:.4f}, Pair = ({word1}, {word2})")

        max_distance, (word1, word2) = far_pair_result
        print(f"Max Distance = {max_distance:.4f}, Pair = ({word1}, {word2})")


if __name__ == "__main__":
    main()
