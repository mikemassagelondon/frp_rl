import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sample_quarter_circle(n_samples: int, radius: float = 2.0) -> np.ndarray:
    """
    Generate i.i.d. samples from the quarter-circle distribution 
    on [0, radius].
    """
    beta_samples = np.random.beta(a=0.5, b=1.5, size=n_samples)
    return radius * np.sqrt(beta_samples)

def quarter_circle_pdf(x: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """
    Exact quarter-circle PDF on [0, radius].
    """
    val = np.zeros_like(x, dtype=float)
    mask = (x >= 0) & (x <= radius)
    # f_X(x) = (4 / (π * radius^2)) * sqrt(radius^2 - x^2)
    val[mask] = (4.0 / (np.pi * radius**2)) * np.sqrt(radius**2 - x[mask]**2)
    return val

def sample_product_of_quarter_circle(n_samples: int, ell: int = 2, radius: float = 2.0) -> np.ndarray:
    """
    Sample the product of 'ell' i.i.d. quarter-circle random variables.
    """
    if ell < 1:
        raise ValueError("ell must be >= 1.")
    
    result = sample_quarter_circle(n_samples, radius=radius)
    for _ in range(ell - 1):
        result *= sample_quarter_circle(n_samples, radius=radius)
    return result

def plot_distribution(ax, ell: int = 1, radius: float = 2.0, 
                      n_samples: int = 100_000, bins: int = 200) -> plt.Axes:
    """
    Plot (onto the given axis 'ax') a histogram of the distribution of:
       - Quarter-circle (ell=1), or
       - The product of ell i.i.d. quarter-circle r.v.s (ell>1).
    Using seaborn's histogram with only the borderline for function approximation.
    """
    samples = sample_product_of_quarter_circle(n_samples, ell=ell, radius=radius)
    range_max = radius**ell
    
    # Plot only the outline/border of the histogram using seaborn
    if ell == 1:
        # For ell=1, plot the outline in black
        sns.histplot(
            samples,
            bins=bins,
            stat="density",
            element="step",  # Use step to show only the outline
            fill=False,      # Don't fill the bars
            color="black",
            linewidth=1.5,
            ax=ax,
            label=f"ell={ell}",
            binrange=(0, range_max)
        )
    else:
        # For ell>1, plot the outline in blue (C1)
        sns.histplot(
            samples,
            bins=bins,
            stat="density",
            element="step",  # Use step to show only the outline
            fill=False,      # Don't fill the bars
            color="C1",
            linewidth=1.5,
            ax=ax,
            label=f"ell={ell}",
            binrange=(0, range_max)
        )
    
    # If ell=1, overlay exact quarter-circle PDF
    if ell == 1:
        xgrid = np.linspace(0, radius, 400)
        ax.plot(xgrid, quarter_circle_pdf(xgrid, radius=radius), 'r--', label="Exact PDF")
        ax.set_xlim(0, radius)
    else:
        ax.set_xlim(0, range_max)

    title_str = f"Quarter-circle product (ell={ell}, radius={radius})"
    ax.set_title(title_str)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_ylim(bottom=0)
    ax.legend()
    return ax

# Example usage
if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    R=2
    # Plot ell=1
    plot_distribution(axes[0], ell=1,  radius=R, n_samples=100_000, bins=100)

    # Plot ell=2 (edges only, no fill)
    plot_distribution(axes[1], ell=2,  radius=R, n_samples=100_000, bins=100)

    # Plot ell=4 (edges only, no fill)
    plot_distribution(axes[2], ell=4,  radius=R, n_samples=100_000, bins=100)

    plt.tight_layout()
    plt.savefig("quater_circular_prods.png")
