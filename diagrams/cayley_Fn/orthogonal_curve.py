import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

def normal(A):
    return np.array([A[1], -A[0]])

def calculate_orthogonal_circle(A, B):
    t = (1 - np.dot(A, B)) / np.dot(B, normal(A))
    center = A + t * normal(A)
    radius = abs(t)
    return center, radius

def plot_orthogonal_circle_in_unit(fig, ax, A, B, arc_color='r', reverse_direction=False):
    # Calculate the orthogonal circle
    center, radius = calculate_orthogonal_circle(A, B)


    # Plot unit circle
    unit_circle = plt.Circle((0, 0), 1, fill=False, color='k')
    ax.add_artist(unit_circle)

    # Calculate intersection points of orthogonal circle and unit circle
    d = np.linalg.norm(center)
    a = (1 + d**2 - radius**2) / (2*d)
    h = np.sqrt(max(0, 1 - a**2))
    x1 = a * center[0] / d + h * center[1] / d
    y1 = a * center[1] / d - h * center[0] / d
    x2 = a * center[0] / d - h * center[1] / d
    y2 = a * center[1] / d + h * center[0] / d

    # Calculate angles for the arc
    start_angle = np.arctan2(y1 - center[1], x1 - center[0])
    end_angle = np.arctan2(y2 - center[1], x2 - center[0])

    # Ensure the arc is drawn inside the unit circle
    if np.cross(center, [x1, y1]) < 0:
        start_angle, end_angle = end_angle, start_angle

    if reverse_direction:
        start_angle, end_angle = end_angle, start_angle

    # Plot orthogonal circle arc
    arc = Arc(center, 2*radius, 2*radius, angle=0, 
              theta1=np.degrees(start_angle), theta2=np.degrees(end_angle), 
              color=arc_color)
    ax.add_artist(arc)
    print(f"Center of orthogonal circle: {center}")
    print(f"Radius of orthogonal circle: {radius}")

    return fig, ax


if __name__ == "__main__":
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Example usage
    A = np.array([1, 0])  # Point on unit circle
    B = np.array([np.cos(np.pi/4), np.sin(np.pi/4)])  # Another point on unit circle
    fig, ax=plot_orthogonal_circle_in_unit(fig, ax, A, B, arc_color='blue', reverse_direction=False)




    # Plot points A and B
    #ax.plot(A[0], A[1], 'go', label='Point A')
    #ax.plot(B[0], B[1], 'go', label='Point B')

    # Set axis limits and labels
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)

    # Add legend
    ax.legend()

    # Show plot
    plt.title('Unit Circle and Orthogonal Arc (Inside Only)')
    plt.show()






    """
    F_2:
    4 points  2  edges: 1/2 : 0 to \pi
    8 - 4  :- \pi /6 to \pi / 6
    24 - 12: - \pi / 18  to \pi /18 
    ...  

    -> 12  -> 36 -> ...
    """