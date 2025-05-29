import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
import cvxpy as cp

def generate_semi_stochastic_matrix(n):
    M = np.random.rand(n, n)
    M = M / M.sum(axis=0)
    return M

def find_optimal_circle(eigenvalues):
    n = len(eigenvalues)
    R = cp.Variable(nonneg=True)
    x_c = cp.Variable()
    y_c = cp.Variable()
    
    constraints = []
    for i in range(n):
        l_real = eigenvalues[i].real
        l_imag = eigenvalues[i].imag
        # Correction : créer un vecteur avec les différences
        diff = cp.vstack([l_real - x_c, l_imag - y_c])
        constraints.append(cp.norm(diff, 2) <= R)
    
    problem = cp.Problem(cp.Minimize(R), constraints)
    problem.solve(solver='ECOS')
    
    if problem.status != 'optimal':
        raise ValueError("Le problème n'a pas convergé vers une solution optimale.")
    
    return R.value, x_c.value, y_c.value

def plot_eigenvalues_with_optimal_circle(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    
    try:
        R, x_c, y_c = find_optimal_circle(eigenvalues)
    except ValueError as e:
        print(f"Erreur lors de l'optimisation: {e}")
        return
    
    plt.figure(figsize=(8, 8))
    
    # Valeurs propres
    plt.scatter(real_parts, imag_parts, color='blue', label='Valeurs propres')
    
    # Cercle unité
    unit_circle = Circle((0, 0), 1, fill=False, color='red', linestyle='--', label='Cercle unité')
    plt.gca().add_patch(unit_circle)
    
    # Cercle optimal
    if R is not None:
        optimal_circle = Circle((x_c, y_c), R, fill=False, color='green', linewidth=2, label='Cercle optimal')
        plt.gca().add_patch(optimal_circle)
        plt.scatter([x_c], [y_c], color='green', marker='x', s=100, label='Centre optimal')
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Valeurs propres et cercles dans le plan complexe')
    plt.xlabel('Partie réelle')
    plt.ylabel('Partie imaginaire')
    plt.axis('equal')
    plt.legend()
    plt.show()

# Exemple d'utilisation
n = 10
matrix = generate_semi_stochastic_matrix(n)
#plot_eigenvalues_with_optimal_circle(matrix)

def find_optimal_ellipse(eigenvalues):
    n = len(eigenvalues)
    A = cp.Variable((2,2))
    b = cp.Variable(2)
    constraints = [A >> 0]
    
    for l in eigenvalues:
        l_v = cp.vstack([l.real,l.imag])
        constraints.append(cp.norm2(A@l_v + b) <= 1)
    problem = cp.Problem(cp.Minimize(-cp.log_det(A)),constraints)
    problem.solve(solver='SCS')
    if problem.status != 'optimal':
        raise ValueError("Le problème n'a pas convergé vers une solution optimale.")
    return A.value, b.value
def plot_eigenvalues_with_optimal_ellipse(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    
    try:
        A, b = find_optimal_ellipse(eigenvalues)
    except ValueError as e:
        print(f"Erreur lors de l'optimisation: {e}")
        return
    
    plt.figure(figsize=(8, 8))
    
    # Valeurs propres
    plt.scatter(real_parts, imag_parts, color='blue', label='Valeurs propres')
    
    # Cercle unité
    unit_circle = Circle((0, 0), 1, fill=False, color='red', linestyle='--', label='Cercle unité')
    plt.gca().add_patch(unit_circle)
    
    # Ellipse optimale
    if A is not None and b is not None:
        # Calcul des paramètres de l'ellipse
        center = -np.linalg.solve(A, b)  # Centre de l'ellipse
        U, s, Vt = np.linalg.svd(A)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))  # Angle de rotation
        width = 2 / np.sqrt(s[0])  # Demi-grand axe
        height = 2 / np.sqrt(s[1])  # Demi-petit axe
        
        # Tracé de l'ellipse
        ellipse = Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,
            fill=False,
            color='green',
            linewidth=2,
            label='Ellipse optimale'
        )
        plt.gca().add_patch(ellipse)
        plt.scatter([center[0]], [center[1]], color='green', marker='x', s=100, label='Centre optimal')
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Valeurs propres et ellipses dans le plan complexe')
    plt.xlabel('Partie réelle')
    plt.ylabel('Partie imaginaire')
    plt.axis('equal')
    plt.legend()
    plt.show()

plot_eigenvalues_with_optimal_ellipse(matrix)