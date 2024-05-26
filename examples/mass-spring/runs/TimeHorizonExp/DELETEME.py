#%%

def loss_function(x, y):
    return (x**2 - 1)**2 + (y**2 - 1)**2


import numpy as np

def gradient(x, y):
    dL_dx = 4*x*(x**2 - 1)
    dL_dy = 4*y*(y**2 - 1)
    return np.array([dL_dx, dL_dy])


def gradient_norm(x, y):
    grad = gradient(x, y)
    return np.linalg.norm(grad)


threshold = 0.1  # Example threshold


x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)

plateau_points = []

for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        norm = gradient_norm(x_vals[i], y_vals[j])
        if norm < threshold:
            plateau_points.append((x_vals[i], y_vals[j]))

plateau_points = np.array(plateau_points)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.contour(X, Y, loss_function(X, Y), levels=50)
if len(plateau_points) > 0:
    plt.scatter(plateau_points[:, 0], plateau_points[:, 1], color='red', s=10)
plt.title('Plateau Region in Loss Landscape')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


#%%

def hessian(x, y):
    d2L_dx2 = 12*x**2 - 4
    d2L_dy2 = 12*y**2 - 4
    d2L_dxdy = 0
    return np.array([[d2L_dx2, d2L_dxdy], [d2L_dxdy, d2L_dy2]])


def hessian_eigenvalues(x, y):
    H = hessian(x, y)
    return np.linalg.eigvals(H)


eigenvalue_threshold = 0.1  # Example threshold


x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)


plateau_points_hessian = []

for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        eigenvalues = hessian_eigenvalues(x_vals[i], y_vals[j])
        if all(abs(ev) < eigenvalue_threshold for ev in eigenvalues):
            plateau_points_hessian.append((x_vals[i], y_vals[j]))

plateau_points_hessian = np.array(plateau_points_hessian)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, loss_function(X, Y), levels=50)
if len(plateau_points_hessian) > 0:
    plt.scatter(plateau_points_hessian[:, 0], plateau_points_hessian[:, 1], color='blue', s=10)
plt.title('Plateau Region in Loss Landscape (Hessian Eigenvalues)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
