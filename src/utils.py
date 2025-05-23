import numpy as np
import matplotlib.pyplot as plt


def plot_contour_with_paths(objective_func, x_limits, y_limits, paths=None, path_names=None, title="Objective Function Contours", levels=20, figsize=(10, 8)):
    x = np.linspace(x_limits[0], x_limits[1], 100)
    y = np.linspace(y_limits[0], y_limits[1], 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            try:
                Z[i, j], _ = objective_func(point, False)
            except:
                Z[i, j] = np.inf  # Handle cases where function might fail
    
    plt.figure(figsize=figsize)
    
    if isinstance(levels, int):
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        if np.isinf(z_max):
            z_max = np.nanpercentile(Z[np.isfinite(Z)], 95)
        if np.isinf(z_min):
            z_min = np.nanpercentile(Z[np.isfinite(Z)], 5)
        
        if z_max / z_min > 100:
            levels = np.logspace(np.log10(max(z_min, 1e-10)), np.log10(z_max), levels)
        else:
            levels = np.linspace(z_min, z_max, levels)
    
    contour = plt.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    
    contourf = plt.contourf(X, Y, Z, levels=levels, alpha=0.3, cmap='viridis')
    plt.colorbar(contourf, label='Function Value')
    
    if paths is not None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'v', 'D', 'p']
        
        for i, path in enumerate(paths):
            if path is not None and len(path) > 0:
                if isinstance(path, list):
                    path = np.array(path)
                
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                plt.plot(path[:, 0], path[:, 1], 
                        color=color, marker=marker, linestyle='-', 
                        linewidth=2, markersize=6, alpha=0.8,
                        label=path_names[i] if path_names else f'Path {i+1}')
                
                plt.plot(path[0, 0], path[0, 1], marker='o', color=color, 
                        markersize=10, markerfacecolor='white', markeredgewidth=2)
                plt.plot(path[-1, 0], path[-1, 1], marker='*', color=color, 
                        markersize=12, markeredgewidth=1)
    
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if paths is not None and any(path is not None for path in paths):
        plt.legend(loc='best')
    
    plt.tight_layout()
    return plt.gcf()


def plot_convergence_comparison(function_histories, method_names, title="Convergence Comparison", figsize=(12, 5)):

    plt.figure(figsize=figsize)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'v', 'D', 'p']
    
    for i, (f_history, method_name) in enumerate(zip(function_histories, method_names)):
        if f_history is not None and len(f_history) > 0:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            plt.plot(range(len(f_history)), f_history, 
                    color=color, marker=marker, linestyle='-', 
                    linewidth=2, markersize=4, alpha=0.8,
                    label=method_name)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Function Value', fontsize=12)
    plt.title(f'{title} - Linear Scale', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    return plt.gcf()


def determine_plot_limits(paths, objective_func, padding_factor=0.3):
    
    if paths is None or all(path is None for path in paths):
        return (-5, 5), (-5, 5)
    
    all_points = []
    for path in paths:
        if path is not None:
            all_points.extend(path)
    
    if not all_points:
        return (-5, 5), (-5, 5)
    
    all_points = np.array(all_points)
    
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    if x_range < 1e-6:
        x_range = 2
    if y_range < 1e-6:
        y_range = 2
    
    x_padding = padding_factor * x_range
    y_padding = padding_factor * y_range
    
    x_limits = (x_min - x_padding, x_max + x_padding)
    y_limits = (y_min - y_padding, y_max + y_padding)
    
    return x_limits, y_limits


def save_plot(filename, dpi=150):
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
