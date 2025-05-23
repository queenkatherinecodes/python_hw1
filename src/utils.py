import numpy as np
import matplotlib.pyplot as plt


def plot_contour_with_paths(objective_func, x_limits, y_limits, paths=None, path_names=None, 
                           title="Objective Function Contours", levels=20, figsize=(10, 8)):
    """
    Plot contour lines of an objective function with optional optimization paths.
    
    Parameters:
    - objective_func: function that takes (x, is_newton) and returns (f, g) or (f, g, h)
    - x_limits: tuple (x_min, x_max) for x-axis limits
    - y_limits: tuple (y_min, y_max) for y-axis limits  
    - paths: list of numpy arrays, each array is Nx2 containing optimization path points
    - path_names: list of strings, names for each path (for legend)
    - title: string, title for the plot
    - levels: int or array-like, contour levels
    - figsize: tuple, figure size
    """
    
    # Create grid for contour plot
    x = np.linspace(x_limits[0], x_limits[1], 100)
    y = np.linspace(y_limits[0], y_limits[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate function on grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            try:
                Z[i, j], _ = objective_func(point, False)
            except:
                Z[i, j] = np.inf  # Handle cases where function might fail
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot contour lines
    if isinstance(levels, int):
        # Automatically choose good levels
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        if np.isinf(z_max):
            z_max = np.nanpercentile(Z[np.isfinite(Z)], 95)
        if np.isinf(z_min):
            z_min = np.nanpercentile(Z[np.isfinite(Z)], 5)
        
        # Use logarithmic spacing for better visualization if range is large
        if z_max / z_min > 100:
            levels = np.logspace(np.log10(max(z_min, 1e-10)), np.log10(z_max), levels)
        else:
            levels = np.linspace(z_min, z_max, levels)
    
    contour = plt.contour(X, Y, Z, levels=levels, colors='gray', alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    
    # Plot filled contours for better visualization
    contourf = plt.contourf(X, Y, Z, levels=levels, alpha=0.3, cmap='viridis')
    plt.colorbar(contourf, label='Function Value')
    
    # Plot optimization paths if provided
    if paths is not None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'v', 'D', 'p']
        
        for i, path in enumerate(paths):
            if path is not None and len(path) > 0:
                # Convert to numpy array if it's a list
                if isinstance(path, list):
                    path = np.array(path)
                
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                # Plot path
                plt.plot(path[:, 0], path[:, 1], 
                        color=color, marker=marker, linestyle='-', 
                        linewidth=2, markersize=6, alpha=0.8,
                        label=path_names[i] if path_names else f'Path {i+1}')
                
                # Mark start and end points
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


def plot_convergence_comparison(function_histories, method_names, title="Convergence Comparison", 
                              figsize=(12, 5), log_scale=True):
    """
    Plot function values vs iteration for multiple methods to compare convergence.
    
    Parameters:
    - function_histories: list of numpy arrays, each containing function values at each iteration
    - method_names: list of strings, names for each method
    - title: string, title for the plot
    - figsize: tuple, figure size
    - log_scale: bool, whether to use log scale for y-axis
    """
    
    plt.figure(figsize=figsize)
    
    # Create subplots for linear and log scale
    if log_scale:
        plt.subplot(1, 2, 1)
    
    # Linear scale plot
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
    plt.title(f'{title} - Linear Scale' if log_scale else title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale plot
    if log_scale:
        plt.subplot(1, 2, 2)
        
        for i, (f_history, method_name) in enumerate(zip(function_histories, method_names)):
            if f_history is not None and len(f_history) > 0:
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                # Handle negative or zero values for log scale
                f_positive = np.maximum(f_history - np.min(f_history) + 1e-16, 1e-16)
                
                plt.semilogy(range(len(f_positive)), f_positive, 
                            color=color, marker=marker, linestyle='-', 
                            linewidth=2, markersize=4, alpha=0.8,
                            label=method_name)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Function Value (log scale)', fontsize=12)
        plt.title(f'{title} - Log Scale', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()


def determine_plot_limits(paths, objective_func, padding_factor=0.3):
    """
    Automatically determine good plot limits based on optimization paths.
    
    Parameters:
    - paths: list of numpy arrays containing optimization paths
    - objective_func: objective function (used to evaluate interesting regions)
    - padding_factor: float, fraction of range to add as padding
    
    Returns:
    - x_limits: tuple (x_min, x_max)
    - y_limits: tuple (y_min, y_max)
    """
    
    if paths is None or all(path is None for path in paths):
        return (-5, 5), (-5, 5)  # Default limits
    
    # Collect all points from all paths
    all_points = []
    for path in paths:
        if path is not None:
            all_points.extend(path)
    
    if not all_points:
        return (-5, 5), (-5, 5)
    
    all_points = np.array(all_points)
    
    # Find bounds
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Ensure minimum range
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
    """Save current plot to file."""
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')


def show_plot():
    """Display current plot."""
    plt.show()