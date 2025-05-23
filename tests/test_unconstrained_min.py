import unittest
import numpy as np
from .examples import quadratic_example_1, quadratic_example_2, quadratic_example_3, rosenbrock_function, linear_function, exponential_function
from src.unconstrained_min import minimize
from src import utils


class TestUnconstrainedMin(unittest.TestCase):
    
    def setUp(self):
        """Set up test parameters"""
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter = 100
        
        # Define test functions and their expected properties
        self.test_functions = {
            'quadratic_example_1': {
                'func': quadratic_example_1,
                'x0': np.array([1.0, 1.0]),
                'expected_min': np.array([0.0, 0.0]),
                'expected_f_min': 0.0,
                'tolerance': 1e-6
            },
            'quadratic_example_2': {
                'func': quadratic_example_2,
                'x0': np.array([1.0, 1.0]),
                'expected_min': np.array([0.0, 0.0]),
                'expected_f_min': 0.0,
                'tolerance': 1e-6
            },
            'quadratic_example_3': {
                'func': quadratic_example_3,
                'x0': np.array([1.0, 1.0]),
                'expected_min': np.array([0.0, 0.0]),
                'expected_f_min': 0.0,
                'tolerance': 1e-6
            },
            'rosenbrock_function': {
                'func': rosenbrock_function,
                'x0': np.array([-1.0, 2.0]),
                'expected_min': np.array([1.0, 1.0]),
                'expected_f_min': 0.0,
                'tolerance': 1e-2
            },
            'linear_function': {
                'func': linear_function,
                'x0': np.array([0.0, 0.0]),
                'expected_min': None,
                'expected_f_min': None,
                'tolerance': None,
            },
            'exponential_function': {
                'func': exponential_function,
                'x0': np.array([0.0, 0.0]),
                'expected_min': np.array([-.346574, 0]),
                'expected_f_min': 2.55927,
                'tolerance': 1e-4,
            }
        }
    
    def test_all_functions_gradient_method(self):
        """Test all functions with gradient descent method"""
        for func_name, func_info in self.test_functions.items():
            with self.subTest(function=func_name, method='gradient'):
                self._test_function_with_method(func_name, func_info, 'gradient')
    
    def test_all_functions_newton_method(self):
        """Test all functions with Newton's method"""
        for func_name, func_info in self.test_functions.items():
            with self.subTest(function=func_name, method='newton'):
                self._test_function_with_method(func_name, func_info, 'newton')
    
    def _test_function_with_method(self, func_name, func_info, method):
        """Helper method to test a specific function with a specific method"""
        if func_name == 'rosenbrock_function' and method == 'gradient':
            self.max_iter = 10000
        else:
            self.max_iter = 100
        result_x, result_f, success, x_history, f_history = minimize(
            func_info['func'],
            func_info['x0'],
            self.obj_tol,
            self.param_tol,
            self.max_iter,
            type=method
        )
        
        print(f"\n{func_name} with {method} method:")
        print(f"  Starting point: {func_info['x0']}")
        print(f"  Final point: {result_x}")
        print(f"  Final function value: {result_f}")
        print(f"  Success: {success}")
        
        # Store results for plotting
        setattr(self, f"{func_name}_{method}_x_history", x_history)
        setattr(self, f"{func_name}_{method}_f_history", f_history)
        setattr(self, f"{func_name}_{method}_success", success)


    def test_create_plots(self):
        """Create contour plots and convergence plots for all functions"""
        print("\n" + "="*60)
        print("GENERATING PLOTS FOR ALL FUNCTIONS")
        print("="*60)
        
        for func_name, func_info in self.test_functions.items():
            if func_info.get('should_fail', False):
                print(f"\nSkipping plots for {func_name} (linear function - no minimum)")
                continue  # Skip linear function
                
            print(f"\nGenerating plots for {func_name}...")
            
            # Run both methods
            print(f"  Running gradient descent...")
            result_grad = minimize(
                func_info['func'], func_info['x0'], self.obj_tol, 
                self.param_tol, self.max_iter, type='gradient'
            )
            
            print(f"  Running Newton's method...")
            result_newton = minimize(
                func_info['func'], func_info['x0'], self.obj_tol, 
                self.param_tol, self.max_iter, type='newton'
            )
            
            # Extract data
            x_grad_history = result_grad[3]
            x_newton_history = result_newton[3]
            f_grad_history = result_grad[4]
            f_newton_history = result_newton[4]
            
            # Determine plot limits
            x_limits, y_limits = utils.determine_plot_limits(
                [x_grad_history, x_newton_history], func_info['func']
            )
            
            # Create contour plot with paths
            print(f"  Creating contour plot...")
            utils.plot_contour_with_paths(
                objective_func=func_info['func'],
                x_limits=x_limits,
                y_limits=y_limits,
                paths=[x_grad_history, x_newton_history],
                path_names=['Gradient Descent', "Newton's Method"],
                title=f'Optimization Paths: {func_name.replace("_", " ").title()}'
            )
            utils.save_plot(f'{func_name}_contour.png')
            
            # Create convergence comparison plot
            print(f"  Creating convergence plot...")
            utils.plot_convergence_comparison(
                function_histories=[f_grad_history, f_newton_history],
                method_names=['Gradient Descent', "Newton's Method"],
                title=f'Convergence Comparison: {func_name.replace("_", " ").title()}'
            )
            utils.save_plot(f'{func_name}_convergence.png')
            
            print(f"  ✓ Plots saved as {func_name}_contour.png and {func_name}_convergence.png")
        
        print("\n" + "="*60)
        print("ALL PLOTS GENERATED SUCCESSFULLY!")
        print("="*60)


if __name__ == '__main__':
    # Run with more verbose output
    unittest.main(verbosity=2)