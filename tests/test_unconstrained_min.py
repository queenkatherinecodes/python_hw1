import unittest
import numpy as np
from .examples import quadratic_example_1
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
            # 'quadratic_example_2': {
            #     'func': examples.quadratic_example_2,
            #     'x0': np.array([1.0, 1.0]),
            #     'expected_min': np.array([0.0, 0.0]),
            #     'expected_f_min': 0.0,
            #     'tolerance': 1e-6
            # },
            # 'quadratic_example_3': {
            #     'func': examples.quadratic_example_3,
            #     'x0': np.array([1.0, 1.0]),
            #     'expected_min': np.array([0.0, 0.0]),
            #     'expected_f_min': 0.0,
            #     'tolerance': 1e-6
            # },
            # 'rosenbrock_function': {
            #     'func': examples.rosenbrock_function,
            #     'x0': np.array([0.0, 0.0]),
            #     'expected_min': np.array([1.0, 1.0]),
            #     'expected_f_min': 0.0,
            #     'tolerance': 1e-4  # More relaxed for Rosenbrock
            # },
            # 'linear_function': {
            #     'func': examples.linear_function,
            #     'x0': np.array([0.0, 0.0]),
            #     'expected_min': None,  # Linear functions don't have minima
            #     'expected_f_min': None,
            #     'tolerance': None,
            #     'should_fail': True  # Linear function should not converge to minimum
            # },
            # 'exponential_function': {
            #     'func': examples.exponential_function,
            #     'x0': np.array([0.0, 0.0]),
            #     'expected_min': None,  # Will need to be determined empirically
            #     'expected_f_min': None,
            #     'tolerance': 1e-4,
            #     'check_convergence_only': True
            # }
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
        
        # Handle special cases
        if func_info.get('should_fail', False):
            # Linear function should not converge to a minimum
            self.assertFalse(success, f"Linear function should not converge with {method} method")
            return
        
        if func_info.get('check_convergence_only', False):
            # Just check that it converges, don't check specific values
            self.assertTrue(success, f"{func_name} with {method} method should converge")
            return
        
        # Standard checks for functions with known minima
        if func_info['expected_min'] is not None:
            self.assertTrue(success, f"{func_name} with {method} method should converge")
            
            # Check that result is close to expected minimum
            np.testing.assert_allclose(
                result_x, 
                func_info['expected_min'], 
                atol=func_info['tolerance'],
                err_msg=f"{func_name} with {method} method: final point not close to expected minimum"
            )
            
            # Check that function value is close to expected minimum value
            self.assertAlmostEqual(
                result_f, 
                func_info['expected_f_min'], 
                delta=func_info['tolerance'],
                msg=f"{func_name} with {method} method: final function value not close to expected minimum"
            )

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
            utils.show_plot()
            
            # Create convergence comparison plot
            print(f"  Creating convergence plot...")
            utils.plot_convergence_comparison(
                function_histories=[f_grad_history, f_newton_history],
                method_names=['Gradient Descent', "Newton's Method"],
                title=f'Convergence Comparison: {func_name.replace("_", " ").title()}'
            )
            utils.save_plot(f'{func_name}_convergence.png')
            utils.show_plot()
            
            print(f"  ✓ Plots saved as {func_name}_contour.png and {func_name}_convergence.png")
        
        print("\n" + "="*60)
        print("ALL PLOTS GENERATED SUCCESSFULLY!")
        print("="*60)


if __name__ == '__main__':
    # Run with more verbose output
    unittest.main(verbosity=2)