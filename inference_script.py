"""
4_inference_and_benchmark.py
Use trained surrogate model for predictions and benchmark against MCNP
"""

import numpy as np
import pandas as pd
import torch
import pickle
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt

class SurrogatePredictor:
    """Load and use trained surrogate model"""
    
    def __init__(self, model_path='best_model.pt', scaler_path='scaler.pkl', 
                 metadata_path='model_metadata.json'):
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load model
        from train_surrogate_model import ShieldingSurrogate
        
        arch = self.metadata['model_architecture']
        self.model = ShieldingSurrogate(
            input_dim=arch['input_dim'],
            hidden_dims=arch['hidden_dims'],
            dropout=0  # No dropout during inference
        )
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.feature_cols = self.metadata['feature_cols']
    
    def predict(self, params_dict):
        """
        Predict flux from parameters
        
        Args:
            params_dict: dict with keys matching feature_cols
        
        Returns:
            Predicted flux (not log-transformed)
        """
        # Extract features in correct order
        X = np.array([[params_dict[col] for col in self.feature_cols]])
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            log_flux_pred = self.model(X_tensor).numpy()[0, 0]
        
        # Transform back from log space
        flux_pred = 10 ** log_flux_pred
        
        return flux_pred
    
    def predict_batch(self, params_df):
        """Predict for batch of parameters"""
        X = params_df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            log_flux_pred = self.model(X_tensor).numpy().flatten()
        
        return 10 ** log_flux_pred

class BenchmarkComparison:
    """Compare surrogate predictions with MCNP results"""
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def benchmark_accuracy(self, test_data_path='training_data.csv', n_samples=100):
        """Compare predictions with MCNP on random test cases"""
        
        # Load data
        df = pd.read_csv(test_data_path)
        df = df.dropna(subset=['total_flux'])
        
        # Sample random cases
        test_df = df.sample(n=min(n_samples, len(df)), random_state=42)
        
        # Get MCNP results
        mcnp_flux = test_df['total_flux'].values
        mcnp_error = test_df['total_error'].values
        
        # Get surrogate predictions
        surrogate_flux = self.predictor.predict_batch(test_df)
        
        # Calculate relative errors
        rel_errors = np.abs(surrogate_flux - mcnp_flux) / mcnp_flux
        
        results = {
            'mcnp_flux': mcnp_flux,
            'mcnp_error': mcnp_error,
            'surrogate_flux': surrogate_flux,
            'rel_error': rel_errors,
            'within_mcnp_uncertainty': rel_errors < mcnp_error
        }
        
        return pd.DataFrame(results)
    
    def benchmark_speed(self, n_predictions=1000):
        """Benchmark prediction speed"""
        
        # Generate random parameters
        params_list = []
        for _ in range(n_predictions):
            params = {
                'concrete_thick': np.random.uniform(10, 100),
                'lead_thick': np.random.uniform(0, 20),
                'poly_thick': np.random.uniform(0, 30),
                'source_energy': np.random.uniform(0.1, 10.0),
                'source_intensity': np.random.uniform(1e8, 1e10)
            }
            params_list.append(params)
        
        # Time predictions
        start_time = time.time()
        for params in params_list:
            _ = self.predictor.predict(params)
        end_time = time.time()
        
        time_per_prediction = (end_time - start_time) / n_predictions
        
        # Typical MCNP runtime (estimate)
        mcnp_time = 60  # seconds for 1M particles
        
        speedup = mcnp_time / time_per_prediction
        
        return {
            'n_predictions': n_predictions,
            'total_time': end_time - start_time,
            'time_per_prediction': time_per_prediction,
            'estimated_mcnp_time': mcnp_time,
            'speedup_factor': speedup
        }

def plot_benchmark_results(results_df, output_dir='plots'):
    """Visualize benchmark results"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Plot 1: Surrogate vs MCNP
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(results_df['mcnp_flux'], results_df['surrogate_flux'], 
               alpha=0.6, s=30)
    
    min_val = min(results_df['mcnp_flux'].min(), results_df['surrogate_flux'].min())
    max_val = max(results_df['mcnp_flux'].max(), results_df['surrogate_flux'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect agreement')
    
    ax.set_xlabel('MCNP Flux')
    ax.set_ylabel('Surrogate Flux')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Surrogate vs MCNP Predictions')
    
    # Relative error histogram
    ax = axes[1]
    ax.hist(results_df['rel_error'] * 100, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(results_df['mcnp_error'].mean() * 100, color='r', 
               linestyle='--', label=f'Mean MCNP uncertainty')
    ax.set_xlabel('Relative Error (%)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Distribution of Relative Errors')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print("\nAccuracy Statistics:")
    print(f"  Mean relative error: {results_df['rel_error'].mean()*100:.2f}%")
    print(f"  Median relative error: {results_df['rel_error'].median()*100:.2f}%")
    print(f"  Max relative error: {results_df['rel_error'].max()*100:.2f}%")
    print(f"  Predictions within MCNP uncertainty: "
          f"{results_df['within_mcnp_uncertainty'].sum()}/{len(results_df)} "
          f"({results_df['within_mcnp_uncertainty'].mean()*100:.1f}%)")

def demonstrate_use_case(predictor):
    """Demonstrate practical use case: parameter sweep"""
    print("\n" + "="*60)
    print("DEMONSTRATION: Shield Optimization")
    print("="*60)
    print("\nProblem: Find minimum concrete thickness to reduce flux below target")
    
    target_flux = 1e-6  # Target flux level
    
    # Fixed parameters
    base_params = {
        'lead_thick': 5.0,
        'poly_thick': 10.0,
        'source_energy': 2.0,
        'source_intensity': 1e9
    }
    
    # Sweep concrete thickness
    concrete_range = np.linspace(10, 100, 50)
    fluxes = []
    
    print("\nRunning parameter sweep...")
    start_time = time.time()
    
    for concrete in concrete_range:
        params = {**base_params, 'concrete_thick': concrete}
        flux = predictor.predict(params)
        fluxes.append(flux)
    
    sweep_time = time.time() - start_time
    
    # Find optimal thickness
    fluxes = np.array(fluxes)
    idx = np.where(fluxes < target_flux)[0]
    
    if len(idx) > 0:
        optimal_thickness = concrete_range[idx[0]]
        optimal_flux = fluxes[idx[0]]
        
        print(f"\nResults:")
        print(f"  Target flux: {target_flux:.2e}")
        print(f"  Minimum concrete thickness: {optimal_thickness:.2f} cm")
        print(f"  Predicted flux: {optimal_flux:.2e}")
        print(f"  Sweep time: {sweep_time:.3f} seconds for {len(concrete_range)} cases")
        print(f"  Equivalent MCNP time: ~{len(concrete_range)*60/60:.1f} hours")
    else:
        print("\nTarget flux not achievable in tested range")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(concrete_range, fluxes, 'b-', linewidth=2)
    plt.axhline(target_flux, color='r', linestyle='--', label='Target flux')
    if len(idx) > 0:
        plt.axvline(optimal_thickness, color='g', linestyle='--', 
                   label=f'Optimal thickness: {optimal_thickness:.1f} cm')
    plt.xlabel('Concrete Thickness (cm)')
    plt.ylabel('Predicted Flux')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Shield Optimization: Flux vs Concrete Thickness')
    plt.savefig('plots/optimization_example.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Loading trained surrogate model...")
    predictor = SurrogatePredictor()
    
    print("\n" + "="*60)
    print("BENCHMARKING SURROGATE MODEL")
    print("="*60)
    
    # Accuracy benchmark
    print("\n1. Testing prediction accuracy...")
    benchmark = BenchmarkComparison(predictor)
    results_df = benchmark.benchmark_accuracy(n_samples=100)
    plot_benchmark_results(results_df)
    
    # Speed benchmark
    print("\n2. Testing prediction speed...")
    speed_results = benchmark.benchmark_speed(n_predictions=1000)
    
    print(f"\nSpeed Benchmark Results:")
    print(f"  Predictions: {speed_results['n_predictions']}")
    print(f"  Total time: {speed_results['total_time']:.3f} seconds")
    print(f"  Time per prediction: {speed_results['time_per_prediction']*1000:.3f} ms")
    print(f"  Estimated MCNP time: {speed_results['estimated_mcnp_time']:.1f} seconds")
    print(f"  Speedup factor: {speed_results['speedup_factor']:.0f}x")
    
    # Demonstrate use case
    demonstrate_use_case(predictor)
    
    print("\n" + "="*60)
    print("Benchmarking complete! Check plots/ directory for visualizations.")
    print("="*60)

if __name__ == '__main__':
    main()
