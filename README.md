# Neutron Shielding Surrogate Model

## Overview
This project develops a neural network surrogate model for neutron transport in multi-layer shielding configurations. The surrogate replaces expensive MCNP6 Monte Carlo simulations with fast neural network predictions, enabling rapid design iterations and parameter optimization.

## Project Structure
```
.
├── 1_generate_mcnp_inputs.py      # Generate MCNP input decks
├── 2_run_mcnp_batch.py            # Run MCNP simulations in parallel
├── 3_train_surrogate_model.py     # Train neural network
├── 4_inference_and_benchmark.py   # Benchmark and demonstrate usage
├── 5_analysis_and_visualization.py # Comprehensive analysis
├── requirements.txt               # Python dependencies
├── manifest.csv                   # Parameter samples (generated)
├── training_data.csv              # MCNP results (generated)
├── best_model.pt                  # Trained model (generated)
├── scaler.pkl                     # Feature scaler (generated)
├── model_metadata.json            # Model configuration (generated)
└── plots/                         # Visualization outputs (generated)
```

## Requirements

### Software
- Python 3.8+
- MCNP6 (with appropriate licenses)
- CUDA-capable GPU (optional, for faster training)

### Python Packages
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
torch>=1.9.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install with:
```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Generate Training Data
```bash
# Generate MCNP input files (creates 2000 cases)
python 1_generate_mcnp_inputs.py

# Run MCNP simulations (modify MCNP path if needed)
# This will take several hours depending on your system
python 2_run_mcnp_batch.py
```

**Note:** Modify `mcnp_exe` path in `2_run_mcnp_batch.py` to point to your MCNP6 installation.

### Step 2: Train Surrogate Model
```bash
# Train neural network on MCNP results
python 3_train_surrogate_model.py
```

This will:
- Load and preprocess MCNP results
- Train a feedforward neural network
- Save the best model and scaler
- Generate training curves

### Step 3: Benchmark and Use Model
```bash
# Benchmark accuracy and speed
python 4_inference_and_benchmark.py

# Run comprehensive analysis
python 5_analysis_and_visualization.py
```

## Usage Examples

### Making Predictions
```python
from inference_and_benchmark import SurrogatePredictor

# Load trained model
predictor = SurrogatePredictor()

# Define shielding configuration
params = {
    'concrete_thick': 50.0,    # cm
    'lead_thick': 5.0,         # cm
    'poly_thick': 10.0,        # cm
    'source_energy': 2.0,      # MeV
    'source_intensity': 1e9    # particles/s
}

# Predict flux
flux = predictor.predict(params)
print(f"Predicted flux: {flux:.2e}")
```

### Parameter Optimization
```python
import numpy as np

# Find minimum concrete thickness for target flux
target_flux = 1e-6
concrete_range = np.linspace(10, 100, 100)

for thickness in concrete_range:
    params['concrete_thick'] = thickness
    flux = predictor.predict(params)
    if flux < target_flux:
        print(f"Optimal thickness: {thickness:.1f} cm")
        break
```

## Model Performance

### Accuracy
- **R² Score**: ~0.95-0.99 (depending on training data quality)
- **Mean Relative Error**: 1-5%
- **Predictions within MCNP uncertainty**: >80% of cases

### Speed
- **Prediction time**: ~0.1-1 ms per case
- **MCNP time**: ~60 seconds per case (1M particles)
- **Speedup factor**: ~10^5x

## Customization

### Adjusting Network Architecture
Edit `3_train_surrogate_model.py`:
```python
model = ShieldingSurrogate(
    input_dim=5,
    hidden_dims=[256, 128, 64, 32],  # Modify layer sizes
    dropout=0.3                       # Adjust dropout rate
)
```

### Changing Parameter Ranges
Edit `1_generate_mcnp_inputs.py`:
```python
self.param_bounds = {
    'concrete_thick': (10, 150),   # Adjust ranges
    'lead_thick': (0, 30),
    # ... etc
}
```

### Adding More Materials
Modify the `materials` dictionary in `MCNPInputGenerator` class.

## Limitations and Considerations

1. **Training Data Coverage**: Model is only reliable within the parameter space used for training. Extrapolation beyond training ranges may be inaccurate.

2. **Geometry Constraints**: Current implementation is limited to 1D slab geometry. 2D/3D geometries require different input representations (e.g., CNNs for grid-based, GNNs for assembly-based).

3. **Physics Fidelity**: Surrogate captures average behavior but may miss edge cases or unusual physics regimes. Always validate critical designs with full MCNP.

4. **Energy Dependence**: Current model predicts total (energy-integrated) flux. For energy-dependent predictions, modify network to output multiple energy groups.

5. **MCNP Statistical Uncertainty**: Training data contains statistical noise from Monte Carlo. Using variance reduction in MCNP can improve surrogate accuracy.

## Extension Ideas

### Short-term (1-2 weeks)
- Multi-energy group predictions
- Angular flux distributions
- Uncertainty quantification using ensemble methods
- Active learning to sample parameter space efficiently

### Medium-term (1-2 months)
- 2D cylindrical geometries
- Time-dependent problems (pulsed sources)
- Inverse problems (source/material identification)
- Physics-informed neural networks (PINNs)

### Long-term (3+ months)
- 3D arbitrary geometries using graph neural networks
- Coupled neutron-photon transport
- Criticality eigenvalue problems
- Integration with CAD-to-MCNP workflows

## Troubleshooting

### MCNP Runs Failing
- Check MCNP installation and PATH
- Verify material definitions (atomic fractions should sum to 1.0)
- Increase NPS if statistical errors are too large

### Training Not Converging
- Increase training data (generate more MCNP cases)
- Adjust learning rate or network architecture
- Check for NaN values in training data
- Use gradient clipping if loss explodes

### Poor Predictions
- Verify parameter is within training range
- Check MCNP statistical uncertainties in training data
- Try log-transforming features with large dynamic ranges
- Collect more training data in poorly-performing regions

## Citation

If you use this code in your research, please cite:

```
@software{neutron_shielding_surrogate,
  title = {Neural Network Surrogate for Neutron Shielding Calculations},
  author = {Kavish Imam},
  year = {2026},
  url = {https://github.com/yourusername/neutron-surrogate}
}
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: kavishimam@gmail.com
