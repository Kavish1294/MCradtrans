"""
1_generate_mcnp_inputs.py
Generates MCNP6 input decks for multi-layer shielding configurations
"""

import numpy as np
from scipy.stats import qmc
import os
from pathlib import Path

class ShieldingParameterSampler:
    """Generate parameter samples for shielding configurations"""
    
    def __init__(self, n_samples=2000, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        
        # Define parameter bounds
        self.param_bounds = {
            'concrete_thick': (10, 100),  # cm
            'lead_thick': (0, 20),         # cm
            'poly_thick': (0, 30),         # cm
            'source_energy': (0.1, 10.0),  # MeV
            'source_intensity': (1e8, 1e10) # particles/s
        }
        
    def generate_samples(self):
        """Generate Latin Hypercube samples"""
        n_params = len(self.param_bounds)
        sampler = qmc.LatinHypercube(d=n_params, seed=self.seed)
        samples = sampler.random(n=self.n_samples)
        
        # Scale to parameter bounds
        param_names = list(self.param_bounds.keys())
        scaled_samples = np.zeros_like(samples)
        
        for i, param in enumerate(param_names):
            lower, upper = self.param_bounds[param]
            scaled_samples[:, i] = qmc.scale(samples[:, i], lower, upper)
        
        return scaled_samples, param_names

class MCNPInputGenerator:
    """Generate MCNP input decks from parameters"""
    
    def __init__(self, output_dir='mcnp_inputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Material definitions (simplified)
        self.materials = {
            'concrete': {
                'card': 'm1   1001 0.168  6000 0.002  8016 0.562  11023 0.015  '
                       '12000 0.002  13027 0.034  14000 0.337  19000 0.013  '
                       '20000 0.044  26000 0.014',
                'density': -2.3  # g/cm3
            },
            'lead': {
                'card': 'm2   82000 1.0',
                'density': -11.34
            },
            'poly': {
                'card': 'm3   1001 0.667  6000 0.333',
                'density': -0.93
            }
        }
    
    def generate_input(self, params, case_id):
        """Generate single MCNP input file"""
        concrete_t, lead_t, poly_t, energy, intensity = params
        
        # Calculate surface positions
        s1 = 0  # Source position
        s2 = concrete_t
        s3 = s2 + lead_t
        s4 = s3 + poly_t
        detector_pos = s4 + 10  # 10 cm after last shield
        
        input_text = f"""Neutron Shielding Problem - Case {case_id}
c Cell cards
1  1  {self.materials['concrete']['density']}  -2 1      imp:n=1  $ Concrete
2  2  {self.materials['lead']['density']}      -3 2      imp:n=1  $ Lead
3  3  {self.materials['poly']['density']}      -4 3      imp:n=1  $ Polyethylene
4  0                                            -5 4      imp:n=1  $ Void to detector
5  0                                             5        imp:n=0  $ Outside world

c Surface cards
1  pz  {s1:.4f}
2  pz  {s2:.4f}
3  pz  {s3:.4f}
4  pz  {s4:.4f}
5  pz  {detector_pos:.4f}

c Data cards
{self.materials['concrete']['card']}
{self.materials['lead']['card']}
{self.materials['poly']['card']}
sdef  pos=0 0 0  erg={energy:.4f}  par=1
f4:n  4  $ Flux in detector cell
e4    0.01 100 log 20  $ Energy bins
nps   1e6
prdmp 2j -1
print
"""
        
        filename = self.output_dir / f'shield_{case_id:04d}.i'
        with open(filename, 'w') as f:
            f.write(input_text)
        
        return filename
    
    def generate_batch(self, samples, param_names):
        """Generate batch of input files"""
        manifest = []
        
        for i, sample in enumerate(samples):
            filename = self.generate_input(sample, i)
            
            # Store metadata
            metadata = {
                'case_id': i,
                'filename': str(filename),
                **dict(zip(param_names, sample))
            }
            manifest.append(metadata)
        
        return manifest

def main():
    # Generate parameter samples
    print("Generating parameter samples...")
    sampler = ShieldingParameterSampler(n_samples=2000)
    samples, param_names = sampler.generate_samples()
    
    print(f"Generated {len(samples)} samples")
    print(f"Parameters: {param_names}")
    
    # Generate MCNP inputs
    print("\nGenerating MCNP input files...")
    generator = MCNPInputGenerator()
    manifest = generator.generate_batch(samples, param_names)
    
    # Save manifest
    import pandas as pd
    df = pd.DataFrame(manifest)
    df.to_csv('manifest.csv', index=False)
    print(f"\nGenerated {len(manifest)} input files")
    print(f"Manifest saved to manifest.csv")
    
    # Print sample statistics
    print("\nParameter ranges:")
    for param in param_names:
        print(f"  {param}: [{df[param].min():.2f}, {df[param].max():.2f}]")

if __name__ == '__main__':
    main()
