"""
2_run_mcnp_batch.py
Runs MCNP6 simulations in parallel and extracts results
"""

import subprocess
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial
import re
import numpy as np
import time

class MCNPRunner:
    """Run MCNP6 simulations"""
    
    def __init__(self, mcnp_exe='mcnp6', n_tasks=4):
        self.mcnp_exe = mcnp_exe
        self.n_tasks = n_tasks
        
    def run_single(self, input_file, output_dir='mcnp_outputs'):
        """Run single MCNP case"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        input_path = Path(input_file)
        case_name = input_path.stem
        
        # MCNP command
        cmd = [
            self.mcnp_exe,
            f'i={input_path}',
            f'n={output_dir / case_name}.',
            f'tasks {self.n_tasks}'
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 min timeout
            )
            
            if result.returncode == 0:
                return {
                    'case': case_name,
                    'status': 'success',
                    'output': str(output_dir / f'{case_name}.o')
                }
            else:
                return {
                    'case': case_name,
                    'status': 'failed',
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'case': case_name,
                'status': 'timeout'
            }
        except Exception as e:
            return {
                'case': case_name,
                'status': 'error',
                'error': str(e)
            }
    
    def run_batch(self, input_files, n_processes=4):
        """Run batch of MCNP cases in parallel"""
        print(f"Running {len(input_files)} cases with {n_processes} processes...")
        
        with mp.Pool(processes=n_processes) as pool:
            results = []
            for i, result in enumerate(pool.imap(self.run_single, input_files)):
                results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{len(input_files)}")
        
        return results

class MCNPOutputParser:
    """Parse MCNP output files to extract tally results"""
    
    def parse_f4_tally(self, output_file):
        """Extract F4 tally results from MCNP output"""
        output_file = Path(output_file)
        
        if not output_file.exists():
            return None
        
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Find F4 tally section
            tally_pattern = r'1tally\s+4.*?energy\s+(.*?)total'
            match = re.search(tally_pattern, content, re.DOTALL)
            
            if not match:
                return None
            
            tally_text = match.group(1)
            
            # Parse energy bins and fluxes
            lines = tally_text.strip().split('\n')
            fluxes = []
            errors = []
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        flux = float(parts[1])
                        error = float(parts[2])
                        fluxes.append(flux)
                        errors.append(error)
                    except ValueError:
                        continue
            
            if fluxes:
                # Return total flux (last value before 'total' line)
                return {
                    'total_flux': fluxes[-1] if fluxes else None,
                    'total_error': errors[-1] if errors else None,
                    'mean_flux': np.mean(fluxes),
                    'max_flux': np.max(fluxes)
                }
            
        except Exception as e:
            print(f"Error parsing {output_file}: {e}")
            return None
        
        return None
    
    def extract_batch_results(self, run_results):
        """Extract tally results from all output files"""
        extracted = []
        
        for result in run_results:
            if result['status'] == 'success':
                tally_data = self.parse_f4_tally(result['output'])
                
                if tally_data:
                    extracted.append({
                        'case': result['case'],
                        **tally_data
                    })
        
        return extracted

def main():
    # Load manifest
    manifest = pd.read_csv('manifest.csv')
    input_files = manifest['filename'].tolist()
    
    print(f"Found {len(input_files)} input files to run")
    
    # Run MCNP simulations
    runner = MCNPRunner(n_tasks=4)
    run_results = runner.run_batch(input_files, n_processes=4)
    
    # Check results
    successful = sum(1 for r in run_results if r['status'] == 'success')
    print(f"\nCompleted: {successful}/{len(run_results)} successful")
    
    # Parse outputs
    print("\nParsing output files...")
    parser = MCNPOutputParser()
    tally_results = parser.extract_batch_results(run_results)
    
    print(f"Extracted {len(tally_results)} tally results")
    
    # Merge with input parameters
    results_df = pd.DataFrame(tally_results)
    
    # Extract case_id from case name
    results_df['case_id'] = results_df['case'].str.extract(r'shield_(\d+)').astype(int)
    
    # Merge with manifest
    final_df = manifest.merge(results_df, on='case_id', how='left')
    
    # Save results
    final_df.to_csv('training_data.csv', index=False)
    print(f"\nTraining data saved to training_data.csv")
    print(f"Total samples with results: {final_df['total_flux'].notna().sum()}")

if __name__ == '__main__':
    main()
