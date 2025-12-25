'''
Runs the quantile regression benchmarks for different models and functions.
'''
import numpy as np
import sys
import time
from pathlib import Path

from funcs import Sanity, Scenario1, Scenario2, Scenario3, Scenario4, Scenario5,\
                  MultivariateScenario1, MultivariateScenario2
from neural_sqerr import SqErrNetwork
from neural_model import QuantileNetwork
from spline_model import QuantileSpline
# from forest_model import QuantileForest
from visualize import heatmap_from_points

# Simple logger
class DualLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout
        
    def write(self, message):
        self.original_stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()
    
    def close(self):
        sys.stdout = self.original_stdout
        self.log_file.close()

def run_benchmarks(demo=True):
    N_trials = 1 if demo else 100
    N_test = 10000
    sample_sizes = [100, 1000, 10000]
    quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    functions = [Scenario1(), Scenario2(), Scenario3(), Scenario4(), Scenario5()]
    models = [lambda: SqErrNetwork(),
              lambda: QuantileNetwork(quantiles=quantiles)]

    # Track the performance results
    mse_results = np.full((N_trials, len(functions), len(models), len(sample_sizes), len(quantiles)), np.nan)
    print(f'Results shape: {mse_results.shape}')

    Path('plots').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)

    for trial in range(N_trials):
        print(f'Trial {trial+1}/{N_trials}')
        for scenario, func in enumerate(functions):
            print(f'  Scenario {scenario+1}')

            # Sample test set covariates and response
            X_test = np.random.random(size=(N_test,func.n_in))
            y_test = func.sample(X_test)

            # Get the ground truth quantiles
            y_quantiles = np.array([func.quantile(X_test, q) for q in quantiles]).T

            # Demo plotting
            if demo and scenario == 0:
                for qidx, q in enumerate((quantiles*100).astype(int)):
                    heatmap_from_points(f'plots/scenario{scenario+1}-quantile{q}-truth.pdf', X_test[:,:2], y_quantiles[:,qidx], vmin=y_quantiles.min(), vmax=y_quantiles.max())

            for nidx, N_train in enumerate(sample_sizes):
                print(f'    N={N_train}')
                # Sample training covariates and response
                X_train = np.random.random(size=(N_train,func.n_in))
                y_train = func.sample(X_train)

                # Evaluate each of the quantile models
                for midx, model in enumerate([m() for m in models]):
                    print(f'      {model.label}')

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    # Evaluate the model on the ground truth quantiles
                    mse_results[trial, scenario, midx, nidx] = ((y_quantiles - preds)**2).mean(axis=0)

                    # Demo plotting
                    if demo and scenario == 0 and nidx == 0:
                        for qidx, q in enumerate((quantiles*100).astype(int)):
                            heatmap_from_points(f'plots/scenario{scenario+1}-quantile{q}-n{N_train}-{model.filename}.pdf', X_test[:,:2],
                                                    preds[:,qidx] if preds.shape[1] > qidx else preds[:,-1],
                                                    vmin=y_quantiles.min(), vmax=y_quantiles.max(),
                                                    colorbar=midx == len(models)-1)

            print(f'  Results: {mse_results[trial, scenario]}')

        if not demo:
            np.save('data/mse_results.npy', mse_results)

    print('\nFinal MSE Results (mean across trials):')
    mean_mse = np.nanmean(mse_results, axis=0)
    for scenario in range(len(functions)):
        print(f'\nScenario {scenario+1}:')
        for midx, model in enumerate([m() for m in models]):
            print(f'  {model.label}:')
            for nidx, N_train in enumerate(sample_sizes):
                print(f'    N={N_train}: {mean_mse[scenario, midx, nidx]}')

    return mse_results

if __name__ == '__main__':
    # Reproducibility
    np.random.seed(42)
    import torch
    torch.manual_seed(42)

    np.set_printoptions(precision=3, suppress=True)
    
    # Setup logging
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Path('logs').mkdir(exist_ok=True)
    log_file = Path('logs') / f'benchmark_original_demo_{current_time}.log'
    
    logger = DualLogger(log_file)
    sys.stdout = logger
    
    print(f"{'='*80}")
    print(f"Original Quantile Regression Benchmark")
    print(f"{'='*80}")
    print(f"Mode: Demo (1 trial)")
    print(f"Time: {current_time}")
    print(f"Log file: {log_file}")
    print(f"{'='*80}\n")
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start_time = time.time()
        run_benchmarks(demo=True)
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print('✅ Demo completed!')
        print(f'📊 Check plots/ directory for visualizations')
        print(f'⏱️  Total time: {elapsed_time:.2f}s')
        print(f'📝 Log saved to: {log_file}')
        print(f"{'='*80}")
    
    logger.close()
