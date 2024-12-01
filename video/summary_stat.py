import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Optional


def display_experiment_results(experiment_dir: str, verbose: bool = True) -> Dict:
    """
    Display and analyze detailed results from an experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        verbose: Whether to print detailed results to console
        
    Returns:
        Dict containing parsed and analyzed results
    """
    exp_dir = Path(experiment_dir)
    
    # Load accuracy summary
    with open(exp_dir / "accuracy_summary.json", "r") as f:
        accuracy_summary = json.load(f)
    
    # Load experiment config
    with open(exp_dir / "experiment_config.json", "r") as f:
        config = json.load(f)
    
    
    # Create detailed results dictionary
    results = {
        "experiment_info": {
            "directory": str(exp_dir),
            "configuration": config
        },
        "performance_summary": {},
        "detailed_results": accuracy_summary
    }
    
    # Calculate overall performance metrics
    for modality in accuracy_summary:
        modality_results = {
            "unweighted": {
                "mean": accuracy_summary[modality]["unweighted"]["mean"] * 100,
                "std": accuracy_summary[modality]["unweighted"]["std"] * 100,
                "min": min(accuracy_summary[modality]["unweighted"]["values"]) * 100,
                "max": max(accuracy_summary[modality]["unweighted"]["values"]) * 100,
                "fold_scores": [v * 100 for v in accuracy_summary[modality]["unweighted"]["values"]]
            },
            "weighted": {
                "mean": accuracy_summary[modality]["weighted"]["mean"] * 100,
                "std": accuracy_summary[modality]["weighted"]["std"] * 100,
                "min": min(accuracy_summary[modality]["weighted"]["values"]) * 100,
                "max": max(accuracy_summary[modality]["weighted"]["values"]) * 100,
                "fold_scores": [v * 100 for v in accuracy_summary[modality]["weighted"]["values"]]
            }
        }
        results["performance_summary"][modality] = modality_results
    
    if verbose:
        print("\n" + "="*50)
        print(f"Experiment Results Summary")
        print("="*50)
        print(f"\nExperiment Details:")
        print(f"Directory: {results['experiment_info']['directory']}")
        
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"- {key}: {value}")
        
        print("\nPerformance Summary:")
        print("-"*50)
        
        # Create a formatted table for results
        headers = ["Modality", "Metric", "Mean ± Std", "Min", "Max"]
        rows = []
        
        for modality in results["performance_summary"]:
            for metric in ["unweighted", "weighted"]:
                stats = results["performance_summary"][modality][metric]
                rows.append([
                    modality.capitalize(),
                    metric.capitalize(),
                    f"{stats['mean']:.2f} ± {stats['std']:.2f}",
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}"
                ])
        
        # Calculate column widths
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        # Print table
        row_format = "  ".join(f"{{:<{width}}}" for width in col_widths)
        print(row_format.format(*headers))
        print("-" * (sum(col_widths) + 8))
        for row in rows:
            print(row_format.format(*row))
        
        print("\nPer-fold Results:")
        print("-"*50)
        for modality in results["performance_summary"]:
            print(f"\n{modality.capitalize()}:")
            for metric in ["unweighted", "weighted"]:
                scores = results["performance_summary"][modality][metric]["fold_scores"]
                print(f"  {metric.capitalize()} accuracies: " + 
                      ", ".join(f"Fold {i+1}: {score:.2f}%" 
                               for i, score in enumerate(scores)))
        
        print("\n" + "="*50)
    
    return results

if __name__ =='__main__':
    results = display_experiment_results("experiments/20241130_235623_sequential_training")