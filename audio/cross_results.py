import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def analyze_fold_predictions(experiment_dir: str) -> Dict:
    """
    Analyze predictions across all folds to calculate accuracies by dataset, 
    emotion class, and dataset-emotion combinations.
    
    Args:
        experiment_dir: Path to the experiment directory containing fold predictions
        
    Returns:
        Dictionary containing detailed accuracy metrics
    """
    # Emotion label mapping
    emotion_mapping = {
        0: "angry",
        1: "happy",
        2: "neutral",
        3: "sad"
    }
    
    exp_path = Path(experiment_dir)
    pred_files = list((exp_path / "fold_predictions").glob("*_predictions.csv"))
    
    # Initialize containers for aggregating results
    dataset_metrics = defaultdict(list)
    emotion_metrics = defaultdict(list)
    dataset_emotion_metrics = defaultdict(list)
    
    # Process each fold's predictions
    for pred_file in pred_files:
        df = pd.read_csv(pred_file)
        
        # Calculate accuracies by dataset
        for dataset in df['dataset'].unique():
            mask = df['dataset'] == dataset
            acc = (df[mask]['prediction'] == df[mask]['ground_truth']).mean()
            dataset_metrics[dataset].append(acc)
            
            # Calculate accuracies by emotion within each dataset
            dataset_df = df[mask]
            for emotion in dataset_df['ground_truth'].unique():
                emotion_mask = dataset_df['ground_truth'] == emotion
                if emotion_mask.sum() > 0:  # Only calculate if we have samples
                    acc = (dataset_df[emotion_mask]['prediction'] == 
                          dataset_df[emotion_mask]['ground_truth']).mean()
                    dataset_emotion_metrics[f"{dataset}-{emotion_mapping[emotion]}"].append(acc)
        
        # Calculate accuracies by emotion class
        for emotion in df['ground_truth'].unique():
            mask = df['ground_truth'] == emotion
            acc = (df[mask]['prediction'] == df[mask]['ground_truth']).mean()
            emotion_metrics[emotion_mapping[emotion]].append(acc)
    
    # Compute summary statistics
    results = {
        'dataset_accuracies': {
            dataset: {
                'mean': np.mean(accs) * 100,
                'std': np.std(accs) * 100,
                'n_folds': len(accs)
            }
            for dataset, accs in dataset_metrics.items()
        },
        'emotion_accuracies': {
            emotion: {
                'mean': np.mean(accs) * 100,
                'std': np.std(accs) * 100,
                'n_folds': len(accs)
            }
            for emotion, accs in emotion_metrics.items()
        },
        'dataset_emotion_accuracies': {
            combo: {
                'mean': np.mean(accs) * 100,
                'std': np.std(accs) * 100,
                'n_folds': len(accs)
            }
            for combo, accs in dataset_emotion_metrics.items()
        }
    }
    
    return results

def print_accuracy_summary(results: Dict) -> None:
    """
    Print a formatted summary of the accuracy results.
    """
    print("\nAccuracies by Dataset:")
    print("-" * 50)
    for dataset, metrics in results['dataset_accuracies'].items():
        print(f"{dataset:<15} {metrics['mean']:6.2f}% (±{metrics['std']:4.2f})")
    
    print("\nAccuracies by Emotion:")
    print("-" * 50)
    for emotion, metrics in results['emotion_accuracies'].items():
        print(f"{emotion:<15} {metrics['mean']:6.2f}% (±{metrics['std']:4.2f})")
    
    print("\nAccuracies by Dataset-Emotion Combination:")
    print("-" * 50)
    for combo, metrics in results['dataset_emotion_accuracies'].items():
        print(f"{combo:<25} {metrics['mean']:6.2f}% (±{metrics['std']:4.2f})")

    # Print summary analysis
    print("\nKey Observations:")
    print("-" * 50)
    
    # Best and worst dataset
    dataset_means = {k: v['mean'] for k, v in results['dataset_accuracies'].items()}
    best_dataset = max(dataset_means.items(), key=lambda x: x[1])
    worst_dataset = min(dataset_means.items(), key=lambda x: x[1])
    print(f"Best performing dataset: {best_dataset[0]} ({best_dataset[1]:.2f}%)")
    print(f"Worst performing dataset: {worst_dataset[0]} ({worst_dataset[1]:.2f}%)")
    
    # Best and worst emotion
    emotion_means = {k: v['mean'] for k, v in results['emotion_accuracies'].items()}
    best_emotion = max(emotion_means.items(), key=lambda x: x[1])
    worst_emotion = min(emotion_means.items(), key=lambda x: x[1])
    print(f"Best recognized emotion: {best_emotion[0]} ({best_emotion[1]:.2f}%)")
    print(f"Most challenging emotion: {worst_emotion[0]} ({worst_emotion[1]:.2f}%)")
    
    # Best and worst combination
    combo_means = {k: v['mean'] for k, v in results['dataset_emotion_accuracies'].items()}
    best_combo = max(combo_means.items(), key=lambda x: x[1])
    worst_combo = min(combo_means.items(), key=lambda x: x[1])
    print(f"Best dataset-emotion combination: {best_combo[0]} ({best_combo[1]:.2f}%)")
    print(f"Worst dataset-emotion combination: {worst_combo[0]} ({worst_combo[1]:.2f}%)")

if __name__ =='__main__':
    results = analyze_fold_predictions("experiments/20241130_064549_two_phase_training")
    print_accuracy_summary(results)