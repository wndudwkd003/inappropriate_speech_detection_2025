import os, json

def save_metrics(metrics, output_dir):
    """Save training metrics to a file."""
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")
