import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_curves(history, fold_no):
    # ... (same code as before)

def aggregate_metrics(val_scores):
    metrics_df = pd.DataFrame(val_scores, columns=['loss', 'accuracy', 'dice_coef', 'iou_coef'])
    avg_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()
    print("Average validation metrics:\n", avg_metrics)
    print("Standard deviation of validation metrics:\n", std_metrics)
    return metrics_df

# Other evaluation functions...
