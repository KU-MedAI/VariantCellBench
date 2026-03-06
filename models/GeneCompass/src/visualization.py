# postprocessing.py

import os
import logging
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

def train_valid_lossplot(log_history: List[Dict], output_dir: str, base_run_name, plot_title):

    logger.info("Processing training history to generate plots and metrics...")

    train_points = []
    eval_points = []
    
    for log in log_history:
        if 'loss' in log and 'epoch' in log:
            train_points.append({'epoch': log['epoch'], 'loss': log['loss']})
        elif 'eval_loss' in log and 'epoch' in log:
            eval_points.append({
                'epoch': int(round(log['epoch'])),
                'eval_loss': log['eval_loss'],
                'eval_mse': log.get('eval_mse'),
                'eval_mae': log.get('eval_mae'),
                'eval_r2_score': log.get('eval_r2_score')
            })

    if not train_points or not eval_points:
        logger.warning("No training or evaluation logs found to process. Skipping plot/CSV generation.")
        return

    try:
        train_df = pd.DataFrame(train_points)
        eval_df = pd.DataFrame(eval_points)
        
        plt.figure(figsize=(12, 7))
        plt.plot(train_df['epoch'], train_df['loss'], label='Training Loss', alpha=0.8)
        plt.plot(eval_df['epoch'], eval_df['eval_loss'], label='Validation Loss') 
        
        plt.title(plot_title, fontsize=16)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()

        max_epoch = eval_df['epoch'].max()
        plt.xticks(range(0, max_epoch + 1))
        
        plot_path = os.path.join(output_dir, f"loss_plot_{base_run_name}.png")
        plt.savefig(plot_path)
        logger.info(f"Loss plot saved to {plot_path}")
        plt.close() 

    except Exception as e:
        logger.error(f"Failed to generate or save the loss plot: {e}")


    try:
        train_df['epoch_rounded'] = train_df['epoch'].apply(round)
        last_train_loss_per_epoch = train_df.groupby('epoch_rounded')['loss'].last()

        eval_df['loss'] = eval_df['epoch'].map(last_train_loss_per_epoch)
        
        final_metrics_df = eval_df[['epoch', 'loss', 'eval_loss', 'eval_mse', 'eval_mae', 'eval_r2_score']]
        final_metrics_df = final_metrics_df.set_index('epoch')

        csv_path = os.path.join(output_dir, f"training_metrics_{base_run_name}.csv")
        final_metrics_df.to_csv(csv_path)
        logger.info(f"Training metrics saved to {csv_path}")
        
        print("\n--- Final Training Metrics ---")
        print(final_metrics_df.to_string())

    except Exception as e:
        logger.error(f"Failed to generate or save the metrics CSV: {e}")