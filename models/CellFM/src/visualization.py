import os, csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mindspore.train.callback import Callback
import wandb
import pandas as pd

try:
    import wandb
except ImportError:
    wandb = None

def _2float(x):
    try: 
        return float(x.asnumpy())
    except AttributeError:
        return float(np.array(x).reshape(-1)[0])

class EpochLossCSV(Callback):
    def __init__(self, csv_path, plot_path=None, verbose=True, val_metric=None, plot_title=None, wandb_project=None, run_name=None, wandb_config=None):
        super().__init__()
        self.csv_path = csv_path
        self.plot_path = plot_path
        self.verbose = verbose
        self.plot_title = plot_title
        self.val_metric = val_metric
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "tr_total", "tr_gw", "tr_cw", "val_total", "val_gw", "val_cw"])

        self.history = {}
        self._reset_buffers()

    def _reset_buffers(self):
        self.tr_total = 0.0
        self.tr_gw = 0.0
        self.tr_cw = 0.0
        self.tr_n = 0

    def on_train_step_end(self, run_context):

        outputs = run_context.original_args().net_outputs
        
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
            total, gw, cw = outputs[0], outputs[1], outputs[2]

        self.tr_total += _2float(total)
        self.tr_gw += _2float(gw)
        self.tr_cw += _2float(cw)
        self.tr_n += 1

    def on_eval_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = int(cb_params.cur_epoch_num)

        count = max(1, self.tr_n)
        tr_res = {
            'tr_total': self.tr_total / count,
            'tr_gw': self.tr_gw / count,
            'tr_cw': self.tr_cw / count
        }

        val_res = {'val_total': np.nan, 'val_gw': np.nan, 'val_cw': np.nan}

        target = None

        if self.val_metric is not None:
            try:
                target = self.val_metric.eval() 
            except Exception as e:
                print(f"[WARNING] Failed to eval() injected metric: {e}")
            
        if isinstance(target, dict):
            val_res['val_total'] = float(target.get('loss', np.nan))
            val_res['val_gw'] = float(target.get('gw_loss', np.nan))
            val_res['val_cw'] = float(target.get('cw_loss', np.nan))
        else:
            if self.verbose:
                print(f"[WARNING] 'val_loss' data is not dict. Type: {type(target)}")

        self.history[epoch] = {**tr_res, **val_res}

        if self.verbose:
            print(f"[Ep {epoch}] Tr Total: {tr_res['tr_total']:.4f} | Val Total: {val_res['val_total']:.4f}")
            print(f"         (Tr GW: {tr_res['tr_gw']:.4f}, CW: {tr_res['tr_cw']:.4f})")
            print(f"         (Val GW: {val_res['val_gw']:.4f}, CW: {val_res['val_cw']:.4f})")

        self._rewrite_csv()
        if self.plot_path: self._save_plot()
        
        self._reset_buffers()

    def _rewrite_csv(self):
        epochs = sorted(self.history.keys())
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "tr_total", "tr_gw", "tr_cw", "val_total", "val_gw", "val_cw"])
            for e in epochs:
                d = self.history[e]
                w.writerow([e, d['tr_total'], d['tr_gw'], d['tr_cw'], d['val_total'], d['val_gw'], d['val_cw']])

    def _save_plot(self):
        epochs = sorted(self.history.keys())
        tr_gw = [self.history[e]['tr_gw'] for e in epochs]
        tr_cw = [self.history[e]['tr_cw'] for e in epochs]
        val_gw = [self.history[e]['val_gw'] for e in epochs]
        val_cw = [self.history[e]['val_cw'] for e in epochs]

        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, tr_gw, 'b-', label="Train GW")  
        plt.plot(epochs, tr_cw, 'b--', label="Train CW") 
        plt.plot(epochs, val_gw, 'r-', label="Val GW")   
        plt.plot(epochs, val_cw, 'r--', label="Val CW")  

        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(self.plot_title or "GW vs CW Loss")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        plt.savefig(self.plot_path, dpi=150, bbox_inches="tight")
        plt.close()

