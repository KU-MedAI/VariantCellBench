import os
import copy
import numpy as np
import mindspore as ms
from mindspore import nn,ops
from mindspore.common.tensor import Tensor
from mindspore import _checkparam as Validator
from mindspore.train.serialization import load_param_into_net
from mindspore import log as logger
from mindspore.ops import ReduceOp
from mindspore.communication import get_group_size,get_rank
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train.callback._callback import Callback, _handle_loss


class EarlyStopping(Callback):

    def __init__(self, monitor='val_loss', patience=5, verbose=True, delta=0.0,
                 mode='min', warmup_epoch=0, save_dir=None, prefix=None):

        super(SimpleEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.warmup_epoch = warmup_epoch
        self.save_dir = save_dir
        self.prefix = prefix        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.best_model_path = None 

        if mode not in ['min', 'max']:
            raise ValueError(f"mode '{mode}' is unknown, choose from 'min' or 'max'")
        self.mode = mode

        if self.mode == 'min':
            self.best_score = np.inf
        else:
            self.best_score = -np.inf

    def on_train_begin(self, run_context):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = np.inf if self.mode == 'min' else -np.inf
        self.best_epoch = 0

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        
        eval_results = cb_params.get("eval_results", {})
        monitor_result = eval_results.get(self.monitor)
        if monitor_result is None:
            return

        if isinstance(monitor_result, dict):
            current_score = monitor_result.get('loss')
        else:
            current_score = monitor_result
        cur_epoch = cb_params.get("cur_epoch_num")
        
        if current_score is None:
            return

        if cur_epoch <= self.warmup_epoch:
             if self.verbose:
                print(f"\nEpoch {cur_epoch}: In warmup phase. (Loss: {current_score:.6f})")
             return

        if cur_epoch == self.warmup_epoch + 1:
            self.best_score = np.inf if self.mode == 'min' else -np.inf


        is_improvement = False
        if self.mode == 'min':
            if current_score < self.best_score - self.delta:
                is_improvement = True
        else: 
            if current_score > self.best_score + self.delta:
                is_improvement = True
        
        if is_improvement:
            prev_best = self.best_score
            self.best_score = current_score
            self.wait = 0
            self.best_epoch = cur_epoch
            
            new_filename = f"{self.prefix}-best-{cur_epoch}.ckpt"
            new_save_path = os.path.join(self.save_dir, new_filename)

            if self.verbose:
                print(f"\nEpoch {cur_epoch}: {self.monitor} improved from {prev_best:.6f} to {self.best_score:.6f}. Saving to {new_filename}")

            ms.save_checkpoint(cb_params.train_network, new_save_path)
            
            if self.best_model_path and os.path.exists(self.best_model_path) and self.best_model_path != new_save_path:
                try:
                    os.remove(self.best_model_path)
                except Exception:
                    pass
            
            self.best_model_path = new_save_path

        else: 
            if cur_epoch > self.warmup_epoch:
                self.wait += 1
                if self.verbose:
                    print(f"\nEpoch {cur_epoch}: {self.monitor} did not improve. Patience: {self.wait}/{self.patience}")

                if self.wait >= self.patience:
                    self.stopped_epoch = cur_epoch
                    run_context.request_stop()
                    if self.verbose:
                        print(f"Epoch {self.stopped_epoch}: Early stopping triggered.")
            else:
                if self.verbose:
                    print(f"\nEpoch {cur_epoch}: In warmup phase (max {self.warmup_epoch}). Patience not counted.")

    def on_train_end(self, run_context):
        if self.best_model_path and os.path.exists(self.best_model_path):
            if self.verbose:
                print(f"Restoring best model from {self.best_model_path}")
            
            param_dict = ms.load_checkpoint(self.best_model_path)
            ms.load_param_into_net(run_context.original_args().train_network, param_dict)
