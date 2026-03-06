import os
import time
import math
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.scipy as msc
import mindspore.dataset as ds
from tqdm import tqdm,trange
from config import Config
from functools import partial
from scipy.sparse import csr_matrix as csm
from mindspore import nn,ops
from mindspore.context import ParallelMode
from mindspore.train.callback import Callback
from mindspore.communication import init, get_rank, get_group_size
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops.operations.math_ops import NPUGetFloatStatusV2, NPUClearFloatStatusV2
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
clip_grad = ops.MultitypeFuncGraph("clip_grad")
grad_scale = ops.MultitypeFuncGraph("grad_scale")
_grad_overflow = ops.MultitypeFuncGraph("_grad_overflow")
grad_overflow = ops.FloatStatus()
reciprocal = ops.Reciprocal()

@clip_grad.register("Number", "Tensor", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    if clip_type not in (0, 1):
        return grad
    dt = ops.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, -clip_value, clip_value)
    else:
        new_grad = nn.ClipByNorm()(grad, clip_value)
    return new_grad

@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(reciprocal(scale), ops.dtype(grad))


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)

class Wrapper(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0, clip_type=0,clip_value=2.0, enable_clip=True):
        super(Wrapper,self).__init__(network, optimizer, sens)
        self.base0 = ms.Tensor(0, ms.int32)
        self.equal = P.Equal()
        self.logic_not = P.LogicalNot()
        self.allreduce = P.AllReduce()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.reduce_all = P.ReduceAll(keep_dims=False)
        self.less_equal = P.LessEqual()
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = ops.Cast()
        self.hyper_map = ops.HyperMap()
        self.clip_type=clip_type
        self.depend = ops.Depend()
        self.clip_value=ms.Tensor([clip_value,])
        self.enable_clip=enable_clip
        self.overflow_logger=ops.ScalarSummary()
    def set_sens(self, value):
        self.sens = value

    @ms.jit
    def clip_grads(self, grads):
        grads = self.hyper_map(ops.partial(clip_grad, self.clip_type, self.clip_value), grads)
        return grads
    
    def construct(self,*input):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(*input)
        sens = self.cast(ops.tuple_to_array((self.sens,)),ms.float32)
        if isinstance(sens, ms.Tensor):
            # loss1, loss2가 추가되었으므로 3개의 요소를 가진 튜플로 만듭니다.
            sens_tuple = (sens, ops.ZerosLike()(sens), ops.ZerosLike()(sens))
        else:
            sens_tuple = sens
        grads = self.grad(self.network, weights)(*input,sens_tuple)
        grads = self.grad_reducer(grads)
        grads = self.clip_grads(grads)
        loss = self.depend(loss, self.optimizer(grads))
        return loss
    def start_overflow_check(self, pre_cond, compute_input):
        status = ms.Tensor([0] * 8, ms.int32)
        status = F.depend(status, pre_cond)
        # clear overflow buffer
        clear_status = NPUClearFloatStatusV2()(status)
        compute_input = F.depend(compute_input, clear_status)
        return status, compute_input
    @ms.jit
    def get_overflow_status(self, status, compute_output):
        status = F.depend(status, compute_output)
        get_status = NPUGetFloatStatusV2()(status)

        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(get_status)
            # get_status not equal to [0]*8 means overflow
            flag = self.equal(self.base0, flag_reduce)
            status = F.depend(status, flag)
            # distributed needs to skip allreduce to avoid its overflow affecting the next step
            clear_status = NPUClearFloatStatusV2()(status)
            flag = F.depend(flag, clear_status)
            overall_finite = self.reduce_all(flag)
        else:
            status = F.depend(status, get_status)
            clear_status = NPUClearFloatStatusV2()(status)
            get_status = F.depend(get_status, clear_status)
            flag = self.equal(self.base0, get_status)
            overall_finite = self.reduce_all(flag)
        overflow = self.logic_not(overall_finite)
        return overflow



class WrapperWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    
    def __init__(self, network, optimizer, scale_update_cell, 
                 clip_grad=False, clip_value=1.0):
        super(WrapperWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        
        self.cast = ops.Cast()
        self.degree = 1
        # Setup for distributed training
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)

        # Loss scale manager setup
        self.loss_scaling_manager = scale_update_cell
        self.loss_scale = ms.Parameter(ms.Tensor(
            scale_update_cell.get_loss_scale(),
            dtype=ms.float32
        ), name="loss_scale")

        # --- Gradient Clipping Setup ---
        self.clip_grad = clip_grad
        self.clip_value = clip_value
        
        # Operators for internal use
        self.scalar_logger = ops.ScalarSummary()
        self.hyper_map = ops.HyperMap()
        self.div = ops.Div()

    def unscale_grads(self, scale, grads):
        """
        Unscale gradients by dividing by the loss scale value.
        """
        # A small epsilon is added to prevent division by zero, although loss_scale should not be zero.
        scale_with_eps = scale + 1e-8 
        scales_tuple = (scale_with_eps,) * len(grads)

        return self.hyper_map(self.div, grads, scales_tuple)


    def construct(self, *inputs):
        # 1. Load weights
        weights = self.weights
        
        # 2. Calculate loss
        loss = self.network(*inputs)
        
        # 3. Start loss scaling and check for hardware overflow
        scaling_sens = self.loss_scale
        status, scaling_sens_check = self.start_overflow_check(loss, scaling_sens)
        
        # 4. Compute scaled gradients
        scaling_sens_filled = ops.ones_like(loss) * self.cast(scaling_sens, ms.float32)
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        
        # 5. (If distributed) Reduce gradients
        grads = self.grad_reducer(grads)
        
        # 6. Unscale gradients
        grads = self.unscale_grads(scaling_sens, grads)
        
        # --- 7. Apply Gradient Clipping (if enabled) ---
        if self.clip_grad:
            # clip_by_global_norm returns (clipped_grads, global_norm)
            grads = ops.clip_by_global_norm(grads, clip_norm=self.clip_value)
        
        # 8. Check for final overflow status (inf/nan in gradients)
        cond = self.get_overflow_status(status, grads)
        
        # 9. Dynamically update loss scale value
        overflow = self.loss_scaling_manager(self.loss_scale, cond)
        
        # 10. Log overflow status and scale value
        self.scalar_logger('overflow', self.cast(overflow, ms.float32))
        self.scalar_logger('scale', scaling_sens)
        
        # 11. Update weights only if there is no overflow
        if not overflow:
            self.optimizer(grads)
            
        # The return value of scaling_sens_check from start_overflow_check is used
        # as it's the value actually used for scaling.
        return loss, overflow, scaling_sens_check
    
class Adam(nn.Adam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_construct = super().construct
        self.lr_summary = ops.ScalarSummary()
    def construct(self, grads):
        self.lr_summary('lr',self.get_lr())
        return self._original_construct(grads)
    
class AdamWeightDecay(nn.AdamWeightDecay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_construct = super().construct
        self.lr_summary = ops.ScalarSummary()
    def construct(self, grads):
        self.lr_summary('lr',self.get_lr())
        return self._original_construct(grads)
    
class WarmCosineDecay(nn.learning_rate_schedule.LearningRateSchedule):
    def __init__(
        self, 
        current_step,start_lr,
        max_lr, min_lr, 
        warmup_steps, decay_steps
    ):
        super(WarmCosineDecay, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.start_lr = start_lr
        self.cur=current_step
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.math_pi = math.pi
        self.delta = 0.5 * (max_lr - min_lr)
        self.cos = P.Cos()
        self.min = P.Minimum()
        self.cast = P.Cast()
    def construct(self, global_step):
        global_step+=self.cur
        if global_step<self.warmup_steps:
            p=self.cast(self.min(global_step, self.warmup_steps), ms.float32)
            lr=self.start_lr+p/self.warmup_steps*(self.max_lr-self.start_lr)
            return lr
        p = self.cast(self.min(global_step-self.warmup_steps, self.decay_steps), ms.float32)
        lr = self.min_lr+self.delta*(1.0+self.cos(self.math_pi*p/self.decay_steps))
        return lr
    
def set_weight_decay(params,weight_decay=1e-5):
    def decay_filter(x):
        name=x.name.lower()
        tag=True
        tag=tag and ('emb' not in name)
        tag=tag and ('layernorm' not in name)
        tag=tag and ('bias' not in name)
        return tag
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [
        {'params': decay_params,'weight_decay': weight_decay}, 
        {'params': other_params,'weight_decay': 0.0}, 
        {'order_params': params}
    ]
    return group_params

def eval_scib_metrics(
    adata,
    batch_key: str = "str_batch",
    label_key: str = "celltype",
    embed_key: str = "X"
):
    import scib

    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed=embed_key,
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )

    result_dict = results[0].to_dict()
    
    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict


class BestModelSaver(Callback):

    def __init__(self, save_dir, prefix, verbose=True):
        super(BestModelSaver, self).__init__()
        self.save_dir = save_dir
        self.prefix = prefix  # 파일명 앞부분 (예: ..._ep10)
        self.verbose = verbose
        self._best_val_loss = float('inf')
        self.last_ckpt_path = None # 이전에 저장된 파일 경로 기억

    def on_eval_epoch_end(self, run_context):
        """
        Called after evaluation.
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        
        # Model.fit 사용 시 metrics에 결과가 담깁니다.
        metrics = cb_params.metrics
        
        if metrics and 'val_loss' in metrics:
            current_val_loss = float(metrics['val_loss'])
            
            # 현재 val_loss가 기록된 최고 성능보다 좋은지 확인
            if current_val_loss < self._best_val_loss:
                # 새 파일명 생성: prefix 뒤에 -{epoch} 추가
                # 결과 예시: ..._ep10-5.ckpt
                new_filename = f"{self.prefix}-{cur_epoch}.ckpt"
                new_save_path = os.path.join(self.save_dir, new_filename)

                if self.verbose:
                    print(f"\nValidation loss improved from {self._best_val_loss:.6f} to {current_val_loss:.6f}.")
                    print(f"Saving best model to {new_save_path}")

                # ★★★ 네트워크 저장 ★★★
                ms.save_checkpoint(cb_params.network, new_save_path)
                
                # 이전 베스트 파일 삭제 (용량 관리)
                if self.last_ckpt_path and os.path.exists(self.last_ckpt_path):
                    if self.last_ckpt_path != new_save_path:
                        try:
                            os.remove(self.last_ckpt_path)
                        except OSError as e:
                            print(f"[Warning] Failed to delete old checkpoint: {e}")

                # 상태 업데이트
                self._best_val_loss = current_val_loss
                self.last_ckpt_path = new_save_path