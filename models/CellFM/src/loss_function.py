import scipy as sp
import numpy as np
import mindspore as ms
import mindspore.ops.operations as P
from mindspore import nn,ops
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode    

class EvalReconstructMSE(nn.Cell):

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.mse = MaskedMSE(tag='_val')
        self.mean = ops.ReduceMean(False)

    def construct(self, 
                  raw_nzdata, 
                  masked_nzdata, 
                  nonz_gene, 
                  mask_gene, 
                  zero_idx,
                  variant_cls_raw,
                  label=None
                  ):

        gw_pred, cw_pred = self.net(
            raw_nzdata, masked_nzdata, nonz_gene, mask_gene, zero_idx, variant_cls_raw, None
        )
        gw = self.mse(gw_pred, raw_nzdata, mask_gene)
        cw = self.mse(cw_pred, raw_nzdata, mask_gene)
        if hasattr(gw, "ndim") and gw.ndim > 0: gw = self.mean(gw)
        if hasattr(cw, "ndim") and cw.ndim > 0: cw = self.mean(cw)
        val = (gw + cw)/2.0
        return val, gw, cw

class eval_batch(ms.train.Metric):
    def __init__(self):
        super().__init__()
        self.clear()

    def clear(self):
        self.total_sum = 0.0
        self.gw_sum = 0.0
        self.cw_sum = 0.0
        self.n = 0


    def update(self, *inputs):

        flat_data = []
        for item in inputs:
            if isinstance(item, (tuple, list)):
                flat_data.extend(item)
            else:
                flat_data.append(item)

        if len(flat_data) >= 2:
            
            gw = self._convert(flat_data[0])  
            cw = self._convert(flat_data[1])  

            val = (gw + cw) / 2.0
            
            self.total_sum += val
            self.gw_sum += gw
            self.cw_sum += cw
            self.n += 1
        else:
            print(f"[WARNING] eval_batch inputs count: {len(flat_data)} (Expected 3). Inputs: {flat_data}")


    def _convert(self, x):
        try: return float(x.asnumpy())
        except: return float(np.array(x).mean())


    def eval(self):
        count = max(1, self.n)
        return {
            'loss': self.total_sum / count,
            'gw_loss': self.gw_sum / count,
            'cw_loss': self.cw_sum / count
        }
    

class MaskedMSE(nn.Cell):
    def __init__(self,tag=None,shard=None):
        super().__init__()
        self.tag=tag or ''
        self.sub=P.Sub()
        self.sq=P.Square()
        self.cast=P.Cast()
        self.mean=P.ReduceMean(False)
        self.sum=P.ReduceSum(False)
        self.fill_m=P.Mul()
        self.div=P.Div()
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
        # logger
        self.loss_logger=ops.ScalarSummary()
    def set_shard(self,dp,mp):
        self.sub.shard(((dp,1),(dp,1)))
        self.sq.shard(((dp,1),))
        self.mean.shard(((dp,1),))
        self.sum.shard(((dp,1),))
        self.fill_m.shard(((dp,1),(dp,1)))
        self.div.shard(((),()))
    def construct(self,pred,target,mask=None):
        pred = self.cast(pred, ms.float32)
        target = self.cast(target, ms.float32)
        loss=self.sq(self.sub(pred,target))
        if mask is not None:
            mask=self.cast(mask,ms.float32)
            loss=self.sum(self.fill_m(loss,mask))
            num=self.sum(mask)
            loss=self.div(loss,num)
            self.loss_logger(f'MaskedMSE{self.tag}',loss)
            return loss
        loss=self.mean(loss)
        self.loss_logger(f'MSE{self.tag}',loss)
        return loss

class BCE(nn.Cell):
    def __init__(self,tag='',shard=None):
        super().__init__()
        self.tag=tag
        self.sigmoid=P.Sigmoid()
        self.log=P.Log()
        self.gather=P.Gather(1)
        self.cat=P.Concat(-1)
        self.sub=P.Sub()
        self.div=P.Div()
        self.mul=P.Mul()
        self.eps=ms.Tensor([1e-12])
        self.sum1=P.ReduceSum(False)
        self.sum=P.ReduceSum(False)
        self.mean=P.ReduceMean(False)
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
        self.loss_logger=ops.ScalarSummary()
    def set_shard(self,dp,mp):
        self.sigmoid.shard(((dp,1),))
        self.sub.shard(((),(dp,1)))
        self.cat.shard(((dp,1),(dp,1)))
        self.log.shard(((dp,),))
        self.mul.shard(((dp,1),(dp,1)))
        self.mean.shard(((dp,),))
    def construct(self,pred,target,mask=None):
        pred=P.Reshape()(pred.astype(ms.float32),(-1,1))
        target=P.Reshape()(target.astype(ms.float32),(-1,1))
        pred=self.cat((self.sub(1,pred),pred))
        pred=self.log(ops.clamp(pred,1e-12,1))
        logit=self.cat((self.sub(1,target),target))
        logit=-self.sum1(self.mul(pred,logit),1)
        if mask is None:
            loss=self.mean(logit)
        else:
            mask=P.Reshape()(mask.astype(ms.float32),(-1,1))
            loss=self.div(self.sum(logit),self.sum(mask))
        self.loss_logger(f'BCE{self.tag}',loss)
        return loss

class NLL_loss(nn.Cell):
    def __init__(self,weight=None,reduction='mean',ignore_index=-100):
        super().__init__()
        self.weight=weight or ms.Tensor([1],ms.float32)
        self.reduction=reduction
        self.ignore_index=ignore_index
        self.gather=P.Gather(1)
        self.sum1=P.ReduceSum()
        self.sum2=P.ReduceSum()
        self.mean=P.ReduceMean()
        self.eq=P.Equal()
        self.loss_logger=ops.ScalarSummary()
    def construct(self,pred,target):
        b,n=pred.shape
        pred=pred.astype(ms.float32)
        nll=self.gather(pred,target,1)
        mask=self.eq(target,self.ignore_index)
        mask=(1-mask).astype(ms.float32)
        nll=-mask*nll*self.weight
        if self.reduction=='sum':
            loss=self.sum1(nll)
            self.loss_logger('nll_loss',loss)
        elif self.reduction=='mean':
            loss=self.sum1(nll)/ops.clamp(self.sum2(mask),1)
            self.loss_logger('nll_loss',loss)
        else:
            loss=nll
        return loss