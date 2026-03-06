import os
import mindspore as ms
import glob
import time
import math
import ast
import datetime
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import itertools
import pickle
import mindspore.numpy as mnp
import mindspore.scipy as msc
import mindspore.dataset as ds
from tqdm import tqdm,trange
from mindspore import nn,ops
from scipy.sparse import csr_matrix as csm
from mindspore.amp import FixedLossScaleManager,all_finite,DynamicLossScaleManager
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.communication import init, get_rank, get_group_size
from config import Config
from utils import Wrapper,WrapperWithLossScaleCell,BestModelSaver
from utils import WarmCosineDecay,Adam,AdamWeightDecay,set_weight_decay
from model import CellFM_gene
from data_process import SCrna,input_gene_filtering
from variant_emb import *
from loss_function import EvalReconstructMSE, eval_batch
from visualization import EpochLossCSV
from earlystop import EarlyStopping
import sys
import subprocess


def set_seed(seed=42):
    ms.set_seed(seed)

    import random
    random.seed(seed)
    np.random.seed(seed)


def freeze_module(module,filter_tag=[None]):
    for param in module.trainable_params():
        x=False
        for tag in filter_tag:
            if tag and tag in param.name:
                x=True
                break
        param.requires_grad = x


class PeriodicCheckpoint(ms.train.callback.Callback):
    def __init__(self, save_dir, prefix, interval=5):
        self.save_dir = save_dir
        self.prefix = prefix
        self.interval = interval
    
    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        
        # 5 에폭마다 저장 (5, 10, 15 ...)
        if cur_epoch % self.interval == 0:
            file_name = f"{self.prefix}-{cur_epoch}.ckpt"
            save_path = os.path.join(self.save_dir, file_name)
            ms.save_checkpoint(cb_params.train_network, save_path)
            print(f"Saved periodic checkpoint: {file_name}")


def run_training(args):
    set_seed(args.seed)
    ms.set_context(device_target='GPU', mode=ms.GRAPH_MODE, device_id=args.npu)

    cfg=Config()
    rank_id = None
    rank_size = None

    datapath=f'{args.datapath}'
    savepath=f'{args.modelpath}/checkpoint_{args.date}/'
    analyzep= f'{args.workpath}/analyse_{args.date}'

    # os.makedirs(datapath,  exist_ok=True)
    os.makedirs(savepath,  exist_ok=True)
    os.makedirs(analyzep,  exist_ok=True)

    # LOAD VARIANT EMBEDDING
    if args.emb_name == 'esm2':
        variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[esm2_t33_650M_UR50D].pkl"
    elif args.emb_name == 'protT5':
        variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[ProtT5-XXL-U50].pkl"
    elif args.emb_name == 'msa':
        variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[esm_msa1_t12_100M_UR50S].pkl"
    elif args.emb_name == 'pglm':
         variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[xTrimoPGLM-10B-MLM].pkl"
    elif args.emb_name == 'ankh':
        variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[Ankh3-Large].pkl"

    
    with open(variant_pkl, "rb") as f:
        variant_dict = pickle.load(f)

    try:
        variant_dim = infer_variant_dim_from_pkl(variant_dict)
        print(f"variant embedding dimension: {variant_dim}")
    except Exception as e:
        print(f"Error inferring variant dimension: {e}")
        exit(1)

    tr_data = f'{args.data_name}_train_{args.num_fold}.h5ad'
    val_data = f'{args.data_name}_valid_{args.num_fold}.h5ad'
    ts_data = f'{args.data_name}_test_{args.num_fold}.h5ad'      


    scrna_tr  = SCrna(datapath, tr_data,  embedding_type = args.emb_type, variant_dict=variant_dict, variant_dim=variant_dim, pool='mean')
    scrna_val = SCrna(datapath, val_data, embedding_type = args.emb_type, variant_dict=variant_dict, variant_dim=variant_dim, pool='mean')

    tp53_pos_idx = scrna_tr.tp53_pos_idx
    tp53_gene_id = scrna_tr.tp53_id

    if tp53_pos_idx is not None:
        tp53_gene_id = int(scrna_tr.gene[tp53_pos_idx])
        print(f"[INFO] TP53 Pos Index: {tp53_pos_idx}, Real Gene ID: {tp53_gene_id}")
    else:
        tp53_gene_id = None
        print("[WARNING] TP53 not found in dataset!")

    # TRAIN DATASET
    tr_generator = lambda: input_gene_filtering(scrna_tr, cfg.nonz_len, tp53_pos_idx, shuffle=True)
    tr_dataset = ds.GeneratorDataset(
            source=tr_generator,
            column_names=['raw_nzdata', 'masked_nzdata', 'nonz_gene', 'mask_gene', 'zero_idx', 'variant_cls_raw'],
            shuffle=False,
        )
    tr_dataset = tr_dataset.batch(args.batch, drop_remainder=True, num_parallel_workers=4)

    # VALIDATION DATASET
    val_generator = lambda: input_gene_filtering(scrna_val, cfg.nonz_len, tp53_pos_idx, shuffle=False)
    val_dataset = ds.GeneratorDataset(
            source=val_generator,
            column_names=['raw_nzdata', 'masked_nzdata', 'nonz_gene', 'mask_gene', 'zero_idx', 'variant_cls_raw'],
            shuffle=False,
        )
    val_dataset = val_dataset.batch(args.batch, drop_remainder=False, num_parallel_workers=4)
    
    # DEFINE MODEL
    model = CellFM_gene(len(scrna_tr.geneset), cfg, variant_dim=variant_dim, tp53_idx=tp53_gene_id)

    # LOAD PRETRAINED WEIGHT 
    latest="/NFS_DATA/samsung/foundation/cellFM/base_weight.ckpt"
    print(f'load from {latest}')
    para=ms.load_checkpoint(latest)
    ms.load_param_into_net(model, para)

    if args.lora > 0:
        cfg.recompute = False
        freeze_module(model, ['lora'])

    params = set_weight_decay(model.trainable_params())

    if args.scheduler == 'use':
        steps_per_epoch = tr_dataset.get_dataset_size()
        total_steps     = steps_per_epoch * args.epoch
        warmup_steps    = max(10, steps_per_epoch // 20) 

        lr = WarmCosineDecay(current_step=0, 
                            start_lr=cfg.start_lr,
                            max_lr=cfg.max_lr, 
                            min_lr=cfg.min_lr,
                            warmup_steps=warmup_steps, 
                            decay_steps=max(1, total_steps - warmup_steps))
        
    elif args.scheduler == 'nouse':
        lr = cfg.min_lr

    optimizer = nn.AdamWeightDecay(params,
                                    learning_rate=lr,   
                                    weight_decay=1e-5)

    if args.losscaler == 'use':
        update_cell = nn.DynamicLossScaleUpdateCell(args.losscaler_value, 2, 1000)
        wrapper = WrapperWithLossScaleCell(model, optimizer, update_cell, clip_grad=True, clip_value=1.0)
    else:
        wrapper = Wrapper(model, optimizer, enable_clip=True, clip_value=1.0)


    eval_net = EvalReconstructMSE(model.to_float(ms.float16))
    val_metric_handle = eval_batch()

    metrics = {'val_loss': val_metric_handle}

    trainer=Model(wrapper, amp_level='O0', eval_network=eval_net, metrics=metrics, eval_indexes=[0, 1, 2])  # 
    
    now_str = datetime.datetime.now().strftime("%m%d_%H%M")
    tag = f"{now_str}_{args.data_name}_{args.emb_name}_{args.emb_type}_{args.num_fold}_ep{args.epoch}" 
    print(tag)

    summary_cb = ms.SummaryCollector(
        summary_dir=f"{analyzep}/{tag}", 
        collect_specified_data={"collect_metric": True},
        collect_freq=1, 
        keep_default_action=False, 
    )

    loss_cb = LossMonitor(100)

 
    csv_cb = EpochLossCSV(
        csv_path=f"{analyzep}/{tag}_epoch_losses.csv",
        plot_path=f"{analyzep}/{tag}_loss_plot.png",
        verbose=True,
        val_metric=val_metric_handle,
        plot_title = f"{args.data_name}_{args.emb_name}_{args.emb_type}_{args.num_fold}",
        run_name=tag
        )


    es_cb = EarlyStopping(
        monitor='val_loss', 
        patience=args.patience,         
        warmup_epoch=5,     
        verbose=True, 
        mode='min',          
        save_dir=savepath,  
        prefix=tag           
    )

    periodic_cb = PeriodicCheckpoint(
        save_dir=savepath,
        prefix=tag,
        interval=5
    )

    cbs = [loss_cb, csv_cb, es_cb, periodic_cb]
    if rank_id==0 or rank_id is None:
        cbs.append(summary_cb)


    now=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Begin training {len(tr_dataset)} steps at {now}')

    trainer.fit(
        args.epoch,
        tr_dataset,
        valid_dataset=val_dataset,
        valid_frequency=1,           
        callbacks=cbs,
        dataset_sink_mode=False,  
        valid_dataset_sink_mode=False
    )

    print('Trained finished')




def run_inference(args):
    set_seed(args.seed)

    ms.set_context(device_target='GPU', mode=ms.GRAPH_MODE, device_id=args.npu)
    cfg = Config()

    BATCH_SIZE = getattr(args, 'batch_size', 40) 
    print(f"Inference Batch Size: {BATCH_SIZE}")

    checkpoint_dir = os.path.join(args.modelpath, f'checkpoint_{args.date}')
    datapath = args.datapath
    savedata = f"/NFS_DATA/samsung/CellFM/eval_dataB_{args.date}"
    os.makedirs(savedata, exist_ok=True)
    
    checkpoint_pattern = os.path.join(checkpoint_dir, "*.ckpt")
    models_to_test = glob.glob(checkpoint_pattern)

    if not models_to_test:
        print(f"ERROR: No checkpoints found matching '{checkpoint_pattern}'. Please check the path.")
        return

    print(f"--- Found {len(models_to_test)} checkpoints to process ---")
    for ckpt in models_to_test:
        print(f" - {os.path.basename(ckpt)}")
    print("-" * 50)

    for ckpt_path in models_to_test:
        print(f"\n\n{'='*20} [Processing Model: {os.path.basename(ckpt_path)}] {'='*20}")

        base_name_no_ext = os.path.splitext(os.path.basename(ckpt_path))[0]
        regex = r'(\d{4}_\d{4})_(hct116|u2os)_(protT5|msa|pglm|ankh|esm2)_(alt|diff)_(1-3|2-3|3-3)_ep\d+-best-\d+'
        match = re.search(regex, base_name_no_ext)

        if not match:
            print(f"[WARNING] Checkpoint name '{base_name_no_ext}' does not match expected format. Skipping.")
            continue
        
        date_time, data_name, emb_name, emb_type, num_fold = match.groups()
        print(f"Parsed Info -> Data: {data_name}, Embedding: {emb_name}, Fold: {num_fold}")

        print("Loading corresponding data and variant embeddings...")
        if emb_name == 'esm2':
            variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[esm2_t33_650M_UR50D].pkl"
        elif emb_name == 'protT5':
            variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[ProtT5-XXL-U50].pkl"
        elif emb_name == 'msa':
            variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[esm_msa1_t12_100M_UR50S].pkl"
        elif emb_name == 'pglm':
            variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[xTrimoPGLM-10B-MLM].pkl"
        elif emb_name == 'ankh':
            variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[Ankh3-Large].pkl"

        with open(variant_pkl, "rb") as f:
            variant_dict = pickle.load(f)
        variant_dim = infer_variant_dim_from_pkl(variant_dict)
        
        ts_data_file = f'{data_name}_test_{num_fold}.h5ad'
        scrna_ts = SCrna(datapath, ts_data_file, embedding_type=emb_type, variant_dict=variant_dict, variant_dim=variant_dim, pool='mean')

        tp53_pos_idx = scrna_ts.tp53_pos_idx
        tp53_gene_id = scrna_ts.tp53_id

        if tp53_pos_idx is not None:
            tp53_gene_id = int(scrna_ts.gene[tp53_pos_idx])
            print(f"[INFO] TP53 Pos Index: {tp53_pos_idx}, Real Gene ID: {tp53_gene_id}")
        else:
            tp53_gene_id = None
            print("[WARNING] TP53 not found in dataset!")

        print(f"Initializing model for variant dim: {variant_dim}")
        model = CellFM_gene(len(scrna_ts.geneset), cfg, variant_dim=variant_dim, tp53_idx=tp53_gene_id)
        
        print(f"Loading weights from: {ckpt_path}")
        para = ms.load_checkpoint(ckpt_path)
        ms.load_param_into_net(model, para)
        model.set_train(False)
        model.to_float(ms.float32)
        
        print("Starting prediction ...")

        all_gene_ids_filtered = scrna_ts.gene
        n_genes_to_predict = len(all_gene_ids_filtered)
        n_cells_to_test = len(scrna_ts)
        chunk_size = cfg.nonz_len

        all_gw_preds_full = np.zeros((n_cells_to_test, n_genes_to_predict), dtype=np.float32)
        all_cw_preds_full = np.zeros((n_cells_to_test, n_genes_to_predict), dtype=np.float32)

        num_batches = int(np.ceil(n_cells_to_test / BATCH_SIZE))

        for b_idx in tqdm(range(num_batches), desc="Predicting Batches"):
            start_idx = b_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, n_cells_to_test)
            current_batch_size = end_idx - start_idx
            
            batch_variant_np = np.stack(scrna_ts.variant_cls_raw[start_idx:end_idx]) 
            variant_ms = ms.Tensor(batch_variant_np, dtype=ms.float32)
            
            batch_X_raw = scrna_ts.X[start_idx:end_idx]
            if hasattr(batch_X_raw, "toarray"):
                batch_X_np = batch_X_raw.toarray() 
            else:
                batch_X_np = batch_X_raw
            
            for i in range(0, n_genes_to_predict, chunk_size):
                gene_id_chunk = all_gene_ids_filtered[i : i + chunk_size]
                current_len = len(gene_id_chunk)
                if current_len == 0:
                    continue

                input_gene_batch = np.zeros((current_batch_size, chunk_size), dtype=np.int32)
                input_gene_batch[:, :current_len] = gene_id_chunk

                value_chunk_batch = batch_X_np[:, i : i + chunk_size]
                input_x_data_batch = np.zeros((current_batch_size, chunk_size), dtype=np.float32)
                input_x_data_batch[:, :current_len] = value_chunk_batch

                input_zidx_batch = np.zeros((current_batch_size, chunk_size + 1), dtype=np.float32)
                input_zidx_batch[:, 0] = 1.0
                input_zidx_batch[:, 1 : current_len + 1] = 1.0

                gene_ms = ms.Tensor(input_gene_batch)
                x_data_ms = ms.Tensor(input_x_data_batch)
                zidx_ms = ms.Tensor(input_zidx_batch)

                gw_pred, cw_pred = model(None, x_data_ms, gene_ms, None, zidx_ms, variant_ms)

                gw_pred_np = gw_pred.asnumpy() 
                cw_pred_np = cw_pred.asnumpy()
                
                all_gw_preds_full[start_idx:end_idx, i : i + current_len] = gw_pred_np[:, :current_len]
                all_cw_preds_full[start_idx:end_idx, i : i + current_len] = cw_pred_np[:, :current_len]

        if num_fold:
            fold = num_fold.replace('-', '_') 
        else:
            fold = "unknown"

        master_template_path = f"/NFS_DATA/samsung/database/gears/kim2023_{data_name}_[benchmark][{fold}-fold]/perturb_processed_metadata.h5ad"       
        adata_master = sc.read_h5ad(master_template_path) 
        adata_master = adata_master[adata_master.obs['split']=='test'].copy()
        master_gene_list = adata_master.var_names.tolist()            
        predicted_gene_list = scrna_ts.adata.var_names.tolist()        
    
        variants = scrna_ts.adata.obs['condition']
        counts = variants.groupby(variants, observed=False).cumcount()
        scrna_index = variants.astype(str) + '_' + counts.astype(str)

        raw_y = scrna_ts.adata.layers['y']
        y_dense = raw_y.toarray() if hasattr(raw_y, 'toarray') else raw_y

        df_truth_test = pd.DataFrame(y_dense, index=scrna_index, columns=predicted_gene_list).reindex(columns=master_gene_list, fill_value=0)            
        df_gw_pred_test = pd.DataFrame(all_gw_preds_full, index=scrna_index, columns=predicted_gene_list).reindex(columns=master_gene_list, fill_value=0)            
        df_cw_pred_test = pd.DataFrame(all_cw_preds_full, index=scrna_index, columns=predicted_gene_list).reindex(columns=master_gene_list, fill_value=0)    
     
        raw_x = scrna_ts.adata.layers['x']
        x_dense = raw_x.toarray() if hasattr(raw_x, 'toarray') else raw_x

        num_control_samples = x_dense.shape[0]
        ctrl_index = [f"ctrl_{i}" for i in range(num_control_samples)]
        df_ctrl = pd.DataFrame(x_dense, index=ctrl_index, columns=predicted_gene_list).reindex(columns=master_gene_list, fill_value=0)

        df_truth_final = pd.concat([df_truth_test, df_ctrl])
        df_gw_pred_final = pd.concat([df_gw_pred_test, df_ctrl])
        df_cw_pred_final = pd.concat([df_cw_pred_test, df_ctrl])

        print("\nAssembling and saving final AnnData objects...")

        obs_with_names = adata_master.obs[['condition', 'condition_name']].copy()
        obs_with_names['clean_key'] = obs_with_names['condition'].str.split('+').str[0] 
        condition_map = obs_with_names.drop_duplicates(subset=['clean_key']).set_index('clean_key')['condition_name']

        obs_test = pd.DataFrame(index=scrna_index)
        obs_test['condition'] = scrna_ts.adata.obs['variant'].values
        mapping_keys = obs_test['condition'].str.rsplit('_', n=1).str[0]
        obs_test['condition_name'] = mapping_keys.map(condition_map)

        obs_ctrl = pd.DataFrame(index=ctrl_index)        
        obs_ctrl['condition'] = 'ctrl'

        if data_name == 'hct116':
            obs_ctrl['condition_name'] = 'HCT116_ctrl_1' 
        elif data_name == 'u2os':
            obs_ctrl['condition_name'] = 'U2OS_ctrl_1'
        
        final_obs = pd.concat([obs_test, obs_ctrl])
        final_var = adata_master.var.copy()              
        final_uns = adata_master.uns.copy()

        adata_truth = sc.AnnData(X=df_truth_final.values, obs=final_obs, var=final_var, uns=final_uns)        
        adata_gw_pred = sc.AnnData(X=df_gw_pred_final.values, obs=final_obs, var=final_var, uns=final_uns)
        adata_cw_pred = sc.AnnData(X=df_cw_pred_final.values, obs=final_obs, var=final_var, uns=final_uns)

        filename_base = f"{date_time}_{data_name}_{emb_name}_{emb_type}_{num_fold}"

        truth_path = os.path.join(savedata, f"{filename_base}_truth.h5ad")
        gw_pred_path = os.path.join(savedata, f"{filename_base}_pred.h5ad")
        cw_pred_path = os.path.join(savedata, f"{filename_base}_cw.h5ad")
        
        adata_truth.write_h5ad(truth_path)
        adata_gw_pred.write_h5ad(gw_pred_path)
        adata_cw_pred.write_h5ad(cw_pred_path)

        print(f"Successfully saved AnnData objects to '{savedata}'")
        
     
if __name__ == "__main__":
    pwd=os.getcwd()
    print(f"current path: {pwd}")

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='single_pipeline', 
                        choices=['single_pipeline', 'train', 'inference'])

    parser.add_argument('--npu', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--lora', type=int, default=0)

    parser.add_argument('--emb_name', type=str, choices=['esm2', 'protT5', 'msa', 'pglm', 'ankh']) 
    parser.add_argument('--data_name', type=str, choices=['hct116', 'u2os'])        
    parser.add_argument('--emb_type', type=str, choices=['alt', 'diff'])           
    
    parser.add_argument('--workpath', type=str, default='/home/tech/variantseq/CellFM')
    parser.add_argument('--datapath', type=str, default='/home/tech/variantseq/DATASETS')
    parser.add_argument('--modelpath', default='/NFS_DATA/samsung/CellFM')
    parser.add_argument('--date', type=int, required=True, help='1031, 1103 etc')
    parser.add_argument('--num_fold', type=str)

    parser.add_argument('--scheduler', type=str, default='use', choices=['use', 'nouse'])
    parser.add_argument('--losscaler', type=str, default='nouse', choices=['use', 'nouse'])
    parser.add_argument('--losscaler_value', type=int, default=1)
    parser.add_argument('--earlystop', action='store_true')
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--use_wandb', action='store_true')

    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()    
    
    if args.mode == 'single_pipeline':
        run_training(args)
        run_inference(args)


    elif args.mode == 'train':
        print("--- [SINGLE TRAINING MODE] ---")
        if not all([args.emb_name, args.data_name, args.emb_type]):
            parser.error("--emb_name, --data_name, --emb_type are required for 'train' mode.")
        
        run_training(args)


    elif args.mode == 'inference':
        print("--- [INFERENCE ONLY MODE] ---")
        run_inference(args)
        
        