
import argparse
import os
import shutil
import subprocess
import scanpy as sc
import pandas as pd
import numpy as np
import sys

def prepare_data(truth_path, pred_path, temp_dir):
    print(f"[Python] Loading input files...\nTruth: {truth_path}\nPred: {pred_path}")
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    try:
        adata_true = sc.read_h5ad(truth_path)
        print(f"[Python] Truth data shape: {adata_true.shape}")

        adata_true.obs.to_csv(os.path.join(temp_dir, "truth_obs.csv"))
        adata_true.var.to_csv(os.path.join(temp_dir, "truth_var.csv"))

        X_true = adata_true.X
        if hasattr(X_true, "toarray"):
            X_true = X_true.toarray()
        np.savetxt(os.path.join(temp_dir, "truth_X.csv"), X_true, delimiter=",")
        
    except Exception as e:
        print(f"[Error] Failed to process Truth file: {e}")
        sys.exit(1)

    try:
        adata_pred = sc.read_h5ad(pred_path)
        print(f"[Python] Pred data shape: {adata_pred.shape}")
        
        X_pred = adata_pred.X
        if hasattr(X_pred, "toarray"):
            X_pred = X_pred.toarray()
        np.savetxt(os.path.join(temp_dir, "pred_X.csv"), X_pred, delimiter=",")
        
    except Exception as e:
        print(f"[Error] Failed to process Pred file: {e}")
        sys.exit(1)
        
    print(f"[Python] CSV files prepared in: {temp_dir}")

def run_r_script(temp_dir, output_dir):
    print("[Python] Launching R script inside 'r-cmap-env'...")
    
    r_script_path = os.path.join(os.path.dirname(__file__), "analysis.R")
    
    if not os.path.exists(r_script_path):
        print(f"[Error] 'analysis.R' not found at {r_script_path}")
        sys.exit(1)

    conda_env_name = "r-cmap-env"
    cmd = [
        "conda", "run",          
        "-n", conda_env_name,     
        "--no-capture-output",     
        "Rscript",                
        r_script_path,             
        temp_dir,                 
        output_dir                  
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("[R Warning/Message]:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print("[Error] R script failed!")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="AUPRC Analysis Pipeline: Python -> R")
    parser.add_argument("--truth", type=str, required=True, help="Path to truth.h5ad")
    parser.add_argument("--pred", type=str, required=True, help="Path to pred.h5ad")
    parser.add_argument("--output", type=str, default="./auprc_paper_results", help="Directory to save PNGs and CSV")
    parser.add_argument("--keep_temp", action="store_true", help="Keep intermediate CSV files")
    
    args = parser.parse_args()
    

    temp_dir = os.path.join(args.output, "temp_csv_data")
    prepare_data(args.truth, args.pred, temp_dir)
    run_r_script(temp_dir, args.output)

    if not args.keep_temp:
        print(f"[Python] Removing temporary files in {temp_dir}...")
        shutil.rmtree(temp_dir)
    
    print(f"[Python] Pipeline finished. Check results in: {args.output}")

if __name__ == "__main__":
    main()