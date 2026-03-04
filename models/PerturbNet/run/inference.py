# inference.py
from additional_utils import *

# ======================================================
# 0. Global settings
# ======================================================

CELL_LINES = ["HCT116", "U2OS"]
FOLDS = ["1", "2", "3"]
DEVICE = "cuda"

BASE_DATA_DIR = "/NFS_DATA/samsung/database/gears"
BASE_PERTURBNET_DIR = "/NFS_DATA/samsung/variantPerturbNet"
OUT_DIR = "/NFS_DATA/samsung/database/benchmark_figure/ann_dataset_oncoKB/PerturbNet"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_SUM = {
    "hct116": 13750,
    "u2os": 12611,
}

CHECKPOINT_MAP = {
    "HCT116": {
        "1": "PerturbNet_+_kim2023_hct116_[benchmark][1_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260104_2155",
        "2": "PerturbNet_+_kim2023_hct116_[benchmark][2_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260104_2135",
        "3": "PerturbNet_+_kim2023_hct116_[benchmark][3_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260104_2113",
    },
    "U2OS": {
        "1": "PerturbNet_+_kim2023_u2os_[benchmark][1_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260104_2205",
        "2": "PerturbNet_+_kim2023_u2os_[benchmark][2_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260104_2145",
        "3": "PerturbNet_+_kim2023_u2os_[benchmark][3_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260104_2126",
    },
}

# ======================================================
# 1. Load embedding cache (once)
# ======================================================

EMB_MODEL = "esm_msa1_t12_100M_UR50S"
with open(
    f"/NFS_DATA/samsung/database/gears/embedding/"
    f"embedding_cache_variant_position_[{EMB_MODEL}]_proto4.pkl",
    "rb",
) as f:
    EMB_CACHE = pickle.load(f)

# ======================================================
# 2. Main automation loop
# ======================================================

for cell_line in CELL_LINES:
    cell_line_l = cell_line.lower()

    for fold in FOLDS:
        print(f"\n🚀 PerturbNet | {cell_line} | fold {fold}")

        ckpt_name = CHECKPOINT_MAP[cell_line][fold]
        ckpt_dir = f"{BASE_PERTURBNET_DIR}/cinn/{ckpt_name}"

        pretrained_scvi = f"kim2023_{cell_line_l}_[benchmark][{fold}_3-fold]"
        scvi_path = f"{BASE_PERTURBNET_DIR}/cellvae/{pretrained_scvi}"

        # ------------------------------
        # 2.1 Load training AnnData
        # ------------------------------
        adata_train = ad.read_h5ad(
            f"{BASE_DATA_DIR}/{pretrained_scvi}/perturb_processed_metadata.h5ad"
        )
        adata_train = adata_train[adata_train.obs.split != "test"].copy()

        # ------------------------------
        # 2.2 Load scVI
        # ------------------------------
        scvi_model = scvi.model.SCVI.load(
            scvi_path,
            adata=adata_train,
            use_cuda=False,
        )
        scvi_model_de = scvi_predictive_z(scvi_model)
        Lsample_obs = scvi_model.get_latent_library_size(
            adata=adata_train, give_mean=False
        )

        # ------------------------------
        # 2.3 Build CINN model
        # ------------------------------
        conditions, seq_repr = cache_to_sequence_representations_from_adata(
            adata_train,
            EMB_CACHE,
            expr_type="DIFF",
        )
        conditions_arr = np.array(conditions)

        adata_train.obsm["X_scVI"] = scvi_model.get_latent_representation(adata_train)
        adata_train.uns["ordered_all_trt"] = conditions
        adata_train.uns["ordered_all_embedding"] = seq_repr

        _, embeddings, perturbToEmbed = prepare_embeddings_cinn_2(
            adata_train,
            perturbation_key="condition",
            trt_key="ordered_all_trt",
            embed_key="ordered_all_embedding",
        )

        flow_model = ConditionalFlatCouplingFlow(
            conditioning_dim=embeddings[0].shape[0],
            embedding_dim=10,
            conditioning_depth=2,
            n_flows=20,
            in_channels=10,
            hidden_dim=1024,
            hidden_depth=2,
            activation="none",
            conditioner_use_bn=True,
        )

        model_c = Net2NetFlow_scVIFixFlow(
            configured_flow=flow_model,
            cond_stage_data=np.array(adata_train.obs["condition"]),
            perturbToEmbedLib=perturbToEmbed,
            embedData=embeddings,
            scvi_model=scvi_model,
        )
        model_c.adata = adata_train
        model_c.load(ckpt_dir)
        model_c.to(DEVICE)
        model_c.eval()

        perturbnet_model = SCVIZ_CheckNet2Net(
            model_c, DEVICE, scvi_model_de
        )
        Lsample_obs = scvi_model.get_latent_library_size(adata = adata_train, give_mean = False)
        # ------------------------------
        # 2.4 Iterate clinvar chunks
        # ------------------------------
        clinvar_dir = f"{BASE_DATA_DIR}/kim2023_{cell_line_l}_[benchmark][oncoKB]"
        adata_files = sorted(
            os.path.join(clinvar_dir, f)
            for f in os.listdir(clinvar_dir)
            if f.startswith("perturb_processed_") and f.endswith(".h5ad")
        )

        for adata_path in adata_files:
            chunk = os.path.basename(adata_path).replace(".h5ad", "")
            print(f"  🧪 chunk: {chunk}")

            adata_test = ad.read_h5ad(adata_path)
            conditions, seq_repr = cache_to_sequence_representations_from_adata(
                adata_test,
                EMB_CACHE,
                expr_type='DIFF',
            )

            adata_test.obsm["X_scVI"] = scvi_model.get_latent_representation(adata_test)
            adata_test.uns['ordered_all_trt'] = np.array(conditions)
            adata_test.uns['ordered_all_embedding'] = np.array(seq_repr)
            Zsample_test = scvi_model.get_latent_representation(adata = adata_test, give_mean = False)
            scvi_model_de = scvi_predictive_z(scvi_model)
            perturbnet_model = SCVIZ_CheckNet2Net(model_c, DEVICE, scvi_model_de)
            Lsample_obs = scvi_model.get_latent_library_size(adata = adata_train, give_mean = False)

            adata_pred_list = []

            # ==========================
            # Reference inference block
            # ==========================
            for unseen_pert in adata_test.obs["condition"].unique():

                # if unseen_pert not in conditions_arr:
                #     continue
                
                pert_idx = np.where(adata_test.uns["ordered_all_trt"] == unseen_pert)[0][0] 
                unseen_pert_embed = adata_test.uns['ordered_all_embedding'][pert_idx]

                cell_idx = np.where(adata_test.obs["condition"] == unseen_pert)[0]
                n_cells = adata_test.layers["counts"].A[cell_idx].shape[0]

                Lsample_idx = np.random.choice(
                    range(Lsample_obs.shape[0]),
                    n_cells,
                    replace=True,
                )
                library_trt_latent = Lsample_obs[Lsample_idx]

                pert_embed = np.tile(unseen_pert_embed, (n_cells, 1))
                pert_embed += np.random.normal(scale=0.001, size=pert_embed.shape)

                _, predict_data = perturbnet_model.sample_data(
                    pert_embed,
                    library_trt_latent,
                )
                
                if np.isnan(predict_data).any():
                    print(f"[WARN] NaNs in predicted data for {pert} → replaced with 0")
                    predict_data = np.nan_to_num(predict_data, nan=0.0)
                adata_pred = ad.AnnData(
                    X=predict_data,
                    obs=adata_test.obs.iloc[cell_idx].copy(),
                    var=adata_test.var.copy(),
                )
                adata_pred.obs["source"] = "PerturbNet"
                adata_pred.obs["condition"] = str(unseen_pert)

                adata_pred_list.append(adata_pred)

            if len(adata_pred_list) == 0:
                print(f"  [SKIP] empty prediction for {chunk}")
                continue

            # ------------------------------
            # 2.5 concat + normalize + save
            # ------------------------------
            adata_pred_all = ad.concat(
                adata_pred_list,
                axis=0,
                merge="same",
                index_unique=None,
            )

            adata_pred_all.layers["counts"] = adata_pred_all.X.copy()
            sc.pp.normalize_total(
                adata_pred_all,
                target_sum=TARGET_SUM[cell_line_l],
            )
            sc.pp.log1p(adata_pred_all)

            for c in adata_pred_all.obs.columns:
                adata_pred_all.obs[c] = adata_pred_all.obs[c].astype(str)

            out_name = f"{cell_line}_PerturbNet_{fold}-3_{chunk}.h5ad"
            out_path = os.path.join(OUT_DIR, out_name)
            adata_pred_all.write_h5ad(out_path)

            print(f"  ✅ saved: {out_name}")
