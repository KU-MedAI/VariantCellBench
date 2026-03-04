# test.py
from additional_utils import *




# Dataset options:
checkpoint_dir = ""   # Directory containing trained model checkpoints
# Load configuration used during training
with open(f"{checkpoint_dir}/config.json", "r") as f:
    config = json.load(f)

# Load precomputed variant embeddings
with open(
    f"/NFS_DATA/samsung/database/gears/embedding/"
    f"embedding_cache_variant_position_[{config['embedding_model']}]_proto4.pkl",
    "rb"
) as f:
    emb = pickle.load(f)

# # Pretrained scVI model used for cell latent representation
pretrained_scvi = 'kim2023_hct116_[benchmark][3_3-fold]'

original_adata = ad.read_h5ad(
    f"/NFS_DATA/samsung/database/gears/{pretrained_scvi}/perturb_processed_metadata.h5ad"
)

# Trained PerturbNet CINN model path
path_cinn_model_save = (
    "/NFS_DATA/samsung/variantPerturbNet/cinn/"
    "PerturbNet_+_kim2023_hct116_[benchmark][3_3-fold]_+_"
    "esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260104_2113"
)

# Load pretrained scVI model
scvi_model_save_path = f"/NFS_DATA/samsung/variantPerturbNet/cellvae/{pretrained_scvi}"

adata_train = original_adata[original_adata.obs.split != "test", :].copy()

scvi_model = scvi.model.SCVI.load(
    scvi_model_save_path,
    adata=adata_train,
    use_cuda=False
)

# Prepare variant embeddings aligned with dataset conditions
conditions, sequence_representations = cache_to_sequence_representations_from_adata(
    adata_train,
    emb,
    expr_type=config['variant_representation'],
)

# Compute scVI latent representation
adata_train.obsm["X_scVI"] = scvi_model.get_latent_representation(adata_train)
# Store condition ordering and embeddings in AnnData
adata_train.uns['ordered_all_trt'] = conditions
adata_train.uns['ordered_all_embedding'] = sequence_representations

cond_stage_data, embeddings, perturbToEmbed = prepare_embeddings_cinn_2(
    adata_train,
    perturbation_key="condition",   # 어떤 obs 컬럼을 perturbation label로 쓸지
    trt_key="ordered_all_trt",        # adata.uns에 저장된 treatment 순서 키
    embed_key="ordered_all_embedding" # adata.uns에 저장된 임베딩 키
)

# Conditional flow model: variant embedding → cell latent state
flow_model = ConditionalFlatCouplingFlow(
    conditioning_dim=embeddings[0].shape[0],  # variant embedding dimension
    embedding_dim=10,                         # scVI latent dimension
    conditioning_depth=2,
    n_flows=20,
    in_channels=10,
    hidden_dim=1024,
    hidden_depth=2,
    activation="none",
    conditioner_use_bn=True
)

# PerturbNet model wrapper
model_c = Net2NetFlow_scVIFixFlow(
    configured_flow=flow_model,
    cond_stage_data=cond_stage_data,
    perturbToEmbedLib=perturbToEmbed,
    embedData=embeddings,
    scvi_model=scvi_model
)

model_c.adata = adata_train

device = "cuda"



# -------------------------------------------------
# Load the trained Flow (CINN) model
# -------------------------------------------------
model_c.load(path_cinn_model_save)
model_c.to(device=device)


# -------------------------------------------------
# Extract SCVI latent representations for the test set
# -------------------------------------------------
Zsample_test = scvi_model.get_latent_representation(
    adata=adata_test,
    give_mean=False
)

# Set models to evaluation mode
model_c.eval()

# Wrapper for generating data from scVI latent space
scvi_model_de = scvi_predictive_z(scvi_model)

# PerturbNet inference wrapper
perturbnet_model = SCVIZ_CheckNet2Net(
    model_c,
    device,
    scvi_model_de
)

# Sample latent library sizes from the training set
Lsample_obs = scvi_model.get_latent_library_size(
    adata=adata_train,
    give_mean=False
)

# -------------------------------------------------
# Attach latent embeddings and perturbation metadata
# to the test AnnData object
# -------------------------------------------------
adata_test.obsm["X_scVI"] = scvi_model.get_latent_representation(adata_test)

adata_test.uns['ordered_all_trt'] = np.array(conditions)
adata_test.uns['ordered_all_embedding'] = np.array(sequence_representations)


# -------------------------------------------------
# Determine cell line from dataset name
# -------------------------------------------------
data_name = config["data_name"].lower()

if "hct116" in data_name:
    cell_line_l = "hct116"
elif "u2os" in data_name:
    cell_line_l = "u2os"
else:
    raise ValueError(f"Unknown cell line in data_name: {data_name}")

# -------------------------------------------------
# Containers for storing predicted and ground-truth AnnData
# -------------------------------------------------
adata_truth_list = []
adata_pred_list  = []

all_conditions = adata_test.obs["condition"].unique()


for i, unseen_pert in enumerate(all_conditions):

    # -----------------------------
    # 1. Retrieve perturbation embedding
    # -----------------------------
    pert_idx = np.where(
        adata_test.uns["ordered_all_trt"] == unseen_pert
    )[0][0]

    unseen_pert_embed = adata_test.uns["ordered_all_embedding"][pert_idx]

    # -----------------------------
    # 2. Ground-truth data
    # -----------------------------
    cell_idx = np.where(
        adata_test.obs["condition"] == unseen_pert
    )[0]

    real_data = adata_test.layers["counts"].A[cell_idx]
    n_cells = real_data.shape[0]

    # Create AnnData for real observations
    adata_truth = ad.AnnData(
        X=real_data,
        obs=adata_test.obs.iloc[cell_idx].copy(),
        var=adata_test.var.copy()
    )

    adata_truth.obs["source"] = "truth"
    adata_truth.obs["condition"] = unseen_pert

    adata_truth_list.append(adata_truth)

    # -----------------------------
    # 3. Generate predicted samples
    # -----------------------------

    # Sample library size latents from training distribution
    Lsample_idx = np.random.choice(
        range(Lsample_obs.shape[0]),
        n_cells,
        replace=True
    )

    library_trt_latent = Lsample_obs[Lsample_idx]

    # Replicate perturbation embedding for all cells
    trt_input_fixed = np.tile(unseen_pert_embed, (n_cells, 1))

    # Add small Gaussian noise to perturbation embedding
    pert_embed = trt_input_fixed + np.random.normal(
        scale=0.001,
        size=trt_input_fixed.shape
    )

    # Generate predicted gene expression
    predict_latent, predict_data = perturbnet_model.sample_data(
        pert_embed,
        library_trt_latent
    )

    # Create AnnData for predicted samples
    adata_pred = ad.AnnData(
        X=predict_data,
        obs=adata_test.obs.iloc[cell_idx].copy(),
        var=adata_test.var.copy()
    )

    adata_pred.obs["source"] = "pred"
    adata_pred.obs["condition"] = unseen_pert

    adata_pred_list.append(adata_pred)


# -------------------------------------------------
# Concatenate all perturbation-specific AnnData objects
# -------------------------------------------------
adata_truth_all = ad.concat(
    adata_truth_list,
    axis=0,
    merge="same",
    index_unique=None
)

adata_pred_all = ad.concat(
    adata_pred_list,
    axis=0,
    merge="same",
    index_unique=None
)

# Copy metadata
adata_truth_all.uns = adata_test.uns
adata_pred_all.uns  = adata_test.uns

# -------------------------------------------------
# Store raw counts in layers
# -------------------------------------------------
adata_truth_all.layers["counts"] = adata_truth_all.X.copy()
adata_pred_all.layers["counts"]  = adata_pred_all.X.copy()


# -------------------------------------------------
# Normalize gene expression and apply log transform
# -------------------------------------------------
sc.pp.normalize_total(
    adata_truth_all,
    target_sum=TARGET_SUM[cell_line_l],
)
sc.pp.log1p(adata_truth_all)

sc.pp.normalize_total(
    adata_pred_all,
    target_sum=TARGET_SUM[cell_line_l],
)
sc.pp.log1p(adata_pred_all)

# -------------------------------------------------
# Save predicted and ground-truth AnnData objects
# -------------------------------------------------
pred_path  = os.path.join(checkpoint_dir, "pred_adata.h5ad")
truth_path = os.path.join(checkpoint_dir, "truth_adata.h5ad")

adata_pred_all.write(pred_path)
adata_truth_all.write(truth_path)

