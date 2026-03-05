# train.py
# sys.path.append(os.path.abspath("../models/PerturbNet/src"))
from ..src.additional_utils import *

TARGET_SUM = {
    "hct116": 13750,
    "u2os": 12611,
}

config = {
    'model':'PerturbNet',                                 # Name of the model being trained
    'data_name':'kim2023_hct116_[benchmark][3_3-fold]',   # Name of the training dataset
    'adata_name':'perturb_processed_metadata',            # AnnData file name
    'embedding_model': 'esm2_t33_650M_UR50D',             # Variant embedding model
                                                           # Options: 'esm2_t33_650M_UR50D'
                                                           #          'esm_msa1_t12_100M_UR50S'
                                                           #          'ProtT5-XXL-U50'
                                                           #          'Ankh3-Large'
                                                           #          'xTrimoPGLM-10B-MLM'
    'variant_representation': 'ALT',                      # Variant representation type: 'ALT' or 'DIFF'
    'compression': 'position_embedding',                  # Embedding compression strategy:
                                                           # 'position_embedding' or 'full_sequence_average'
}

# # --- read override json if provided -- #
# import os, json
# _override_path = os.environ.get("CONFIG_OVERRIDE_PATH")
# if _override_path and os.path.exists(_override_path):
#     with open(_override_path) as f:
#         config.update(json.load(f))
# # ------------------------------------- #

adata = ad.read_h5ad(f"/NFS_DATA/samsung/database/gears/{config['data_name']}/{config['adata_name']}.h5ad")

# Load precomputed variant embedding cache
with open(f"/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[{config['embedding_model']}]_proto4.pkl", 'rb') as f:
    emb = pickle.load(f)

import re
import numpy as np

# Generate sequence representations aligned with AnnData conditions
conditions, sequence_representations = cache_to_sequence_representations_from_adata(
    adata,
    emb,
    expr_type=config['variant_representation'],
)

# -------------------------------------------------
# Split dataset into train and test sets based on obs["split"]
# -------------------------------------------------

# Select cells where obs["split"] == "train"
adata_train = adata[adata.obs.split == "train", :].copy()

# Select cells where obs["split"] == "test"
adata_test = adata[adata.obs.split == "test", :].copy()

# -------------------------------------------------
# Path for saving the trained SCVI model
# -------------------------------------------------
scvi_model_save_path = f"/NFS_DATA/samsung/variantPerturbNet/cellvae/{config['data_name']}"

# -------------------------------------------------
# Configure AnnData for SCVI (using the counts layer)
# ⚠ setup_anndata must be called both during training and loading
# -------------------------------------------------
scvi.data.setup_anndata(
    adata_train,
    layer="counts"
)

# -------------------------------------------------
# Train SCVI model if no saved model exists
# -------------------------------------------------
if not os.path.exists(scvi_model_save_path):

    print("SCVI model not found. Training a new model...")

    # Latent space dimension = 10
    scvi_model = scvi.model.SCVI(
        adata_train,
        n_latent=10
    )

    # Recommended training setting from tutorials: 700 epochs
    scvi_model.train(
        n_epochs=700,
        frequency=20  # record training history every 20 steps
    )

    # Save trained model
    scvi_model.save(scvi_model_save_path)
    print(f"SCVI model saved to: {scvi_model_save_path}")

# -------------------------------------------------
# Load existing SCVI model if available
# -------------------------------------------------
else:
    print("Loading existing SCVI model...")

    scvi_model = scvi.model.SCVI.load(
        scvi_model_save_path,
        adata=adata_train,
        use_cuda=False  # set True if using GPU
    )

    print("SCVI model loaded successfully.")


# -------------------------------------------------
# Compute SCVI latent representations for training cells
# -------------------------------------------------

# Select all cells except the test set
adata_train = adata[adata.obs.split != "test", :].copy()

# Store latent embedding in obsm
adata_train.obsm["X_scVI"] = scvi_model.get_latent_representation(adata_train)

# Variant-level metadata used by PerturbNet
adata_train.uns['ordered_all_trt'] = conditions
adata_train.uns['ordered_all_embedding'] = sequence_representations



# Prepare condition labels, variant embeddings, and mapping for CINN(flow)

# prepare_embeddings_cinn returns:
#  - cond_stage_data : perturbation label for each cell
#  - embeddings      : PLM-derived variant embeddings
#  - perturbToEmbed  : mapping between perturbation label and embedding index
cond_stage_data, embeddings, perturbToEmbed = prepare_embeddings_cinn_2(
    adata_train,
    perturbation_key="condition",      # obs column used as perturbation label
    trt_key="ordered_all_trt",         # key storing perturbation order in adata.uns
    embed_key="ordered_all_embedding"  # key storing variant embeddings in adata.uns
)

# Attach AnnData to the scVI model
scvi_model.adata = adata_train


# Define Conditional INN (Flow) model and create Net2NetFlow wrapper
# Conditional flow model:
# - condition: variant embedding (e.g., 1280-dim from ESM)
# - target: cell state (scVI latent representation, 10-dim)
flow_model = ConditionalFlatCouplingFlow(
    conditioning_dim=embeddings[0].shape[0],   # dimension of variant embedding
    embedding_dim=10,                          # dimension of cell latent space (scVI)
    conditioning_depth=2,                      # depth of conditioner network
    n_flows=20,                                # number of flow blocks
    in_channels=10,                            # input channel size (latent dimension)
    hidden_dim=1024,                           # hidden layer dimension
    hidden_depth=2,                            # number of hidden layers
    activation="none",                         # activation function
    conditioner_use_bn=True                    # whether to use batch normalization
)


# Cell-level perturbation labels
cond_stage_data = np.array(adata_train.obs["condition"])

# Net2NetFlow wrapper:
#  - configured_flow : flow model defined above
#  - cond_stage_data : perturbation labels
#  - perturbToEmbed  : perturbation → embedding index mapping
#  - embedData       : PLM-derived variant embeddings
#  - scvi_model      : pretrained scVI model for cell representation
model_c = Net2NetFlow_scVIFixFlow(
    configured_flow=flow_model,
    cond_stage_data=cond_stage_data,
    perturbToEmbedLib=perturbToEmbed,
    embedData=embeddings,
    scvi_model=scvi_model,
)

model_c.adata = adata_train


# Define Conditional INN (Flow) model and create Net2NetFlow wrapper
# Conditional flow model:
# - condition: variant embedding (e.g., 1280-dim from ESM)
# - target: cell state (scVI latent representation, 10-dim)
flow_model = ConditionalFlatCouplingFlow(
    conditioning_dim=embeddings[0].shape[0],   # dimension of variant embedding
    embedding_dim=10,                          # dimension of cell latent space (scVI)
    conditioning_depth=2,                      # depth of conditioner network
    n_flows=20,                                # number of flow blocks
    in_channels=10,                            # input channel size (latent dimension)
    hidden_dim=1024,                           # hidden layer dimension
    hidden_depth=2,                            # number of hidden layers
    activation="none",                         # activation function
    conditioner_use_bn=True                    # whether to use batch normalization
)

# Cell-level perturbation labels
cond_stage_data = np.array(adata_train.obs["condition"])

# Net2NetFlow wrapper:
#  - configured_flow : flow model defined above
#  - cond_stage_data : perturbation labels
#  - perturbToEmbed  : perturbation → embedding index mapping
#  - embedData       : PLM-derived variant embeddings
#  - scvi_model      : pretrained scVI model for cell representation
model_c = Net2NetFlow_scVIFixFlow(
    configured_flow=flow_model,
    cond_stage_data=cond_stage_data,
    perturbToEmbedLib=perturbToEmbed,
    embedData=embeddings,
    scvi_model=scvi_model,
)

model_c.adata = adata_train



# seed_all(42)
timestamp = get_timestamp()
project_name = make_project_name(config, timestamp)
checkpoint_dir = os.path.join('/NFS_DATA/samsung/variantPerturbNet/cinn', project_name)

# Train CINN (Conditional Flow) model
# Select device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to save CINN model
path_cinn_model_save = checkpoint_dir

# Move model to device
model_c.to(device=device)

# Train flow model
model_c.train(
    n_epochs=25,     # number of training epochs
    batch_size=128,  # mini-batch size
    lr=4.5e-6,       # learning rate
    train_ratio=0.8, # train/validation split ratio
)

# Save trained model
model_c.save(path_cinn_model_save)

# Save configuration
import pickle
with open(f"{checkpoint_dir}/config.pkl", "wb") as f:
    pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"{checkpoint_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

