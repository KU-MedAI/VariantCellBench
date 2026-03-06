# Benchmarking Virtual Cell Models for Predicting Variant-Induced Single-Cell Transcriptomic Responses

## Abstract
<div align="center">
<img width="1049" height="752" alt="image" src="https://github.com/user-attachments/assets/07615da7-1b2e-4229-8b6d-abf58e381d54" />
  
</div>
<br>
<br>

Background: Missense variants constitute a major class of disease-associated genetic variations. By altering amino acid sequences, these variants can disrupt protein structure, perturb protein–protein interactions, and ultimately influence gene expression changes that drive disease phenotypes. Despite their clinical relevance, the functional implications of many missense variants are unknown. Characterizing their cellular effects at scale is therefore critical; however, experimental approaches are expensive and have low throughput. Therefore, computational approaches are becoming increasingly important to infer the cellular impact of genetic variants. Although many virtual cell models have been developed to predict gene expression responses to genetic perturbations such as gene knockouts, their ability to predict transcriptional changes induced by missense variants has yet to be systematically benchmarked.

Results: We benchmarked virtual cell models to predict missense variant-induced transcriptomic responses. Eight virtual cell models were evaluated, including supervised perturbation and single-cell foundation models. Variant information was encoded using five protein language models paired with two variant representation strategies. All configurations were assessed using a Perturb-seq dataset of TP53 missense variants. Across eight evaluation metrics, supervised perturbation models achieved higher predictive accuracy for transcriptomic responses to TP53 variants than single-cell foundation models, even without large-scale single-cell pretraining. Downstream analyses further investigated the relationship between predicted transcriptomic changes and known functional consequences of TP53 variants.

Conclusions: This study provides the first systematic benchmark of virtual cell models for predicting the transcriptomic effects of missense variants. The results establisg a reference for modeling cellular responses to genetic variants and improve computational interpretation of missense variants.

## Introduction
This is the official code repository for **VariantCellBench**. In this benchmark, we evaluate the ability of various virtual cell models to predict transcriptomic changes induced by missense variants. 

We have integrated and evaluated **8 virtual cell models**, which are categorized into supervised perturbation models and single-cell foundation models:

| Model | Category | Paper link | Github |
|-------|----------|------------|--------|
| **GEARS** | Supervised Perturbation | [Nature Biotech](https://www.nature.com/articles/s41587-023-01770-9) | [Jurelab/gears](https://github.com/snap-stanford/GEARS) |
| **PerturbNet** | Supervised Perturbation | [Nature Machine Intelligence](https://www.nature.com/articles/s42256-023-00746-9) | [PerturbNet](https://github.com/YuanLab-SJTU/PerturbNet) |
| **CellFlow** | Supervised Perturbation | - | - |
| **scLAMBDA** | Supervised Perturbation | - | - |
| **scGPT** | Single-cell Foundation | [Nature Methods](https://www.nature.com/articles/s41592-024-02201-0) | [scGPT](https://github.com/bowang-lab/scGPT) |
| **scFoundation** | Single-cell Foundation | [Nature Methods](https://www.nature.com/articles/s41592-024-02305-7) | [scFoundation](https://github.com/biomap-research/scFoundation) |
| **GeneCompass** | Single-cell Foundation | [Cell Research](https://www.nature.com/articles/s41422-024-01034-y) | [GeneCompass](https://github.com/xcompass/GeneCompass) |
| **CellFM** | Single-cell Foundation | - | - |

---

## Dependencies & Environment Setup

We provide YAML and requirements files to reproduce the environment.

### Conda Environment
You can set up the environment using the provided configurations. For instance, to set up the evaluation environment:

```bash
conda env create -f evalutation_metrics/variant-cell-eval_nz/cell-eval-dev.yml
conda activate cell-eval-dev
#(Note: Each model under the models/ directory also contains its specific requirements.txt or setup instructions which you may need to install depending on the model you want to run.)
```

1. Protein Language Models (PLMs) for Variant Encoding
We use PLMs (e.g., ProtT5) to encode missense variants into meaningful embeddings before passing them to the cell models.
```bash
cd PLMs
python ProtT5.py --input ../data/variants.fasta --output ../data/embeddings/
```
See `PLMs/README.md` for more details on different representation strategies.

2. Running Virtual Cell Models
Each of the 8 evaluated models is located in the `models/` directory with a standardized structure for training and inference.

**Training a Model**
To train a model (e.g., GEARS) with the variant embeddings:
```bash
cd models/GEARS/run
python train.py --config ../src/config.py --data_dir ../../../data/
```

**Inference (Predicting Transcriptomic Responses)**
To run inference on unseen variants:

```bash
cd models/GEARS/run
python train.py --config ../src/config.py --data_dir ../../../data/
```
(Repeat the above structure for `scGPT`, `scFoundation`, `GeneCompass`, etc., navigating to their respective `run/` folders).

3. Evaluation Metrics
We comprehensively evaluate the predicted gene expression changes against the ground-truth single-cell transcriptomic responses across 8 metrics. The evaluation codes are located in `evalutation_metrics/`.
```bash
cd evalutation_metrics/variant-cell-eval_nz/src/cell_eval/

# Example: Run AUPRC analysis
python AUPRC/main.py --pred_path <predictions.csv> --true_path <ground_truth.csv>

# Example: Run Gene-wise Pearson Correlation Coefficient (PCC)
python gene_wise_pcc/condition.py --input <data>
```

Evaluation scripts support analyses like AUPRC, Gene-wise PCC, and model-specific metrics (e.g., GEARS and Systema evaluation suites). Finally, you can summarize results using:
```bash
python evalutation_metrics/variant-cell-eval_nz/run/result_summary.py
```

## Acknowledgments
This repository integrates implementations and builds upon several open-source projects. We thank the authors of GEARS, scGPT, scFoundation, GeneCompass, PerturbNet, CellFlow, scLAMBDA, CellFM, Protein Language models, Evaluation Metrics and scvi-tools for their foundational work and codebases.