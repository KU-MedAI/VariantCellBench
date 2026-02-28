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
