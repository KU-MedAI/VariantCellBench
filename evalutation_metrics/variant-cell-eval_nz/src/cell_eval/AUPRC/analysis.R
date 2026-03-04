
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript analysis.R <data_dir> <output_dir>")
}
data_dir <- args[1]
output_dir <- args[2]

suppressPackageStartupMessages({
  library(limma)
  library(dplyr)
  library(tidyr)
  library(edgeR)
  library(yardstick)
  library(ggplot2)
})

truth_obs <- read.csv(file.path(data_dir, "truth_obs.csv"), row.names=1)
truth_var <- read.csv(file.path(data_dir, "truth_var.csv"), row.names=1)
truth_X <- read.csv(file.path(data_dir, "truth_X.csv"), header=FALSE)
if("gene_name" %in% colnames(truth_var)){
    names(truth_X) <- truth_var$gene_name
} else {
    names(truth_X) <- rownames(truth_var)
}
genes <- names(truth_X)
pred_X_full <- read.csv(file.path(data_dir, "pred_X.csv"), header=FALSE)
names(pred_X_full) <- genes

all_conditions <- unique(truth_obs$condition)
X_by_cond <- list()
obs_by_cond <- list()
pred_by_cond <- list()

for (cond in all_conditions) {
  idx <- truth_obs$condition == cond
  X_by_cond[[cond]] <- truth_X[idx, ]
  obs_by_cond[[cond]] <- truth_obs[idx, ]
  pred_by_cond[[cond]] <- pred_X_full[idx, ]
}

ctrl_name <- "ctrl" 
mutants <- all_conditions[all_conditions != ctrl_name]

DEG_genes_obs <- list()

for (mut in mutants) {

  idx <- truth_obs$condition %in% c(ctrl_name, mut)
  temp_X <- truth_X[idx, ]
  temp_obs <- truth_obs[idx, ]
  
  counts <- t(temp_X)

  cond_factor <- factor(temp_obs$condition, levels = c(ctrl_name, mut))
  design <- model.matrix(~cond_factor)
  

  fit <- lmFit(counts, design)
  fit <- eBayes(fit)
  top.table <- topTable(fit, coef=2, sort.by = "logFC", n = Inf)
  

  p_lvl <- 0.05
  lfc <- 0.58
  DEG_genes_obs[[mut]] <- rownames(top.table)[top.table$adj.P.Val < p_lvl & abs(top.table$logFC) > lfc]
}

score_scgpt <- list()
score_baseline <- list()
DEG_genes_scgpt <- list()

for (mut in mutants){

  obs_ctrl_data <- X_by_cond[[ctrl_name]]
  pred_mut_data <- pred_by_cond[[mut]]
  
  temp <- rbind(obs_ctrl_data, pred_mut_data)
  counts <- t(temp)
  
  condition <- c(rep("control", nrow(obs_ctrl_data)), rep("prediction", nrow(pred_mut_data)))
  cond_factor <- factor(condition, levels = c("control", "prediction"))
  design <- model.matrix(~cond_factor)
  
  fit <- lmFit(counts, design)
  fit <- eBayes(fit)
  toptable <- topTable(fit, coef=2, sort.by = "logFC", n = Inf)
  
  p_lvl <- 0.05
  lfc <- 0.58
  DEG_genes_scgpt[[mut]] <- rownames(toptable)[toptable$adj.P.Val < p_lvl & abs(toptable$logFC) > lfc]
  
  score_scgpt[[mut]] <- rep(0, length(genes))
  names(score_scgpt[[mut]]) <- genes
  score_baseline[[mut]] <- rep(0, length(genes))
  
  common_genes <- intersect(rownames(toptable), genes)
  score_scgpt[[mut]][common_genes] <- abs(toptable[common_genes, "logFC"])
}

overlap_stats <- data.frame()
dir.create(output_dir, showWarnings = FALSE)

for (mut in mutants){

  true_vec <- rep(0, length(genes))
  if (!is.null(DEG_genes_obs[[mut]]) && length(DEG_genes_obs[[mut]]) > 0) {
    true_vec[which(genes %in% DEG_genes_obs[[mut]])] <- 1
  }
  

  if (sum(true_vec) == 0) next
  
  truth_factor <- factor(true_vec, levels = c("0", "1"))
  

  dtf <- data.frame(
    true = truth_factor, 
    score = as.numeric(score_scgpt[[mut]]), 
    method = "Prediction"
  )

  auprc_val <- dtf %>%
    filter(method == "Prediction") %>%
    pr_auc(true, score, event_level = "second") %>%
    pull(.estimate)

  plot_filename <- file.path(output_dir, paste0("AUPRC_", gsub("~|\\+", "_", mut), ".png"))
  
  p <- dtf %>%
    group_by(method) %>%
    pr_curve(true, score, event_level = "second") %>%
    ggplot(aes(x = recall, y = precision, color=method)) +
    geom_path(linewidth = 1) +
    coord_equal() +
    theme_bw() +
    ggtitle(paste("PR Curve:", mut))
  
  ggsave(filename = plot_filename, plot = p, width = 6, height = 5)
  
  true_genes <- if(!is.null(DEG_genes_obs[[mut]])) DEG_genes_obs[[mut]] else character(0)
  pred_genes <- if(!is.null(DEG_genes_scgpt[[mut]])) DEG_genes_scgpt[[mut]] else character(0)
  
  common <- intersect(true_genes, pred_genes)
  n_true <- length(true_genes)
  n_pred <- length(pred_genes)
  n_common <- length(common)
  
  recall <- if (n_true > 0) n_common / n_true else 0
  precision <- if (n_pred > 0) n_common / n_pred else 0
  f1 <- if ((recall + precision) > 0) 2 * (recall * precision) / (recall + precision) else 0
  
  overlap_stats <- rbind(overlap_stats, data.frame(
    Condition = mut,
    n_True = n_true,
    n_Pred = n_pred,
    n_Overlap = n_common,
    Recall = round(recall, 3),
    Precision = round(precision, 3),
    F1_Score = round(f1, 3),
    AUPRC = round(auprc_val, 4)
  ))
}

if (nrow(overlap_stats) > 0) {

  mean_row <- data.frame(
    Condition = "mean",
    n_True = round(mean(overlap_stats$n_True, na.rm = TRUE), 1),
    n_Pred = round(mean(overlap_stats$n_Pred, na.rm = TRUE), 1),
    n_Overlap = round(mean(overlap_stats$n_Overlap, na.rm = TRUE), 1),
    Recall = round(mean(overlap_stats$Recall, na.rm = TRUE), 3),       
    Precision = round(mean(overlap_stats$Precision, na.rm = TRUE), 3), 
    F1_Score = round(mean(overlap_stats$F1_Score, na.rm = TRUE), 3),    
    AUPRC = round(mean(overlap_stats$AUPRC, na.rm = TRUE), 4)  
  )
  
  overlap_stats <- rbind(overlap_stats, mean_row)
}

write.csv(overlap_stats, file.path(output_dir, "overlap_stats.csv"), row.names = FALSE)
print(paste("Analysis Complete. Results saved to:", output_dir))