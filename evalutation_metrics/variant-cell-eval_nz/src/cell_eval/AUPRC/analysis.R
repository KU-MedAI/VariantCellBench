# analysis.R

# 명령줄 인수 받기 (1: 데이터 경로, 2: 출력 경로)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript analysis.R <data_dir> <output_dir>")
}
data_dir <- args[1]
output_dir <- args[2]

# 라이브러리 로드 (메시지 숨김)
suppressPackageStartupMessages({
  library(limma)
  library(dplyr)
  library(tidyr)
  library(edgeR)
  library(yardstick)
  library(ggplot2)
})

# === 1. 데이터 읽기 ===
# Truth Data
truth_obs <- read.csv(file.path(data_dir, "truth_obs.csv"), row.names=1)
truth_var <- read.csv(file.path(data_dir, "truth_var.csv"), row.names=1)
truth_X <- read.csv(file.path(data_dir, "truth_X.csv"), header=FALSE)
# var에 있던 유전자 이름을 컬럼으로 설정 (Python에서 gene_name 컬럼이 있는지 확인 필요, 없으면 index 사용)
if("gene_name" %in% colnames(truth_var)){
    names(truth_X) <- truth_var$gene_name
} else {
    names(truth_X) <- rownames(truth_var)
}
genes <- names(truth_X)
# Pred Data
pred_X_full <- read.csv(file.path(data_dir, "pred_X.csv"), header=FALSE)
names(pred_X_full) <- genes

# === 2. 데이터 분할 (Condition 별) ===
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

# === 3. Ground Truth DEG 분석 (Limma) ===
ctrl_name <- "ctrl" 
mutants <- all_conditions[all_conditions != ctrl_name]

DEG_genes_obs <- list()

for (mut in mutants) {
  # Data filtering
  idx <- truth_obs$condition %in% c(ctrl_name, mut)
  temp_X <- truth_X[idx, ]
  temp_obs <- truth_obs[idx, ]
  
  counts <- t(temp_X)
  
  # Design Matrix
  cond_factor <- factor(temp_obs$condition, levels = c(ctrl_name, mut))
  design <- model.matrix(~cond_factor)
  
  # Limma
  fit <- lmFit(counts, design)
  fit <- eBayes(fit)
  top.table <- topTable(fit, coef=2, sort.by = "logFC", n = Inf)
  
  # DEG Filtering
  p_lvl <- 0.05
  lfc <- 0.58
  DEG_genes_obs[[mut]] <- rownames(top.table)[top.table$adj.P.Val < p_lvl & abs(top.table$logFC) > lfc]
}

# === 4. Prediction DEG 분석 & Scoring ===
score_scgpt <- list()
score_baseline <- list()
DEG_genes_scgpt <- list()

for (mut in mutants){
  # (1) Prediction DEG Analysis (Obs Ctrl vs Pred Mut)
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
  
  # Filtering Pred DEGs
  p_lvl <- 0.05
  lfc <- 0.58
  DEG_genes_scgpt[[mut]] <- rownames(toptable)[toptable$adj.P.Val < p_lvl & abs(toptable$logFC) > lfc]
  
  # (2) Scoring for AUPRC
  score_scgpt[[mut]] <- rep(0, length(genes))
  names(score_scgpt[[mut]]) <- genes
  score_baseline[[mut]] <- rep(0, length(genes))
  
  common_genes <- intersect(rownames(toptable), genes)
  score_scgpt[[mut]][common_genes] <- abs(toptable[common_genes, "logFC"])
}

# === 5. Plotting (PR Curve) & Overlap Stats ===
overlap_stats <- data.frame()
dir.create(output_dir, showWarnings = FALSE)

for (mut in mutants){
  # -- AUPRC Plot --
  true_vec <- rep(0, length(genes))
  if (!is.null(DEG_genes_obs[[mut]]) && length(DEG_genes_obs[[mut]]) > 0) {
    true_vec[which(genes %in% DEG_genes_obs[[mut]])] <- 1
  }
  
  # Skip if no True DEGs
  if (sum(true_vec) == 0) next
  
  truth_factor <- factor(true_vec, levels = c("0", "1"))
  
  # dtf <- rbind(
  #   data.frame(true = truth_factor, score = as.numeric(score_scgpt[[mut]]), method = "Prediction"),
  #   data.frame(true = truth_factor, score = as.numeric(score_baseline[[mut]]), method = "Baseline")
  # )
  dtf <- data.frame(
    true = truth_factor, 
    score = as.numeric(score_scgpt[[mut]]), 
    method = "Prediction"
  )
  # [수정됨] AUPRC 값 계산 (Prediction 모델만)
  auprc_val <- dtf %>%
    filter(method == "Prediction") %>%
    pr_auc(true, score, event_level = "second") %>%
    pull(.estimate)
  
  # Save Plot
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
  
  # -- Overlap Stats --
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
    AUPRC = round(auprc_val, 4)  # AUPRC 값 추가
  ))
}

# === [추가됨] 전체 평균(Mean) 계산 및 행 추가 ===
if (nrow(overlap_stats) > 0) {
  # 숫자형 컬럼들의 평균 계산 (Recall 포함)
  mean_row <- data.frame(
    Condition = "mean",
    n_True = round(mean(overlap_stats$n_True, na.rm = TRUE), 1),
    n_Pred = round(mean(overlap_stats$n_Pred, na.rm = TRUE), 1),
    n_Overlap = round(mean(overlap_stats$n_Overlap, na.rm = TRUE), 1),
    Recall = round(mean(overlap_stats$Recall, na.rm = TRUE), 3),       # Recall 평균
    Precision = round(mean(overlap_stats$Precision, na.rm = TRUE), 3), # Precision 평균
    F1_Score = round(mean(overlap_stats$F1_Score, na.rm = TRUE), 3),    # F1 Score 평균
    AUPRC = round(mean(overlap_stats$AUPRC, na.rm = TRUE), 4)  # AUPRC 평균 추가
  )
  
  # 기존 데이터프레임 아래에 평균 행 추가
  overlap_stats <- rbind(overlap_stats, mean_row)
}

# === 6. Save Stats CSV ===
write.csv(overlap_stats, file.path(output_dir, "overlap_stats.csv"), row.names = FALSE)
print(paste("Analysis Complete. Results saved to:", output_dir))