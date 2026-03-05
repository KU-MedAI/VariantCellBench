from sklearn.metrics import r2_score, mean_absolute_error

def compute_metrics(p):
    preds = p.predictions
    labels = p.label_ids
    
    r2 = r2_score(labels.flatten(), preds.flatten())
    mae = mean_absolute_error(labels.flatten(), preds.flatten())
    mse = ((preds - labels) ** 2).mean()

    return {"mse": float(mse), "mae": float(mae), "r2_score": float(r2)}