import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import GNNModel  # Make sure this matches the model in evaluate.py
from train import run_training
from evaluate import (
    load_model,
    evaluate,
    evaluate_with_uncertainty,
    plot_predictions_with_uncertainty
)
from data_loader import dataset, train_loader as loader_tr, test_loader as loader_te, label_scaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from model_evaluation import plot_cv_predictions_color_std
from confidence import _compute_confidences
from pathlib import Path
import random
from sklearn.model_selection import KFold
from scaler_helper import fit_and_apply_scalers, get_dataset
from torch_geometric.loader import DataLoader 

def main():
  kf = KFold(n_splits=5, shuffle=True, random_state=0)
  
  all_preds, all_true, all_stds = [], [], []
  dataset = get_dataset()
  for fold, (train_idx, test_idx) in enumerate(kf.split(dataset), start=1):
      print(f"\n=== Fold {fold} ===")
  
      # 80% trainval, 20% test
      trainval_set = [dataset[i] for i in train_idx]
      test_set     = [dataset[i] for i in test_idx]
  
      # Optional: carve validation from trainval (e.g., first 10%)
      val_size = max(1, int(0.1 * len(trainval_set)))
      dataset_val = trainval_set[:val_size]
      dataset_tr  = trainval_set[val_size:]
  
      # --- Fit on TRAIN ONLY, then transform TRAIN/VAL/TEST (per fold) ---
      dataset_tr_s, dataset_val_s, dataset_te_s, label_scaler_fold = fit_and_apply_scalers(
          dataset_tr, dataset_val, test_set
      )
  
      # Loaders (same names you already use)
      loader_tr = DataLoader(dataset_tr_s, batch_size=config.get("batch_size", 100), shuffle=True)
      loader_va = DataLoader(dataset_val_s, batch_size=config.get("batch_size", 100))
      loader_te = DataLoader(dataset_te_s, batch_size=config.get("batch_size", 100))
  
      # === Train model (unchanged) ===
      model = GNNModel(
          orig_node_fea_len=dataset[0].num_node_features,
          edge_fea_len=config["edge_fea_len"],
          node_fea_len=config["node_fea_len"],
          n_conv=config["n_conv"],
          h_fea_len=config["h_fea_len"],
          n_h=config["n_h"]
      ).to(device)
  
      optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
      scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4, verbose=True)
  
      # If your training expects test loader as "validation", pass loader_va here
      run_training(model, loader_tr, loader_va, optimizer, scheduler, device, config)
  
      # === Evaluate ===
      test_preds, test_true = evaluate(loader_te, model, label_scaler_fold)
      r2 = r2_score(test_true, test_preds)
      mse = mean_squared_error(test_true, test_preds)
      mae = mean_absolute_error(test_true, test_preds)

      print(f"Fold {fold} — R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
      #print(f"Fold {fold} R²: {r2_score(test_true, test_preds):.4f}")
  
      # === Uncertainty (optional) ===
      num_samples = config.get("mc_dropout_samples", 50)
      _, _, test_epi, test_alea = evaluate_with_uncertainty(
          loader_te, model, num_samples, device, label_scaler_fold
      )
      std = np.sqrt(test_epi**2 + test_alea**2)
  
      all_preds.append(test_preds)
      all_true.append(test_true)
      all_stds.append(std)
  
  # === Concatenate & plot CV results ===
  all_preds = np.concatenate(all_preds)
  all_true  = np.concatenate(all_true)
  all_stds  = np.concatenate(all_stds)
  
  plot_cv_predictions_color_std(
      all_preds, all_true, all_stds,
      "Residual plot of GNN model ($\\mu_0 H_c$)",
      "5-fold CV",
      "cv_predictions",
      Path(".")
  )
  
  
def run_10_trials_confidence(config, dataset, loader_tr, loader_te, label_scaler, device, n_trials):
  
    y_test_trials = []
    y_pred_trials = []
    std_total_trials = []
    std_alea_trials = []
    std_epi_trials = []

    for t in range(n_trials):
        # Different seed per trial
        seed = 0 + t
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Fresh model, optimizer, scheduler each trial
        model = GNNModel(
            orig_node_fea_len=dataset[0].num_node_features,
            edge_fea_len=config["edge_fea_len"],
            node_fea_len=config["node_fea_len"],
            n_conv=config["n_conv"],
            h_fea_len=config["h_fea_len"],
            n_h=config["n_h"]
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4, verbose=True)

        # Train this trial
        run_training(model, loader_tr, loader_te, optimizer, scheduler, device, config)

        # Evaluate with uncertainty on the SAME test loader for consistent N
        num_samples = config.get("mc_dropout_samples", 50)
        test_preds_u, test_true_u, test_epi, test_alea = evaluate_with_uncertainty(
            loader_te, model, num_samples, device, label_scaler
        )

        train_preds, train_true, train_epi, train_alea = evaluate_with_uncertainty(
            loader_tr, model, num_samples, device, label_scaler
        )

        # Collect (flatten to 1D per trial)
        y_test_trials.append(np.asarray(test_true_u).flatten())
        y_pred_trials.append(np.asarray(test_preds_u).flatten())
        std_epi_trials.append(np.asarray(test_epi).flatten())
        std_alea_trials.append(np.asarray(test_alea).flatten())
        std_total_trials.append(np.sqrt(np.asarray(test_epi)**2 + np.asarray(test_alea)**2).flatten())

    # Stack to (n_trials, n_samples)
    y_test_arr = np.vstack(y_test_trials)
    y_pred_arr = np.vstack(y_pred_trials)
    std_total_arr = np.vstack(std_total_trials)
    std_al_arr = np.vstack(std_alea_trials)
    std_ep_arr = np.vstack(std_epi_trials)

    # Make sure output dir exists
    out_dir = Path("confidence_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    metric = "mae"  # or "mse"
    plot_name = f"confidence_curve_{metric}_all"
    title = "Confidence curve of GCN model ($\mu_0 H_c$)"  
    details = f"{metric.upper()} - {n_trials} trials on {y_test_arr.shape[1]} samples"

    _compute_confidences(
        metric=metric,
        output_dir=out_dir,
        plot_name=plot_name,
        title=title,
        details=details,
        y_test_arr=y_test_arr,
        y_pred_arr=y_pred_arr,
        std_total_arr=std_total_arr,
        std_al_arr=std_al_arr,
        std_ep_arr=std_ep_arr,
        plot_std_total_only=False,   # set True to plot total-only
        epistemic_model=False,
    )

    # --- After you already computed train/test preds/true and uncertainties ---
    # Total uncertainty per sample
    train_total_unc = np.sqrt(train_epi**2 + train_alea**2)
    test_total_unc  = np.sqrt(test_epi**2  + test_alea**2)

    # Choose metric: 'mae' (default) or 'mse'
    metric_choice = 'mae'

    # Train: compute curves & AUCO
    """
    cov_tr, rc_tr, ro_tr, auco_tr_raw, auco_tr_norm, mname_tr = compute_confidence_oracle_curves(
        train_preds, train_true, train_total_unc, metric=metric_choice
    )
    plot_confidence_oracle_curve(
        cov_tr, rc_tr, ro_tr, auco_tr_raw, auco_tr_norm,
        label='Train', metric_name=mname_tr, save_path='train_confidence_oracle.svg'
    )
    print(f"[Train] AUCO (raw): {auco_tr_raw:.6e} | AUCO (norm): {auco_tr_norm:.4f} | metric={mname_tr}")

    # Test: compute curves & AUCO
    cov_te, rc_te, ro_te, auco_te_raw, auco_te_norm, mname_te = compute_confidence_oracle_curves(
        test_preds, test_true, test_total_unc, metric=metric_choice
    )
    plot_confidence_oracle_curve(
        cov_te, rc_te, ro_te, auco_te_raw, auco_te_norm,
        label='Test', metric_name=mname_te, save_path='test_confidence_oracle.svg'
    )
    print(f"[Test] AUCO (raw): {auco_te_raw:.6e} | AUCO (norm): {auco_te_norm:.4f} | metric={mname_te}")
    """
# Set seed and device
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
run_10_trials_confidence(config, dataset, loader_tr, loader_te, label_scaler, device, n_trials=10)

if __name__ == "__main__":
    main()
