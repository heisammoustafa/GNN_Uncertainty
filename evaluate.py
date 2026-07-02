import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from models import GNNModel

# Load model
def load_model(path, input_dim, edge_dim):
    model = GNNModel(orig_node_fea_len=input_dim, edge_fea_len=edge_dim)
    map_loc = torch.device('cpu') if not torch.cuda.is_available() else None
    model.load_state_dict(torch.load(path, map_location=map_loc))
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    return model

# Basic evaluation
def evaluate(loader, model, scaler=None):
    predictions, true_values = [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            mu, _ = model(data)
            pred = mu.cpu().numpy()
            true = data.y.view(-1, 1).cpu().numpy()
            if scaler:
                pred = scaler.inverse_transform(pred)
                true = scaler.inverse_transform(true)
            predictions.append(pred)
            true_values.append(true)
    return np.concatenate(predictions), np.concatenate(true_values)

# Evaluation with uncertainty
def evaluate_with_uncertainty(loader, model, num_samples, device, scaler=None):
    model.eval()
    # Determine the device the model is currently on
    device = next(model.parameters()).device
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    predictions, true_values, epistemic_uncertainty, aleatoric_uncertainty = [], [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds, log_vars = [], []

            for _ in range(num_samples):
                mu, log_var = model(data)
                preds.append(mu.unsqueeze(0))
                log_vars.append(log_var.unsqueeze(0))

            preds = torch.cat(preds, dim=0)
            log_vars = torch.cat(log_vars, dim=0)

            mu_pred = preds.mean(dim=0)
            epistemic_var = preds.var(dim=0)
            aleatoric_var = torch.exp(log_vars.mean(dim=0))

            predictions.append(mu_pred.cpu().numpy())
            true_values.append(data.y.view(-1, 1).cpu().numpy())
            epistemic_uncertainty.append(torch.sqrt(epistemic_var).cpu().numpy())
            aleatoric_uncertainty.append(torch.sqrt(aleatoric_var).cpu().numpy())

    predictions = np.concatenate(predictions)
    true_values = np.concatenate(true_values)
    epistemic_uncertainty = np.concatenate(epistemic_uncertainty)
    aleatoric_uncertainty = np.concatenate(aleatoric_uncertainty)

    if scaler:
        predictions = scaler.inverse_transform(predictions)
        true_values = scaler.inverse_transform(true_values)

    return predictions, true_values, epistemic_uncertainty, aleatoric_uncertainty

# Plotting
def plot_predictions_with_uncertainty(predictions, true_values, epistemic_uncertainty, aleatoric_uncertainty, label='Test', save_path='plot.svg'):
    sorted_idx = np.argsort(true_values.flatten())
    pred_sorted = predictions.flatten()[sorted_idx]
    true_sorted = true_values.flatten()[sorted_idx]
    epistemic_sorted = epistemic_uncertainty.flatten()[sorted_idx]
    aleatoric_sorted = aleatoric_uncertainty.flatten()[sorted_idx]
    total_uncertainty = np.sqrt(epistemic_sorted**2 + aleatoric_sorted**2)

    plt.figure(figsize=(10, 6))
    plt.scatter(pred_sorted, true_sorted, color='blue', alpha=0.5, label=f'{label} Predictions')
    plt.errorbar(pred_sorted, true_sorted, xerr=total_uncertainty, fmt='o', color='blue', alpha=0.5, capsize=3)
    plt.plot([true_sorted.min(), true_sorted.max()], [true_sorted.min(), true_sorted.max()], 'r--')
    plt.xlabel("Predicted Coercivity with 1 Std Dev", fontsize=15)
    plt.ylabel("Computed Coercivity in Tesla", fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()

# Main
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('weights_uncertainty.pt', dataset[0].num_node_features, edge_fea_len=1)

    # Evaluate
    train_preds, train_true = evaluate(loader_tr, model, label_scaler)
    test_preds, test_true = evaluate(loader_te, model, label_scaler)
    print(f"R^2 Score (Test): {r2_score(test_true, test_preds):.4f}")

    # Evaluate with uncertainty
    num_samples = 50
    train_preds, train_true, train_epi, train_alea = evaluate_with_uncertainty(loader_tr, model, num_samples, device, label_scaler)
    test_preds, test_true, test_epi, test_alea = evaluate_with_uncertainty(loader_te, model, num_samples, device, label_scaler)

    # Plot
    plot_predictions_with_uncertainty(train_preds, train_true, train_epi, train_alea, label='Train', save_path='train_uncertainty.svg')
    plot_predictions_with_uncertainty(test_preds, test_true, test_epi, test_alea, label='Test', save_path='test_uncertainty.svg')
