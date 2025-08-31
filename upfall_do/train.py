
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, log_loss)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import DEVICE, LR, EPOCHS, ROUNDS, BATCH_SIZE
from .data import SequenceDataset
from .models import TemporalCNN, BiLSTM, CNN_LSTM, TransformerEncoderClassifier

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def train_one(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = EPOCHS):
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    hist = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    for ep in range(epochs):
        model.train(); tloss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward(); opt.step()
            tloss += loss.item()
        model.eval(); vloss, corr, tot = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = crit(out, yb)
                vloss += loss.item()
                preds = out.argmax(dim=1)
                corr += (preds == yb).sum().item()
                tot += len(yb)
        hist['train_loss'].append(tloss / max(1, len(train_loader)))
        hist['val_loss'].append(vloss / max(1, len(val_loader)))
        hist['val_acc'].append(corr / max(1, tot))
    return hist

def predict_proba(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            prob = torch.softmax(out, dim=1)[:,1]
            y_prob.append(prob.detach().cpu().numpy())
            y_true.append(yb.numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)

def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    pp = np.vstack([1 - y_prob, y_prob]).T
    try:
        ll = log_loss(y_true, pp, labels=[0,1])
    except ValueError:
        ll = float('nan')
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'specificity': spec, 'roc_auc': auc, 'loss': ll}

def train_eval_deep_for_modality(X, y, model_name: str):
    from sklearn.model_selection import train_test_split
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
    ds_tr = SequenceDataset(X_train, y_train)
    ds_va = SequenceDataset(X_val, y_val, scaler=ds_tr.scaler)
    ds_te = SequenceDataset(X_test, y_test, scaler=ds_tr.scaler)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False)
    in_feats = X.shape[-1]
    if model_name == 'cnn': model = TemporalCNN(in_feats)
    elif model_name == 'lstm': model = BiLSTM(in_feats)
    elif model_name == 'cnn_lstm': model = CNN_LSTM(in_feats)
    elif model_name == 'transformer': model = TransformerEncoderClassifier(in_feats)
    else: raise ValueError("unknown model")
    round_metrics = {k: [] for k in ['accuracy','precision','recall','f1','specificity','roc_auc','loss']}
    for r in range(ROUNDS):
        set_seed(100 + r)
        _ = train_one(model, dl_tr, dl_va, epochs=EPOCHS)
        y_true, y_prob = predict_proba(model, dl_te)
        m = compute_metrics(y_true, y_prob)
        for k in round_metrics.keys(): round_metrics[k].append(m[k])
    return round_metrics


# from typing import Dict, List, Tuple
# import numpy as np
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
#                              f1_score, roc_auc_score, confusion_matrix, log_loss)
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

# from .config import DEVICE, LR, EPOCHS, ROUNDS, BATCH_SIZE
# from .data import SequenceDataset
# from .models import TemporalCNN, BiLSTM, CNN_LSTM, TransformerEncoderClassifier

# def set_seed(seed: int):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

# def train_one(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = EPOCHS):
#     model.to(DEVICE)

#     # ---- class weights for imbalance ----
#     y_train = train_loader.dataset.y
#     class_counts = np.bincount(y_train)
#     weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
#     weights = weights / weights.sum()
#     crit = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

#     # optimizer
#     opt = optim.Adam(model.parameters(), lr=LR)

#     hist = {'train_loss': [], 'val_loss': [], 'val_acc': []}

#     # ---- early stopping ----
#     best_val_loss = float("inf")
#     patience, patience_counter = 5, 0
#     best_state = None

#     for ep in range(epochs):
#         model.train()
#         tloss = 0.0
#         for xb, yb in train_loader:
#             xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#             opt.zero_grad()
#             out = model(xb)
#             loss = crit(out, yb)
#             loss.backward()
#             opt.step()
#             tloss += loss.item()
        
#         # validation
#         model.eval()
#         vloss, corr, tot = 0.0, 0, 0
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#                 out = model(xb)
#                 loss = crit(out, yb)
#                 vloss += loss.item()
#                 preds = out.argmax(dim=1)
#                 corr += (preds == yb).sum().item()
#                 tot += len(yb)
        
#         train_loss = tloss / max(1, len(train_loader))
#         val_loss = vloss / max(1, len(val_loader))
#         val_acc = corr / max(1, tot)

#         hist['train_loss'].append(train_loss)
#         hist['val_loss'].append(val_loss)
#         hist['val_acc'].append(val_acc)

#         # ---- early stopping check ----
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             best_state = model.state_dict()
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print(f"[EARLY STOPPING] at epoch {ep+1}")
#                 if best_state is not None:
#                     model.load_state_dict(best_state)
#                 break

#     return hist

# def predict_proba(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
#     model.eval()
#     y_true, y_prob = [], []
#     with torch.no_grad():
#         for xb, yb in loader:
#             xb = xb.to(DEVICE)
#             out = model(xb)
#             prob = torch.softmax(out, dim=1)[:,1]
#             y_prob.append(prob.detach().cpu().numpy())
#             y_true.append(yb.numpy())
#     return np.concatenate(y_true), np.concatenate(y_prob)

# def compute_metrics(y_true, y_prob, thr=0.5):
#     y_pred = (y_prob >= thr).astype(int)
#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, zero_division=0)
#     rec = recall_score(y_true, y_pred, zero_division=0)
#     f1 = f1_score(y_true, y_pred, zero_division=0)
#     try:
#         auc = roc_auc_score(y_true, y_prob)
#     except ValueError:
#         auc = float('nan')
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
#     pp = np.vstack([1 - y_prob, y_prob]).T
#     try:
#         ll = log_loss(y_true, pp, labels=[0,1])
#     except ValueError:
#         ll = float('nan')
#     return {
#         'accuracy': acc, 
#         'precision': prec, 
#         'recall': rec, 
#         'f1': f1, 
#         'specificity': spec, 
#         'roc_auc': auc, 
#         'loss': ll
#     }

# def train_eval_deep_for_modality(X, y, model_name: str):
#     from sklearn.model_selection import train_test_split

#     # split train/val/test
#     X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
#     X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

#     ds_tr = SequenceDataset(X_train, y_train)
#     ds_va = SequenceDataset(X_val, y_val, scaler=ds_tr.scaler)
#     ds_te = SequenceDataset(X_test, y_test, scaler=ds_tr.scaler)

#     dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
#     dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)
#     dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False)

#     in_feats = X.shape[-1]
#     if model_name == 'cnn':
#         model = TemporalCNN(in_feats)
#     elif model_name == 'lstm':
#         model = BiLSTM(in_feats)
#     elif model_name == 'cnn_lstm':
#         model = CNN_LSTM(in_feats)
#     elif model_name == 'transformer':
#         model = TransformerEncoderClassifier(in_feats)
#     else:
#         raise ValueError("unknown model")

#     round_metrics = {k: [] for k in ['accuracy','precision','recall','f1','specificity','roc_auc','loss']}
    
#     for r in range(ROUNDS):
#         set_seed(100 + r)
#         _ = train_one(model, dl_tr, dl_va, epochs=EPOCHS)
#         y_true, y_prob = predict_proba(model, dl_te)
#         m = compute_metrics(y_true, y_prob)
#         for k in round_metrics.keys():
#             round_metrics[k].append(m[k])
#     return round_metrics
