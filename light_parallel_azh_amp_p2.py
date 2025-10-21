# light_parallel_azh_amp.py
import os, math, argparse, json, csv, random, re, glob
from pathlib import Path
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

# ---------------- plotting & metrics (headless-safe) ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)
import cv2

# ---------------- Utils ----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def top1_acc(logits, target):
    return (logits.argmax(1).eq(target).float().mean()).item()

# --------------- MixUp/CutMix ---------------
def mixup_data(x, y, alpha=0.15):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

def cutmix_data(x, y, alpha=0.0):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    b, _, H, W = x.size()
    idx = torch.randperm(b, device=x.device)
    cut_rat = math.sqrt(1. - lam)
    cw, ch = int(W*cut_rat), int(H*cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = np.clip(cx - cw//2, 0, W), np.clip(cy - ch//2, 0, H)
    x2, y2 = np.clip(cx + cw//2, 0, W), np.clip(cy + ch//2, 0, H)
    x2c = x.clone()
    x2c[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2-x1)*(y2-y1)/(W*H))
    return x2c, y, y[idx], lam

def mix_criterion(criterion, pred, ya, yb, lam):
    return lam*criterion(pred, ya) + (1-lam)*criterion(pred, yb)

# --------------- Attention blocks ---------------
class ECA(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=(k-1)//2, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self,x):
        y = self.pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2))
        y = self.sig(y.transpose(-1,-2).unsqueeze(-1))
        return x * y.expand_as(x)

class cSE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c//r, c, 1), nn.Sigmoid()
        )
    def forward(self,x): return x * self.attn(x)

class sSE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c,1,1); self.sig = nn.Sigmoid()
    def forward(self,x): return x * self.sig(self.conv(x))

class P_scSE(nn.Module):
    def __init__(self,c,r=8):
        super().__init__()
        self.cse, self.sse = cSE(c,r), sSE(c)
    def forward(self,x):
        a, b = self.cse(x), self.sse(x)
        return torch.max(a,b) + a

# --------------- Model: EfficientNet-B3 ⨉ MobileNetV3-Small ---------------
class EffB3_MNV3S_Parallel(nn.Module):
    def __init__(self, num_classes=6, drop=0.25, fuse_ch=512):
        super().__init__()
        eff = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.eff_feat = eff.features            # out 1536
        m3s = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.mnv3_feat = m3s.features          # out 576

        self.red_a = nn.Sequential(
            nn.Conv2d(1536,1536,3,padding=1,groups=1536,bias=False),
            nn.BatchNorm2d(1536), nn.ReLU(inplace=True),
            nn.Conv2d(1536,256,1,bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.red_b = nn.Sequential(
            nn.Conv2d(576,576,3,padding=1,groups=576,bias=False),
            nn.BatchNorm2d(576), nn.ReLU(inplace=True),
            nn.Conv2d(576,256,1,bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(512, fuse_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_ch), nn.ReLU(inplace=True),
            P_scSE(fuse_ch, r=8), ECA(3), nn.Dropout2d(drop)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(fuse_ch, 512), nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Linear(512, num_classes)
        )

    def forward(self,x):
        a = self.eff_feat(x)
        b = self.mnv3_feat(x)
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False)
        a, b = self.red_a(a), self.red_b(b)
        y = self.fuse(torch.cat([a,b],1))
        y = self.pool(y).flatten(1)
        return self.head(y)

# --------------- Losses ---------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__(); self.gamma=gamma; self.reduction=reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else: self.alpha=None
    def forward(self, logits, target):
        logpt = F.log_softmax(logits,1); pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt    = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        loss  = -(1-pt)**self.gamma * logpt
        if self.alpha is not None: loss *= self.alpha[target]
        return loss.mean() if self.reduction=='mean' else loss.sum()

# --------------- Data ---------------
def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.15,0.15,0.15,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, test_tf

class RemapTargets(Dataset):
    def __init__(self, base: datasets.ImageFolder, src_to_dst: dict):
        self.base=base; self.src_to_dst=src_to_dst
        self.samples=[(p,src_to_dst[y]) for (p,y) in base.samples if y in src_to_dst]
        self.classes=sorted(set(src_to_dst.values()))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        p,y = self.samples[i]; x = self.base.loader(p)
        if self.base.transform: x=self.base.transform(x)
        return x, y

# ---- helper to filter ImageFolder to a class subset (by folder names) ---
def _filter_imagefolder_by_classes(ds: datasets.ImageFolder, allowed_names):
    """
    Return a shallow copy of ImageFolder `ds` that only contains samples whose
    class (folder) name is in `allowed_names`. class_to_idx is reindexed to 0..K-1
    based on sorted(allowed_names).
    """
    allowed_names = sorted(list(allowed_names))
    new_c2i = {name: i for i, name in enumerate(allowed_names)}

    inv = {v: k for k, v in ds.class_to_idx.items()}  # old_idx -> class name
    new_samples = []
    for p, old_y in ds.samples:
        cname = inv[old_y]
        if cname in new_c2i:
            new_samples.append((p, new_c2i[cname]))

    # make a shallow clone preserving transforms
    ds_f = ds.__class__(root=ds.root, transform=ds.transform, target_transform=ds.target_transform)
    ds_f.samples = new_samples
    ds_f.targets = [y for _, y in new_samples]
    ds_f.class_to_idx = new_c2i
    ds_f.classes = allowed_names
    return ds_f

def build_datasets_aligned(data_root, img_size, allowed_classes=None):
    """
    Build train/val/test aligned to the TRAIN label space.
    If `allowed_classes` is provided (list of folder names), only those classes are kept.
    """
    tr_tf, te_tf = build_transforms(img_size)
    tr = datasets.ImageFolder(Path(data_root)/'train', tr_tf)
    va = datasets.ImageFolder(Path(data_root)/'val',   te_tf)
    te = datasets.ImageFolder(Path(data_root)/'test',  te_tf)

    # Optional class subset: filter TRAIN to define the label space
    if allowed_classes is not None:
        missing = [c for c in allowed_classes if c not in tr.class_to_idx]
        if missing:
            raise RuntimeError(f"Classes not found in train: {missing}")
        tr = _filter_imagefolder_by_classes(tr, allowed_classes)

    c2i = tr.class_to_idx  # ground truth mapping we align others to

    def map_to_train(ds):
        m={}
        for cname, idx in ds.class_to_idx.items():
            if cname not in c2i:
                # skip classes that are not in the (possibly reduced) train set
                continue
            m[idx]=c2i[cname]
        if len(m) == 0:
            raise RuntimeError("After filtering, no overlapping classes remain between train and this split.")
        return m

    va = RemapTargets(va, map_to_train(va))
    te = RemapTargets(te, map_to_train(te))
    names=[k for k,_ in sorted(c2i.items(), key=lambda kv: kv[1])]
    return tr, va, te, names

def class_weights_from_samples(ds, num_classes):
    counts = Counter([y for _,y in ds.samples]); total=sum(counts.values())
    return torch.tensor([total/(num_classes*counts.get(i,1)) for i in range(num_classes)], dtype=torch.float32)

# --------------- Metrics & Interpretability helpers ---------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)

@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = _softmax_probs(logits).detach().cpu().numpy()
        yp = logits.argmax(1).detach().cpu().numpy()
        y_true.append(y.numpy()); y_pred.append(yp); y_prob.append(prob)
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred); y_prob = np.concatenate(y_prob, axis=0)
    return y_true, y_pred, y_prob

def prf1_macro(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return float(p), float(r), float(f1)

def plot_training_curves(history, out_path_png):
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure(figsize=(10,7))
    plt.subplot(2,2,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend()
    plt.subplot(2,2,2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'],   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Acc (%)'); plt.title('Accuracy'); plt.legend()
    plt.subplot(2,2,3)
    plt.plot(epochs, history['train_prec'], label='Train Prec')
    plt.plot(epochs, history['val_prec'],   label='Val Prec')
    plt.xlabel('Epoch'); plt.ylabel('Precision'); plt.title('Precision (macro)'); plt.legend()
    plt.subplot(2,2,4)
    plt.plot(epochs, history['train_recall'], label='Train Recall')
    plt.plot(epochs, history['val_recall'],   label='Val Recall')
    plt.plot(epochs, history['train_f1'], '--', label='Train F1')
    plt.plot(epochs, history['val_f1'],   '--', label='Val F1')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Recall & F1 (macro)'); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_png, bbox_inches='tight'); plt.close()

def plot_confmat(cm, classes, out_path_png, normalize=True):
    cmx = cm.astype('float')
    if normalize and cm.sum(axis=1)[:, None].sum() > 0:
        cmx = cmx / (cm.sum(axis=1)[:, None] + 1e-12)
    plt.figure(figsize=(7,6))
    plt.imshow(cmx, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix' + (' (normalized)' if normalize else ''))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right'); plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cmx.max() / 2.
    for i in range(cmx.shape[0]):
        for j in range(cmx.shape[1]):
            v = cmx[i, j]
            txt = f"{v:{fmt}}"
            plt.text(j, i, txt, ha="center", va="center",
                     color="white" if v > thresh else "black")
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout(); plt.savefig(out_path_png, bbox_inches='tight'); plt.close()

def plot_roc_ovr(y_true, y_prob, class_names, out_path_png):
    n_classes = len(class_names)
    y_true_bin = np.eye(n_classes)[y_true]
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure(figsize=(8,7))
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC={roc_auc['micro']:.3f})", linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"], label=f"macro-average (AUC={roc_auc['macro']:.3f})", linewidth=2)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], alpha=0.6, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC (one-vs-rest)'); plt.legend(fontsize=8, loc='lower right')
    plt.tight_layout(); plt.savefig(out_path_png, bbox_inches='tight'); plt.close()

def save_history_csv(history, out_csv):
    fields = list(history.keys())
    rows = zip(*[history[k] for k in fields])
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(fields); w.writerows(rows)

def denormalize(img_t):
    x = img_t.clone().cpu()
    for c in range(3):
        x[c] = x[c]*IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return x.clamp(0,1)

def overlay_heatmap(rgb, heatmap):
    hm = (heatmap*255.0).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
    overlay = (0.4*hm_color + 0.6*rgb)
    return np.clip(overlay, 0, 1)

def gradcam_single(model, x, target_layer, device, class_idx=None):
    """
    Grad-CAM using a forward hook to capture activations and a TENSOR
    gradient hook to capture d(score)/d(activation).
    """
    model.eval()
    feats = []
    grads = []

    def f_hook(_, __, output):
        feats.append(output)
        output.register_hook(lambda g: grads.append(g.clone()))

    h = target_layer.register_forward_hook(f_hook)

    x = x.to(device)
    with torch.enable_grad():
        logits = model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        score = logits[:, class_idx].sum()

        model.zero_grad(set_to_none=True)
        score.backward()

    A = feats[0].detach()[0]   # [C,H,W]
    G = grads[0].detach()[0]   # [C,H,W]

    weights = G.mean(dim=(1, 2))           # [C]
    cam = (weights[:, None, None] * A).sum(0)  # [H,W]
    cam = torch.relu(cam)

    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    h.remove()
    return cam.cpu().numpy(), class_idx

@torch.no_grad()
def compute_epoch_prf1(model, loader, device):
    y_true, y_pred, _ = collect_preds(model, loader, device)
    return prf1_macro(y_true, y_pred)

def _last_conv_in(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

# --- conv discovery + even sampling helpers ---
def _conv_layers_in(module: nn.Module):
    """Return a list of (name, layer) for all Conv2d inside `module` in forward order."""
    convs = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            convs.append((name, m))
    return convs

def _even_pick(items, k):
    """Pick up to k items evenly spaced across the list (preserving order)."""
    if not items: return []
    if k >= len(items): return items
    idxs = np.linspace(0, len(items)-1, num=k, dtype=int).tolist()
    seen = set(); picked=[]
    for i in idxs:
        if i not in seen:
            picked.append(items[i]); seen.add(i)
    return picked

def save_multistage_gradcam(model, x1, class_idx, class_names, ytrue, ypred, out_path_png,
                            per_branch=4, include_reducers=True, include_fuse=True):
    """
    Multistage Grad-CAM across many intermediate layers.
    """
    H, W = x1.shape[-2:]
    dev = next(model.parameters()).device

    eff_convs = _conv_layers_in(model.eff_feat)
    m3s_convs = _conv_layers_in(model.mnv3_feat)

    eff_pick = _even_pick(eff_convs, per_branch)
    m3s_pick = _even_pick(m3s_convs, per_branch)

    layers = []
    labels = []

    for name, layer in eff_pick:
        layers.append(layer); labels.append(f"eff:{name}")
    for name, layer in m3s_pick:
        layers.append(layer); labels.append(f"mnv3:{name}")

    if include_reducers:
        layers.append(model.red_a[0]); labels.append("red_a[0] (dw)")
    if include_reducers:
        layers.append(model.red_b[0]); labels.append("red_b[0] (dw)")
    if include_fuse:
        layers.append(model.fuse[0]);  labels.append("fuse[0]")

    if len(layers) == 0:
        return

    cams = []
    for L in layers:
        cam, _ = gradcam_single(model, x1, L, dev, class_idx=class_idx)
        cam = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-12)
        cams.append(cam)

    rgb = denormalize(x1[0]).permute(1,2,0).numpy()
    cols = len(layers)
    plt.figure(figsize=(3*cols, 6))

    for i in range(cols):
        plt.subplot(2, cols, i+1)
        plt.imshow(cams[i], cmap='jet'); plt.axis('off')
        plt.title(labels[i], fontsize=8)

    for i in range(cols):
        plt.subplot(2, cols, cols+i+1)
        ov = overlay_heatmap(rgb, cams[i])
        plt.imshow(ov); plt.axis('off')
        if i == 0:
            plt.title(f"True:{class_names[ytrue]} | Pred:{class_names[ypred]}", fontsize=9)
        else:
            plt.title("overlay", fontsize=8)

    plt.tight_layout(); plt.savefig(out_path_png, bbox_inches='tight'); plt.close()

def save_sample_heatmaps(model, loader, class_names, device, out_path_png, max_samples=8):
    """
    Grid: image | Grad-CAM | overlay, with Actual vs Predicted labels.
    Picks random images with class diversity: at most one sample per class.
    """
    model.eval()
    per_class = {i: [] for i in range(len(class_names))}
    with torch.no_grad():
        for xb, yb in loader:
            xb_dev = xb.to(device)
            logits = model(xb_dev)
            yp = logits.argmax(1).cpu()
            probs = torch.softmax(logits, dim=1).cpu()
            for i in range(xb.size(0)):
                ytrue_i = int(yb[i])
                ypred_i = int(yp[i])
                conf_i  = float(probs[i, ypred_i])
                per_class[ytrue_i].append((xb[i].cpu(), ytrue_i, ypred_i, conf_i))

    available_classes = [c for c, lst in per_class.items() if len(lst) > 0]
    if len(available_classes) == 0:
        return

    random.shuffle(available_classes)
    take_n = min(max_samples, len(available_classes))
    chosen_classes = available_classes[:take_n]

    samples = []
    for c in chosen_classes:
        ximg, ytrue, ypred, conf = random.choice(per_class[c])
        samples.append((ximg, ytrue, ypred, conf))

    rows = len(samples)
    if rows == 0:
        return

    plt.figure(figsize=(9, rows*3))
    dev = next(model.parameters()).device

    for r, (ximg, ytrue, ypred, conf) in enumerate(samples, start=1):
        x1 = ximg.unsqueeze(0)
        target_conv = model.fuse[0]

        cam, _ = gradcam_single(model, x1, target_conv, dev, class_idx=ypred)
        H, W = ximg.shape[-2:]
        cam_resized = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_resized, size=(H, W), mode='bilinear', align_corners=False)
        cam_resized = cam_resized.squeeze().cpu().numpy()
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() + 1e-12)

        rgb = denormalize(ximg).permute(1,2,0).numpy()
        overlay = overlay_heatmap(rgb, cam_resized)

        plt.subplot(rows, 3, 3*(r-1)+1); plt.imshow(rgb); plt.axis('off'); plt.title('Image')
        plt.subplot(rows, 3, 3*(r-1)+2); plt.imshow(cam_resized, cmap='jet'); plt.axis('off'); plt.title("Grad-CAM")
        plt.subplot(rows, 3, 3*(r-1)+3); plt.imshow(overlay); plt.axis('off')
        ok = "✓" if ytrue == ypred else "✗"
        plt.title(f"Actual:{class_names[ytrue]} | Pred:{class_names[ypred]} ({conf:.2f}) {ok}")

    plt.tight_layout(); plt.savefig(out_path_png, bbox_inches='tight'); plt.close()

# --------------- Eval helper (KEEPING ORIGINAL BATCH-AVERAGE ACC) ---------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ce = nn.CrossEntropyLoss().to(device)
    losses=[]; accs=[]
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(x); loss = ce(logits,y)
        losses.append(loss.item()); accs.append(top1_acc(logits,y))
    return float(np.mean(losses)), float(np.mean(accs))
# @torch.no_grad()
# def evaluate(model, loader, device):
#     model.eval(); ce = nn.CrossEntropyLoss().to(device)
#     total_loss, total_correct, total_examples = 0.0, 0, 0
#     for x,y in loader:
#         x,y = x.to(device), y.to(device)
#         with torch.amp.autocast('cuda'):
#             logits = model(x); loss = ce(logits,y)
#         bs = y.size(0)
#         total_loss += loss.item() * bs
#         total_correct += (logits.argmax(1) == y).sum().item()
#         total_examples += bs
#     return total_loss/total_examples, total_correct/total_examples

# --------------- Train/Eval (single trial) ---------------
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler,
                    mixup=0.15, cutmix=0.0):
    model.train(); losses=[]; accs=[]
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        lam, ya, yb = 1.0, y, y
        if cutmix>0: x, ya, yb, lam = cutmix_data(x,y,cutmix)
        elif mixup>0: x, ya, yb, lam = mixup_data(x,y,mixup)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logits = model(x)
            loss = mix_criterion(criterion, logits, ya, yb, lam) if lam<1.0 else criterion(logits,y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        if scheduler: scheduler.step()

        losses.append(loss.item()); accs.append(top1_acc(logits,y))
    return float(np.mean(losses)), float(np.mean(accs))

class EarlyStop:
    def __init__(self, patience=12, mode='max', min_delta=0.0):
        self.pat=patience; self.mode=mode; self.md=min_delta
        self.best=None; self.bad=0
    def step(self, val):
        if self.best is None: self.best=val; return False
        imp = (val>self.best+self.md) if self.mode=='max' else (val<self.best-self.md)
        if imp: self.best=val; self.bad=0; return False
        self.bad+=1; return self.bad>self.pat

# --------------- Weights I/O ---------------
def load_model_weights(model, ckpt_path, device, strict=False):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if not strict:
        if len(missing)>0:
            print(f"[load] Missing keys ({len(missing)}): first 5 -> {missing[:5]}")
        if len(unexpected)>0:
            print(f"[load] Unexpected keys ({len(unexpected)}): first 5 -> {unexpected[:5]}")
    print(f"[load] Loaded weights from: {ckpt_path}")

def discover_checkpoints(args):
    """
    Returns a list of (label, path) tuples to evaluate.
    Priority:
      - If --weights is a file: evaluate that file only.
      - If --weights is a directory: evaluate all *.pt under it (best-first).
      - Else: evaluate best.pt for each trial_XX in --save_dir up to --trials.
    """
    if args.weights:
        p = Path(args.weights)
        if p.is_file():
            return [(p.stem, str(p))]
        if p.is_dir():
            found = sorted(Path(p).glob("**/*.pt"))
            return [(fp.parent.name + "/" + fp.name, str(fp)) for fp in found]
        found = sorted(Path().glob(args.weights))
        return [(fp.parent.name + "/" + fp.name, str(fp)) for fp in found]

    ckpts=[]
    for t in range(args.trials):
        trial_dir = Path(args.save_dir)/f"trial_{t:02d}"
        cand = trial_dir/'best.pt'
        if cand.exists():
            ckpts.append((f"trial_{t:02d}", str(cand)))
    return ckpts

# --------------- Single-trial run (train) ---------------
def run_single_trial(tidx, base_seed, device, args, train_ds, val_ds, test_ds, class_names):
    seed = base_seed + tidx; set_seed(seed)
    tdir = Path(args.save_dir)/f"trial_{tidx:02d}"; tdir.mkdir(parents=True, exist_ok=True)
    outdir = tdir / 'output'; outdir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = EffB3_MNV3S_Parallel(num_classes=len(class_names), drop=0.25, fuse_ch=512).to(device)
    print(f"[Trial {tidx:02d}] Params: {count_params(model)/1e6:.2f}M")

    # optional init from weights for training
    if args.weights and Path(args.weights).is_file():
        load_model_weights(model, args.weights, device, strict=args.strict_load)

    # optional warmup: freeze backbones for a few epochs
    freeze_epochs = args.freeze_epochs
    for p in model.eff_feat.parameters(): p.requires_grad = (freeze_epochs == 0)
    for p in model.mnv3_feat.parameters(): p.requires_grad = (freeze_epochs == 0)

    cls_w = class_weights_from_samples(train_ds, len(class_names)).to(device)
    if args.focal:
        criterion = FocalLoss(alpha=cls_w/cls_w.sum())
    else:
        criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=args.label_smoothing)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.15, anneal_strategy='cos', final_div_factor=25.0
    )
    scaler = torch.amp.GradScaler('cuda')
    stopper = EarlyStop(patience=12)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'train_prec': [], 'train_recall': [], 'train_f1': [],
        'val_prec':   [], 'val_recall':   [], 'val_f1':   [],
    }

    best_acc=-1e9; best_path=str(tdir/'best.pt')
    for epoch in range(1, args.epochs+1):
        if epoch==freeze_epochs+1 and freeze_epochs>0:
            for p in model.eff_feat.parameters(): p.requires_grad=True
            for p in model.mnv3_feat.parameters(): p.requires_grad=True
            print(f"[Trial {tidx:02d}] Unfrozen backbones at epoch {epoch}")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler,
                                          mixup=args.mixup, cutmix=args.cutmix)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[Trial {tidx:02d}] Epoch {epoch:03d} | Train {tr_acc:.2f}% | Val {val_acc:.2f}% | Loss {val_loss:.4f}")

        tr_p, tr_r, tr_f1 = compute_epoch_prf1(model, train_loader, device)
        va_p, va_r, va_f1 = compute_epoch_prf1(model, val_loader, device)
        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss);  history['val_acc'].append(val_acc)
        history['train_prec'].append(tr_p); history['train_recall'].append(tr_r); history['train_f1'].append(tr_f1)
        history['val_prec'].append(va_p);   history['val_recall'].append(va_r);   history['val_f1'].append(va_f1)
        save_history_csv(history, tdir/'history.csv')
        plot_training_curves(history, outdir/'training_curves.png')

        if val_acc>best_acc:
            best_acc=val_acc; torch.save({'model':model.state_dict()}, best_path)

        if stopper.step(val_acc):
            print(f"[Trial {tidx:02d}] Early stop."); break

    if not Path(best_path).exists(): torch.save({'model':model.state_dict()}, best_path)

    ckpt=torch.load(best_path, map_location=device); model.load_state_dict(ckpt['model'])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[Trial {tidx:02d}] Final Test {test_acc:.4f}% | Loss {test_loss:.4f}")

    # ----- artifacts (val + test) -----
    y_true_v, y_pred_v, y_prob_v = collect_preds(model, val_loader, device)
    cm_v = confusion_matrix(y_true_v, y_pred_v, labels=list(range(len(class_names))))
    plot_confmat(cm_v, class_names, outdir/'val_confusion_matrix.png', normalize=True)
    plot_roc_ovr(y_true_v, y_prob_v, class_names, outdir/'val_roc.png')

    y_true_t, y_pred_t, y_prob_t = collect_preds(model, test_loader, device)
    cm_t = confusion_matrix(y_true_t, y_pred_t, labels=list(range(len(class_names))))
    plot_confmat(cm_t, class_names, outdir/'test_confusion_matrix.png', normalize=True)
    plot_roc_ovr(y_true_t, y_prob_t, class_names, outdir/'test_roc.png')

    p_v, r_v, f1_v = prf1_macro(y_true_v, y_pred_v)
    p_t, r_t, f1_t = prf1_macro(y_true_t, y_pred_t)
    with open(tdir/'metrics_summary.json','w') as f:
        json.dump({
            'val': {'precision_macro': p_v, 'recall_macro': r_v, 'f1_macro': f1_v,
                    'acc': float((y_true_v==y_pred_v).mean())},
            'test': {'precision_macro': p_t, 'recall_macro': r_t, 'f1_macro': f1_t,
                     'acc': float((y_true_t==y_pred_t).mean())}
        }, f, indent=2)

    # ----- Interpretability -----
    correct_mask = (y_true_t == y_pred_t)
    if correct_mask.any():
        conf = y_prob_t[np.arange(len(y_prob_t)), y_pred_t]
        conf_correct = conf.copy(); conf_correct[~correct_mask] = -1.0
        best_idx = int(conf_correct.argmax())
        cum = 0; x_best = None; yb = None; pb = None
        for xb, yb_ in test_loader:
            bs = xb.size(0)
            if cum <= best_idx < cum+bs:
                x_best = xb[best_idx-cum].unsqueeze(0)
                yb = int(yb_[best_idx-cum].item())
                pb = int(y_pred_t[best_idx])
                break
            cum += bs
        if x_best is not None:
            target_conv = model.fuse[0]
            dev = next(model.parameters()).device
            cam, cls_idx = gradcam_single(model, x_best, target_conv, dev, class_idx=pb)
            H, W = x_best.shape[-2:]
            cam_resized = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
            cam_resized = F.interpolate(cam_resized, size=(H, W), mode='bilinear', align_corners=False)
            cam_resized = cam_resized.squeeze().cpu().numpy()
            cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() + 1e-12)
            rgb = denormalize(x_best[0]).permute(1,2,0).numpy()
            overlay = overlay_heatmap(rgb, cam_resized)
            plt.figure(figsize=(9,4))
            plt.subplot(1,3,1); plt.imshow(rgb); plt.axis('off'); plt.title('Image')
            plt.subplot(1,3,2); plt.imshow(cam_resized, cmap='jet'); plt.axis('off'); plt.title('Grad-CAM')
            plt.subplot(1,3,3); plt.imshow(overlay); plt.axis('off')
            plt.title(f"Pred: {class_names[pb]}\nTrue: {class_names[yb]}")
            plt.tight_layout(); plt.savefig(outdir/'interpretability_gradcam.png', bbox_inches='tight'); plt.close()

            # multistage panel
            save_multistage_gradcam(model, x_best, pb, class_names, yb, pb, outdir/'gradcam_stages.png')

    save_sample_heatmaps(model, test_loader, class_names, device, outdir/'test_samples_with_heatmaps.png', max_samples=8)

    with open(tdir/'result.json','w') as f:
        json.dump({'trial':tidx,'seed':seed,'val_best_acc':best_acc,'test_acc':test_acc,'ckpt':best_path}, f, indent=2)
    return {'trial':tidx,'seed':seed,'val_best_acc':best_acc,'test_acc':test_acc,'ckpt':best_path}

# --------------- Main (multi-trial + test-only) ---------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_root',type=str,required=True)
    ap.add_argument('--img_size',type=int,default=256)
    ap.add_argument('--batch_size',type=int,default=64)
    ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--weight_decay',type=float,default=1e-4)
    ap.add_argument('--label_smoothing',type=float,default=0.0)
    ap.add_argument('--mixup',type=float,default=0.15)
    ap.add_argument('--cutmix',type=float,default=0.0)
    ap.add_argument('--focal',action='store_true')
    ap.add_argument('--freeze_epochs',type=int,default=5)
    ap.add_argument('--trials',type=int,default=10)
    ap.add_argument('--base_seed',type=int,default=42)
    ap.add_argument('--save_dir',type=str,default='runs/multirun')

    # class selection CLI
    ap.add_argument('--keep_classes', type=str, default=None,
                    help='Comma-separated class folder names to KEEP (e.g., "BG,N,D"). '
                         'If omitted, all classes found in train/ are used.')

    # weights + test-only options
    ap.add_argument('--weights', type=str, default=None,
                    help='Path to a checkpoint file/dir (or glob) to init/evaluate. Omit in --test_only to scan trials.')
    ap.add_argument('--strict_load', action='store_true',
                    help='Use strict=True when loading weights.')
    ap.add_argument('--test_only', action='store_true',
                    help='Only run evaluation on saved checkpoints and exit.')
    ap.add_argument('--eval_split', type=str, default='test', choices=['test','val'],
                    help='Which split to evaluate in --test_only mode.')
    args=ap.parse_args()

    set_seed(args.base_seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Parse keep_classes flag -> allowed list
    allowed = None
    if args.keep_classes:
        allowed = [s.strip() for s in args.keep_classes.split(',') if s.strip()]

    train_ds, val_ds, test_ds, class_names = build_datasets_aligned(
        args.data_root, args.img_size, allowed_classes=allowed
    )

    # -------- Test-only multi-eval --------
    if args.test_only:
        split_ds = test_ds if args.eval_split=='test' else val_ds
        loader = DataLoader(split_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        ckpts = discover_checkpoints(args)
        if len(ckpts)==0:
            raise FileNotFoundError("No checkpoints found for --test_only. "
                                    "Either pass --weights (file/dir/glob) or ensure trial_XX/best.pt exist under --save_dir.")

        def _safe(s: str) -> str:
            return re.sub(r'[^a-zA-Z0-9._-]+', '_', s)

        def _trial_dir_from_path(p: Path, save_dir: Path) -> Path:
            for parent in [p] + list(p.parents):
                if parent.name.startswith("trial_") and len(parent.name)==7:
                    return parent
            return None

        rows=[]
        print(f"[TEST-ONLY] Evaluating {len(ckpts)} checkpoint(s) on split: {args.eval_split}")
        for label, path in ckpts:
            model = EffB3_MNV3S_Parallel(num_classes=len(class_names), drop=0.25, fuse_ch=512).to(device)
            load_model_weights(model, path, device, strict=args.strict_load)

            loss, acc = evaluate(model, loader, device)
            print(f"[TEST-ONLY] {label:20s} | Loss: {loss:.4f} | Acc: {acc:.4f}%")
            m = re.search(r"trial_(\d{2})", path)
            trial = int(m.group(1)) if m else None
            rows.append({'label':label, 'trial':trial, 'split':args.eval_split, 'loss':loss, 'acc':acc, 'ckpt':path})

            # artifacts folder = same trial's output/ if present, else fallback
            p = Path(path)
            trial_dir = _trial_dir_from_path(p, Path(args.save_dir))
            if trial_dir is None:
                outdir = Path(args.save_dir) / 'test_only' / _safe(label) / 'output'
            else:
                outdir = trial_dir / 'output'
            outdir.mkdir(parents=True, exist_ok=True)

            # predictions
            y_true, y_pred, y_prob = collect_preds(model, loader, device)

            # confusion matrix + ROC
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
            plot_confmat(cm, class_names, outdir/'confmat.png', normalize=True)
            plot_roc_ovr(y_true, y_prob, class_names, outdir/'roc.png')

            # macro PR/F1
            p_, r_, f1_ = prf1_macro(y_true, y_pred)
            with open(outdir/'metrics.json','w') as f:
                json.dump({'split': args.eval_split,
                           'loss': float(loss), 'acc': float(acc),
                           'precision_macro': p_, 'recall_macro': r_, 'f1_macro': f1_}, f, indent=2)

            # Grad-CAM grid for samples (fusion layer)
            save_sample_heatmaps(model, loader, class_names, device, outdir/'samples_with_heatmaps.png', max_samples=8)

            # Best confident correct example + multistage CAM
            correct_mask = (y_true == y_pred)
            if correct_mask.any():
                conf = y_prob[np.arange(len(y_prob)), y_pred]
                conf_correct = conf.copy(); conf_correct[~correct_mask] = -1.0
                best_idx = int(conf_correct.argmax())

                cum = 0; x_best = None; yb = None; pb = None
                for xb, yb_ in loader:
                    bs = xb.size(0)
                    if cum <= best_idx < cum+bs:
                        x_best = xb[best_idx-cum].unsqueeze(0)
                        yb = int(yb_[best_idx-cum].item())
                        pb = int(y_pred[best_idx])
                        break
                    cum += bs

                if x_best is not None:
                    target_conv = model.fuse[0]
                    dev = next(model.parameters()).device
                    cam, _ = gradcam_single(model, x_best, target_conv, dev, class_idx=pb)
                    H, W = x_best.shape[-2:]
                    cam_resized = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
                    cam_resized = F.interpolate(cam_resized, size=(H, W), mode='bilinear', align_corners=False)
                    cam_resized = cam_resized.squeeze().cpu().numpy()
                    cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() + 1e-12)
                    rgb = denormalize(x_best[0]).permute(1,2,0).numpy()
                    overlay = overlay_heatmap(rgb, cam_resized)
                    plt.figure(figsize=(9,4))
                    plt.subplot(1,3,1); plt.imshow(rgb); plt.axis('off'); plt.title('Image')
                    plt.subplot(1,3,2); plt.imshow(cam_resized, cmap='jet'); plt.axis('off'); plt.title('Grad-CAM')
                    plt.subplot(1,3,3); plt.imshow(overlay); plt.axis('off')
                    plt.title(f"Pred: {class_names[pb]}\nTrue: {class_names[yb]}")
                    plt.tight_layout(); plt.savefig(outdir/'interpretability_gradcam.png', bbox_inches='tight'); plt.close()

                    save_multistage_gradcam(model, x_best, pb, class_names, yb, pb, outdir/'gradcam_stages.png')
            # -----------------------------------------

        # write CSV + best json
        csv_path = Path(args.save_dir)/f'test_only_{args.eval_split}_results.csv'
        with open(csv_path,'w',newline='') as f:
            w=csv.DictWriter(f, fieldnames=['label','trial','split','loss','acc','ckpt'])
            w.writeheader(); [w.writerow(r) for r in rows]

        best = max(rows, key=lambda r: r['acc'])
        with open(Path(args.save_dir)/f'test_only_{args.eval_split}_best.json','w') as f:
            json.dump(best, f, indent=2)

        print("--------------------------------")
        print(f"[TEST-ONLY] BEST → {best['label']} | Acc {best['acc']:.2f}% | {best['ckpt']}")
        print(f"[TEST-ONLY] CSV:  {csv_path}")
        print(f"[TEST-ONLY] Artifacts saved under each trial's 'output/' folder (or fallback under runs/multirun/test_only/<label>/output).")
        return
    # ------------------------------------

    with open(Path(args.save_dir)/'classes.json','w') as f: json.dump(class_names, f, indent=2)

    results=[]
    for t in range(args.trials):
        r = run_single_trial(t, args.base_seed, device, args, train_ds, val_ds, test_ds, class_names)
        results.append(r)

    csv_path = Path(args.save_dir)/'trials_results.csv'
    with open(csv_path,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=['trial','seed','val_best_acc','test_acc','ckpt']); w.writeheader()
        for r in results: w.writerow(r)

    best=max(results, key=lambda z: z['test_acc'])
    with open(Path(args.save_dir)/'best_overall.json','w') as f: json.dump(best,f,indent=2)

    print("\n===== Multi-trial summary =====")
    for r in results:
        print(f"Trial {r['trial']:02d} | seed {r['seed']} | Val* {r['val_best_acc']:.4f}% | Test {r['test_acc']:.4f}% | {r['ckpt']}")
    print("--------------------------------")
    print(f"BEST overall → Trial {best['trial']:02d} (seed {best['seed']}), Test {best['test_acc']:.4f}%")
    print(f"Checkpoint: {best['ckpt']}")
    print(f"CSV: {csv_path}")

if __name__ == '__main__':
    main()
