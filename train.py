"""Langkah 3: latih ResNet-18 dengan 3 mode: feature | partial | scratch.

Pemakaian:
    python train.py                 # jalankan ketiga mode berurutan
    python train.py --mode partial  # satu mode saja
Keluaran (folder results/ dan models/):
    log_<mode>.csv, results_table.csv, akurasi_per_epoch.png, resnet18_<mode>.pt
"""
import argparse, csv, json, random, time
from pathlib import Path
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

ROOT = Path("dataset_raw")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class DS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y  # PIL = RGB


def load_data(batch, workers):
    rows = list(csv.DictReader(open(ROOT / "split.csv")))
    classes = sorted({r["kelas"] for r in rows})
    idx = {c: i for i, c in enumerate(classes)}
    get = lambda s: [(ROOT / r["kelas"] / r["nama_file"], idx[r["kelas"]]) for r in rows if r["split"] == s]
    tr_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                       T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.ToTensor(), T.Normalize(MEAN, STD)])
    va_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    tr, va = get("train"), get("val")
    tl = DataLoader(DS(tr, tr_tf), batch, shuffle=True, num_workers=workers, drop_last=len(tr) > batch)
    vl = DataLoader(DS(va, va_tf), batch, shuffle=False, num_workers=workers)
    return tl, vl, classes


def build(mode, n_cls):
    m = models.resnet18(weights=None if mode == "scratch" else models.ResNet18_Weights.IMAGENET1K_V1)
    if mode in ("feature", "partial"):
        for p in m.parameters():
            p.requires_grad = False
    if mode == "partial":
        for p in m.layer4.parameters():
            p.requires_grad = True
    m.fc = nn.Linear(m.fc.in_features, n_cls)  # head baru, requires_grad=True
    if mode == "feature":
        groups = [{"params": m.fc.parameters(), "lr": 1e-3}]
    elif mode == "partial":
        groups = [{"params": m.layer4.parameters(), "lr": 1e-4}, {"params": m.fc.parameters(), "lr": 1e-3}]
    else:
        groups = [{"params": m.parameters(), "lr": 1e-3}]
    return m, torch.optim.Adam(groups)


def set_mode(m, mode):
    """Lapisan beku tetap eval() agar statistik BatchNorm ImageNet tidak berubah."""
    if mode == "scratch":
        m.train()
    else:
        m.eval()
        m.fc.train()
        if mode == "partial":
            m.layer4.train()


@torch.no_grad()
def evaluate(m, dl, dev, crit):
    m.eval()
    loss = correct = n = 0
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        o = m(x)
        loss += crit(o, y).item() * len(y)
        correct += (o.argmax(1) == y).sum().item()
        n += len(y)
    return loss / n, correct / n


def run(mode, a, dev):
    torch.manual_seed(a.seed); random.seed(a.seed)
    tl, vl, classes = load_data(a.batch, a.workers)
    m, opt = build(mode, len(classes))
    m.to(dev)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss()
    log, best, best_ep, reach90 = [], 0.0, 0, None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        set_mode(m, mode)
        tl_loss = tc = tn = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            o = m(x)
            loss = crit(o, y)
            loss.backward()
            opt.step()
            tl_loss += loss.item() * len(y); tc += (o.argmax(1) == y).sum().item(); tn += len(y)
        sched.step()
        vloss, vacc = evaluate(m, vl, dev, crit)
        log.append([ep, tl_loss / tn, tc / tn, vloss, vacc])
        if vacc > best:
            best, best_ep = vacc, ep
            Path("models").mkdir(exist_ok=True)
            torch.save({"state_dict": m.state_dict(), "classes": classes}, f"models/resnet18_{mode}.pt")
        if reach90 is None and vacc >= 0.9:
            reach90 = ep
        print(f"[{mode}] ep {ep:02d}/{a.epochs} train_acc={tc/tn:.3f} val_acc={vacc:.3f} val_loss={vloss:.3f}")
    secs = time.time() - t0
    Path("results").mkdir(exist_ok=True)
    with open(f"results/log_{mode}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        w.writerows(log)
    return {"mode": mode, "best_val_acc": round(best, 4), "best_epoch": best_ep,
            "train_time_s": round(secs, 1), "epoch_acc_ge_90": reach90 if reach90 else "-", "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "feature", "partial", "scratch"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)
    modes = ["feature", "partial", "scratch"] if a.mode == "all" else [a.mode]
    res = [run(md, a, dev) for md in modes]

    with open("results/results_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "best_val_acc", "best_epoch", "train_time_s", "epoch_acc_ge_90"])
        for r in res:
            w.writerow([r[k] for k in ["mode", "best_val_acc", "best_epoch", "train_time_s", "epoch_acc_ge_90"]])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    for r in res:
        plt.plot([l[0] for l in r["log"]], [l[4] for l in r["log"]], marker="o", label=r["mode"])
    plt.axhline(0.9, ls="--", c="gray", lw=0.8)
    plt.xlabel("Epoch"); plt.ylabel("Akurasi validasi"); plt.ylim(0, 1.02)
    plt.title("Akurasi validasi per epoch (ResNet-18)"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("results/akurasi_per_epoch.png", dpi=150)
    print("\nRingkasan:")
    for r in res:
        print({k: v for k, v in r.items() if k != "log"})


if __name__ == "__main__":
    main()
