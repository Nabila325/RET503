"""Langkah 4: ukur latensi ResNet-18 vs MobileNetV3-Small (anggaran 15 FPS = 67 ms).

Pemakaian: python latency.py [--runs 100] [--device cpu|cuda]
Jalankan di perangkat robot (Raspberry Pi/Jetson) untuk angka yang relevan.
Keluaran: results/latency.csv
"""
import argparse, csv, time
from pathlib import Path
import cv2, numpy as np, torch
from torchvision import models

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # OpenCV BGR -> RGB
    x = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def bench(name, model, dev, runs, frame):
    model = model.to(dev).eval()
    x = preprocess(frame).to(dev)
    with torch.no_grad():
        for _ in range(20):                              # warm-up
            model(x)
        sync(dev)
        pre, inf = [], []
        for _ in range(runs):
            t0 = time.perf_counter(); xi = preprocess(frame).to(dev); t1 = time.perf_counter()
            model(xi); sync(dev); t2 = time.perf_counter()
            pre.append((t1 - t0) * 1000); inf.append((t2 - t1) * 1000)
    inf, pre = np.array(inf), np.array(pre)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    tot = inf.mean() + pre.mean()
    return {"model": name, "device": dev, "params_M": round(params, 2),
            "preprocess_ms": round(pre.mean(), 2), "inference_ms": round(inf.mean(), 2),
            "inference_p95_ms": round(np.percentile(inf, 95), 2), "total_ms": round(tot, 2),
            "fps": round(1000 / tot, 1), "inferensi_le_35ms": bool(inf.mean() <= 35)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    n_cls = 2  # pena, kotak_kabel
    r18 = models.resnet18(weights=None); r18.fc = torch.nn.Linear(512, n_cls)
    mb = models.mobilenet_v3_small(weights=None); mb.classifier[-1] = torch.nn.Linear(1024, n_cls)
    res = [bench("ResNet-18", r18, a.device, a.runs, frame),
           bench("MobileNetV3-Small", mb, a.device, a.runs, frame)]
    Path("results").mkdir(exist_ok=True)
    with open("results/latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=res[0].keys()); w.writeheader(); w.writerows(res)
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
