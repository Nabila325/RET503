"""Uji model hasil training.

1) Uji langsung lewat kamera (undistort -> BGR->RGB -> resize 224 -> normalisasi, sama dengan training):
       python predict.py --cam 2
       python predict.py --cam 2 --mode partial --thr 0.8
   Tombol: S = simpan screenshot ke test_shots/ | Q = keluar

2) Uji angka pada data val (confusion matrix + daftar citra yang salah):
       python predict.py --eval
       python predict.py --eval --mode scratch
"""
import argparse, csv, time
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

ROOT = Path("dataset_raw")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_model(mode, dev):
    ck = torch.load(f"models/resnet18_{mode}.pt", map_location=dev)
    classes = ck["classes"]
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(ck["state_dict"])
    return m.to(dev).eval(), classes


def prep(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def evaluate(m, classes, dev):
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN.tolist(), STD.tolist())])
    rows = [r for r in csv.DictReader(open(ROOT / "split.csv")) if r["split"] == "val"]
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), int)
    salah = []
    with torch.no_grad():
        for r in rows:
            img = Image.open(ROOT / r["kelas"] / r["nama_file"]).convert("RGB")
            p = m(tf(img).unsqueeze(0).to(dev)).softmax(1)[0]
            pred = int(p.argmax())
            cm[idx[r["kelas"]], pred] += 1
            if pred != idx[r["kelas"]]:
                salah.append((r["nama_file"], classes[pred], float(p[pred])))
    acc = np.trace(cm) / cm.sum()
    print(f"\nAkurasi val: {acc:.4f}  ({np.trace(cm)}/{cm.sum()})\n")
    print("Confusion matrix (baris = asli, kolom = prediksi):")
    print(" " * 14 + "".join(f"{c:>14}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:<14}" + "".join(f"{v:>14}" for v in cm[i]))
    print(f"\nCitra salah prediksi: {len(salah)}")
    for f, p, c in salah[:20]:
        print(f"  {f} -> {p} ({c:.2f})")


def live(m, classes, dev, a):
    cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Kamera tidak bisa dibuka. Coba --cam lain.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(30):
        ok, f = cap.read(); time.sleep(0.03)
    h, w = f.shape[:2]
    mapx = mapy = None
    if Path("calib.npz").exists():
        d = np.load("calib.npz")
        if "size" in d and tuple(d["size"]) != (w, h):
            print("PERINGATAN: resolusi beda dengan kalibrasi, undistort dimatikan.")
        else:
            mapx, mapy = cv2.initUndistortRectifyMap(d["mtx"], d["dist"], None, d["mtx"], (w, h), cv2.CV_32FC1)
            print("Undistort aktif.")
    Path("test_shots").mkdir(exist_ok=True)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mapx is not None:
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        t0 = time.perf_counter()
        with torch.no_grad():
            p = m(prep(frame).to(dev)).softmax(1)[0]
        ms = (time.perf_counter() - t0) * 1000
        conf, k = float(p.max()), int(p.argmax())
        label = classes[k] if conf >= a.thr else "tidak yakin"
        color = (0, 200, 0) if conf >= a.thr else (0, 165, 255)
        view = frame.copy()
        cv2.putText(view, f"{label}  {conf*100:.1f}%", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(view, f"{ms:.1f} ms | {a.mode}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("predict (S screenshot, Q keluar)", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            n += 1
            cv2.imwrite(f"test_shots/{a.mode}_{n:02d}_{label}.png", view)
            print("Screenshot disimpan")
    cap.release(); cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="feature", choices=["feature", "partial", "scratch"])
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.8, help="di bawah ini ditampilkan 'tidak yakin'")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, classes = load_model(a.mode, dev)
    print("Kelas:", classes)
    evaluate(m, classes, dev) if a.eval else live(m, classes, dev, a)


if __name__ == "__main__":
    main()
