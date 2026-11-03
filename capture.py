"""Langkah 1: ambil citra dari kamera robot/webcam.

Pemakaian:
    python capture.py pena terang
    python capture.py kotak_kabel redup --cam 0

Tombol: SPASI = simpan 1 citra | A = auto-capture on/off (tiap 0.7 dtk) | Q = keluar
Jika ada calib.npz (mtx, dist dari pertemuan 2), citra otomatis di-undistort.
"""
import argparse, csv, datetime as dt, time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("dataset_raw")
META = ROOT / "metadata.csv"
HEADER = ["nama_file", "kelas", "tanggal", "kondisi_cahaya", "sesi"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kelas", help="pena | kotak_kabel")
    ap.add_argument("kondisi", help="mis. terang, redup, jendela, bayangan")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--calib", default="calib.npz")
    ap.add_argument("--backend", default="dshow", choices=["dshow", "msmf", "any"],
                    help="dshow (default, paling stabil di Windows) | msmf | any")
    ap.add_argument("--no-res", action="store_true",
                    help="jangan set resolusi (pakai default kamera)")
    a = ap.parse_args()

    (ROOT / a.kelas).mkdir(parents=True, exist_ok=True)
    if not META.exists():
        with open(META, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)

    be = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[a.backend]
    cap = cv2.VideoCapture(a.cam, be)
    if not cap.isOpened():
        raise SystemExit("Kamera tidak bisa dibuka. Coba --cam 1 atau --backend msmf. "
                         "Pastikan aplikasi Camera/Zoom/Teams sudah ditutup.")
    if not a.no_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)

    # warm-up: buang beberapa frame awal (sering hitam) dan cek apakah gambar benar-benar ada
    frame = None
    for _ in range(30):
        ok, frame = cap.read()
        time.sleep(0.03)
    if not ok or frame is None:
        raise SystemExit("Kamera terbuka tapi tidak ada frame. Coba --no-res atau --cam 1")
    if frame.mean() < 2:
        print("PERINGATAN: frame masih hitam. Coba: --no-res, --cam 1, atau --backend msmf")
    print(f"Kamera OK: {frame.shape[1]}x{frame.shape[0]} (backend {a.backend}, index {a.cam})")

    mapx = mapy = None
    if Path(a.calib).exists():
        d = np.load(a.calib)
        h, w = frame.shape[:2]
        if "size" in d and tuple(d["size"]) != (w, h):
            print(f"PERINGATAN: calib.npz dibuat di {tuple(d['size'])}, kamera sekarang {w}x{h}. "
                  "Undistort DIMATIKAN. Kalibrasi ulang di resolusi yang sama.")
        else:
            mapx, mapy = cv2.initUndistortRectifyMap(d["mtx"], d["dist"], None, d["mtx"], (w, h), cv2.CV_32FC1)
            print("Undistort aktif (calib.npz ditemukan).")
    else:
        print("PERINGATAN: calib.npz tidak ada, citra TIDAK di-undistort.")

    now = dt.datetime.now()
    tanggal = now.strftime("%Y%m%d")
    sesi = f"{tanggal}_{a.kondisi}_{now.strftime('%H%M%S')}"
    prefix = f"{a.kelas}_{tanggal}_lab_{a.kondisi}_"
    n = len(list((ROOT / a.kelas).glob(prefix + "*.png")))
    total_kelas = len(list((ROOT / a.kelas).glob("*.png")))
    auto, last = False, 0.0

    def save(img):
        nonlocal n, total_kelas
        n += 1
        name = f"{prefix}{n:03d}.png"
        cv2.imwrite(str(ROOT / a.kelas / name), img)
        with open(META, "a", newline="") as f:
            csv.writer(f).writerow([name, a.kelas, tanggal, a.kondisi, sesi])
        total_kelas += 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mapx is not None:
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        if auto and time.time() - last > 0.7:
            save(frame); last = time.time()
        view = frame.copy()
        txt = f"{a.kelas} | {a.kondisi} | total kelas: {total_kelas} | auto: {'ON' if auto else 'OFF'}"
        cv2.putText(view, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("capture (SPASI simpan, A auto, Q keluar)", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == 32:
            save(frame)
        elif k == ord("a"):
            auto = not auto
    cap.release()
    cv2.destroyAllWindows()
    print(f"Selesai. Total citra kelas '{a.kelas}': {total_kelas} (target >= 50)")


if __name__ == "__main__":
    main()
