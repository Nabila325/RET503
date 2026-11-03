"""Kalibrasi kamera dengan papan catur (chessboard) -> calib.npz (mtx, dist, size).

Pemakaian:
    python calibrate.py                          # papan 9x6 sudut dalam, kotak 25 mm
    python calibrate.py --cols 8 --rows 5 --square 30
    python calibrate.py --from-folder calib_images   # pakai foto yang sudah ada

Tombol: SPASI = simpan (hanya jika papan terdeteksi) | D = deteksi ukuran papan otomatis | C = hitung & simpan | Q = keluar
Cols/rows = jumlah SUDUT DALAM, bukan jumlah kotak (papan 10x7 kotak -> 9x6 sudut).
PENTING: pakai resolusi yang SAMA dengan capture.py (default 1280x720).
Ambil 15-25 foto dengan posisi, sudut miring, dan jarak berbeda; papan harus memenuhi frame di beberapa foto.
"""
import argparse, glob, time
from pathlib import Path
import cv2
import numpy as np


def calibrate(images, size, cols, rows, square):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    obj_pts, img_pts = [], []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for g in images:
        ok, c = cv2.findChessboardCorners(g, (cols, rows), None)
        if ok:
            obj_pts.append(objp)
            img_pts.append(cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit))
    if len(obj_pts) < 10:
        raise SystemExit(f"Hanya {len(obj_pts)} foto valid, minimal 10 (disarankan 15-25).")
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    np.savez("calib.npz", mtx=mtx, dist=dist, size=np.array(size))  # simpan dulu
    # reprojection error rata-rata (kompatibel OpenCV 4 dan 5)
    err = 0.0
    for i in range(len(obj_pts)):
        p, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
        a_ = np.asarray(img_pts[i], np.float64).reshape(-1, 2)
        b_ = np.asarray(p, np.float64).reshape(-1, 2)
        err += np.sqrt(((a_ - b_) ** 2).sum(axis=1)).mean()
    err /= len(obj_pts)
    print(f"\nFoto dipakai : {len(obj_pts)}")
    print(f"RMS          : {rms:.4f}")
    print(f"Reproj. error: {err:.4f} px  (bagus < 0.5; < 1.0 masih bisa diterima)")
    print(f"fx, fy       : {mtx[0,0]:.1f}, {mtx[1,1]:.1f}")
    print(f"cx, cy       : {mtx[0,2]:.1f}, {mtx[1,2]:.1f}")
    print(f"dist         : {dist.ravel()}")
    print("Tersimpan: calib.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=25.0, help="ukuran kotak (mm)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--backend", default="dshow", choices=["dshow", "msmf", "any"])
    ap.add_argument("--no-res", action="store_true")
    ap.add_argument("--from-folder", default=None)
    a = ap.parse_args()

    if a.from_folder:
        files = sorted(glob.glob(f"{a.from_folder}/*.png") + glob.glob(f"{a.from_folder}/*.jpg"))
        if not files:
            raise SystemExit("Folder kosong.")
        grays = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY) for f in files]
        h, w = grays[0].shape
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        if not cv2.findChessboardCorners(grays[0], (a.cols, a.rows), flags)[0]:
            print("Ukuran default tidak cocok, mencari ukuran papan otomatis...")
            hit = None
            for g in grays[:5]:
                for c in range(14, 2, -1):
                    for r in range(c, 2, -1):
                        if cv2.findChessboardCorners(g, (c, r), flags)[0]:
                            hit = (c, r); break
                    if hit: break
                if hit: break
            if not hit:
                raise SystemExit("Ukuran papan tidak ketemu. Isi manual --cols dan --rows (sudut dalam).")
            a.cols, a.rows = hit
            print(f"Ukuran terdeteksi: cols={a.cols}, rows={a.rows}")
        return calibrate(grays, (w, h), a.cols, a.rows, a.square)

    be = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[a.backend]
    cap = cv2.VideoCapture(a.cam, be)
    if not cap.isOpened():
        raise SystemExit("Kamera tidak bisa dibuka. Coba --cam 1 / --backend msmf")
    if not a.no_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    for _ in range(30):
        ok, frame = cap.read(); time.sleep(0.03)
    if not ok:
        raise SystemExit("Tidak ada frame dari kamera.")
    h, w = frame.shape[:2]
    print(f"Resolusi kalibrasi: {w}x{h} (pakai resolusi yang sama saat capture.py)")

    Path("calib_images").mkdir(exist_ok=True)
    grays, n = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (a.cols, a.rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, (a.cols, a.rows), corners, found)
        cv2.putText(view, f"foto: {len(grays)} | papan: {'TERDETEKSI' if found else 'tidak'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("kalibrasi (SPASI simpan, D deteksi ukuran, C hitung, Q keluar)", view)
        k = cv2.waitKey(1) & 0xFF
        if k == 32 and found:
            n += 1
            cv2.imwrite(f"calib_images/calib_{n:02d}.png", frame)
            grays.append(gray)
            print(f"Foto {len(grays)} disimpan")
        elif k == ord("d"):
            print("Mencari ukuran papan... (tahan papan diam, terlihat penuh)")
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            hit = None
            for c in range(14, 2, -1):
                for r in range(c, 2, -1):
                    if cv2.findChessboardCorners(gray, (c, r), flags)[0]:
                        hit = (c, r); break
                if hit: break
            if hit:
                a.cols, a.rows = hit
                print(f"TERDETEKSI: --cols {hit[0]} --rows {hit[1]} (sudah dipakai otomatis)")
            else:
                print("Tidak ketemu. Lihat tips di README/penjelasan (cahaya, papan rata, tepi putih).")
        elif k == ord("c"):
            cap.release(); cv2.destroyAllWindows()
            return calibrate(grays, (w, h), a.cols, a.rows, a.square)
        elif k == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
