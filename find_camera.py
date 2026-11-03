"""Cari index kamera yang benar (mis. RealSense D455 RGB).

Pemakaian: python find_camera.py
Mencoba index 0-6, menyimpan snapshot cam_<i>.png, dan menebak mana yang berwarna (RGB).
Buka file cam_<i>.png untuk memastikan, lalu pakai: python capture.py pena terang --cam <i>
"""
import time
import cv2
import numpy as np

print("Tutup RealSense Viewer / aplikasi Camera dulu.\n")
for i in range(7):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"index {i}: tidak ada")
        continue
    frame = None
    for _ in range(30):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
        time.sleep(0.03)
    cap.release()
    if frame is None:
        print(f"index {i}: terbuka tapi tidak ada frame")
        continue
    h, w = frame.shape[:2]
    b, g, r = [frame[:, :, c].astype(int) for c in range(3)]
    warna = np.abs(r - g).mean() + np.abs(g - b).mean()
    jenis = "BERWARNA -> kemungkinan RGB" if warna > 3 else "abu-abu/hitam -> depth atau IR"
    cv2.imwrite(f"cam_{i}.png", frame)
    print(f"index {i}: {w}x{h}, rata-rata kecerahan {frame.mean():.1f}, {jenis}  (cam_{i}.png)")
