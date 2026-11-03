"""Langkah 2: bagi train/val berdasarkan SESI pengambilan (mencegah data leakage).

Pemakaian: python split.py [--val 0.2] [--seed 42]
Keluaran : dataset_raw/split.csv (nama_file, kelas, split)
"""
import argparse, csv, random
from collections import defaultdict
from pathlib import Path

ROOT = Path("dataset_raw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--by", default="sesi", choices=["sesi", "kondisi"],
                    help="sesi (default) | kondisi: val = kondisi cahaya yang tidak ada di train")
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    rows = list(csv.DictReader(open(ROOT / "metadata.csv")))
    rows = [r for r in rows if (ROOT / r["kelas"] / r["nama_file"]).exists()]
    per_kelas = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_kelas[r["kelas"]][r[("kondisi_cahaya" if a.by == "kondisi" else "sesi")]].append(r["nama_file"])

    out = []
    for kelas, sesi_map in sorted(per_kelas.items()):
        total = sum(len(v) for v in sesi_map.values())
        target = max(1, round(total * a.val))
        val_files = set()
        if len(sesi_map) >= 2:
            sesi = list(sesi_map)
            rnd.shuffle(sesi)
            for s in sesi[:-1]:  # sisakan minimal 1 sesi untuk train
                if len(val_files) >= target:
                    break
                val_files.update(sesi_map[s])
            print(f"[{kelas}] split per {a.by} ({len(sesi_map)} grup: {', '.join(sorted(sesi_map)) if a.by == 'kondisi' else '...'})")
        else:
            files = sorted(f for v in sesi_map.values() for f in v)
            val_files = set(files[-target:])  # blok terakhir, bukan acak
            print(f"[{kelas}] PERINGATAN: hanya 1 sesi -> split blok kronologis. "
                  "Ambil data di >= 2 sesi/kondisi agar validasi lebih jujur.")
        for v in sesi_map.values():
            for f in v:
                out.append((f, kelas, "val" if f in val_files else "train"))

    with open(ROOT / "split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nama_file", "kelas", "split"])
        w.writerows(out)

    for kelas in sorted(per_kelas):
        tr = sum(1 for o in out if o[1] == kelas and o[2] == "train")
        va = sum(1 for o in out if o[1] == kelas and o[2] == "val")
        print(f"  {kelas}: train={tr} val={va}")


if __name__ == "__main__":
    main()
