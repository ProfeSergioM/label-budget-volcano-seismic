"""
Extrae los descriptores del corpus de Isla Decepcion con la misma tuberia.

Dataset: Cortes et al. (2018), Zenodo 10.5281/zenodo.1307437, CC-BY 4.0. Ver
experiments/scripts/inventario_decepcion.py para el formato, que el deposito
no documenta.

La unica adaptacion es el remuestreo de 50 a 100 Hz, y se hace ANTES de
filtrar. Con eso todo lo demas queda literalmente identico: las mismas bandas,
el mismo orden de filtro, el mismo m=2 y tau=3 -- que a 50 Hz serian 60 ms en
vez de 30 --, la misma ventana de duracion y el mismo criterio de epsilon. A
100 Hz el borde superior de 25 Hz deja de caer sobre el Nyquist, que es la
otra razon para remuestrear en vez de mover la banda.

No hace falta la respuesta del instrumento, que el deposito no entrega: los
cuatro descriptores son invariantes a la escala de amplitud. SR y FI_hi son
razones entre amplitudes espectrales, y DET y W_entr se calculan sobre la
serie tipificada.

FI_lo no se calcula. Los autores ya aplicaron un filtro que deja entre el 1,5
y el 10 % de la energia bajo 0,9 Hz, de modo que ese indice mediria la caida
de su filtro y no la senal.

Uso:
    python scripts/extract_deception.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import rqa_core as C  # noqa: E402

BASE = ROOT / "data" / "external" / "deception" / "VSR_DBs_@cortes-GRL.18"
OUT = ROOT / "data" / "processed" / "features_deception"
DBS = [("dec.95M.16c", 1995), ("dec.09U.4c", 2009)]
SPS_IN, SPS = 50.0, 100.0       # el remuestreo es por dos, exacto
UNIT = 1e-4                     # segundos por unidad de tiempo del mlf
KEEP = ("LP", "VT")             # NTR no tiene equivalente en la taxonomia OVDAS
FIVE = ["SR_w", "FI_hi", "DET", "Wentr"]


def parse_mlf(path: Path):
    segs, cur = [], None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith('"'):
            cur = os.path.basename(line.strip('"')).replace(".lab", ".raw")
        elif line in (".", "") or line.startswith("#!"):
            continue
        elif cur:
            p = line.split()
            if len(p) >= 3:
                segs.append((cur, int(p[0]) * UNIT, int(p[1]) * UNIT, p[2]))
    return segs


def extract(db: str, year: int) -> pd.DataFrame:
    raw_dir = BASE / db / "raw"
    rows, skipped = [], 0
    for f, a, b, label in parse_mlf(BASE / f"{db}.mlf"):
        if label not in KEEP or not (raw_dir / f).exists():
            continue
        dur = b - a
        if dur < C.MIN_DURATION_S:
            skipped += 1
            continue
        plan = C.duration_window_plan(dur, SPS)
        x = np.fromfile(raw_dir / f, dtype="<i2").astype(np.float64)
        # El plan esta en muestras de 100 Hz; los indices en el archivo, en
        # muestras de 50. Se recorta con margen y se remuestrea el tramo.
        i0 = int(round((a - plan["pre_s"]) * SPS_IN))
        i1 = i0 + int(round((plan["pre_s"] + plan["post_s"]) * SPS_IN))
        if i0 < 0 or i1 > len(x):
            skipped += 1
            continue
        seg = resample_poly(x[i0:i1], 2, 1)
        if len(seg) < plan["total_samples"]:
            skipped += 1
            continue
        seg = seg[:plan["total_samples"]]

        wide = C.preprocess_segment(seg, SPS, band=C.WIDE_FILTER_BAND)
        rqa_f = C.preprocess_segment(seg, SPS, band=C.DURATION_FILTER_BAND)
        w_sig = wide[plan["signal_start"]:plan["signal_end"]]
        r_sig = rqa_f[plan["signal_start"]:plan["signal_end"]]
        if len(r_sig) != plan["n_window"]:
            skipped += 1
            continue

        rqa, eps, rr = C.compute_rqa_domain(C.zscore(r_sig), C.M_TIME, C.TAU_TIME)
        spec = C.compute_spectral_wide(w_sig, SPS)

        # SNR cuando la ventana de ruido cabe entera; si no, NaN. Solo se usa
        # para describir, nunca como descriptor.
        snr_f = C.preprocess_segment(seg, SPS, band=C.SNR_BAND)
        noise = snr_f[plan["noise_start"]:plan["noise_end"]]
        snr = (C.compute_snr_db(noise, snr_f[plan["signal_start"]:plan["signal_end"]])
               if len(noise) == plan["noise_end"] - plan["noise_start"] else np.nan)

        rows.append(dict(event_id=f"{db}:{f}:{a:.1f}", volcano="deception",
                         db=db, year=year, label=label,
                         duration_s_original=dur,
                         duration_s_used=plan["n_window"] / SPS,
                         n_window=plan["n_window"], snr_db=snr,
                         eps_time=eps, rr_time=rr,
                         DET=rqa["DET"], Wentr=rqa["Wentr"],
                         SR_w=spec["SR_w"], FI_hi=spec["FI_hi"],
                         FI_lo=spec["FI_lo"], fpeak_w=spec["fpeak_w"],
                         E_vlp=spec["E_vlp"], E_lp=spec["E_lp"], E_vt=spec["E_vt"]))
    print(f"  {db} ({year}): {len(rows)} extraidos, {skipped} descartados")
    return pd.DataFrame(rows)


def main():
    if not BASE.exists():
        print(f"falta el dataset en {BASE}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for db, year in DBS:
        d = extract(db, year)
        if len(d):
            d.to_parquet(OUT / f"features_{db}.parquet", index=False)
            parts.append(d)
    if not parts:
        return
    t = pd.concat(parts, ignore_index=True)
    t.to_parquet(OUT / "features_deception.parquet", index=False)
    print()
    for (db, year), g in t.groupby(["db", "year"]):
        for lab, h in g.groupby("label"):
            print(f"  {year} {lab}  n={len(h):3d}  " +
                  "  ".join(f"{c} {h[c].median():+.2f}" for c in FIVE) +
                  f"   SNR {h.snr_db.median():.0f} dB ({h.snr_db.notna().sum()} med.)")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
