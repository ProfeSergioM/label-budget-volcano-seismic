"""
Extrae los descriptores del corpus VCSEIS con la ventana de coda estimada.

Dataset: Zhong, Y., y Tan, Y. J. (2024), Deep-learning-based phase picking for
volcano-tectonic and long-period earthquakes, GRL 51, e2024GL108438, CC-BY
4.0. Formato SeisBench, servido por HTTPS plano en

    https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets/vcseis/

VCSEIS entrega arribos P y S pero no fin de coda, de modo que la duracion que
define la ventana del estudio no existe ahi. Se estima con coda_duration.py,
cuyo acuerdo con la duracion que escribe el analista esta medido sobre el
corpus OVDAS.

Todo lo demas queda identico: 100 Hz nativo, misma banda, mismo orden de
filtro, mismo m y tau. Se usa la componente vertical, primera del orden ZNE.
No hace falta restituir la respuesta, que el conjunto no trae: los
descriptores son invariantes a la escala de amplitud.

Uso:
    python scripts/extract_vcseis.py --chunks cascade_lp cascade_vt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import rqa_core as C  # noqa: E402
from coda_duration import coda_duration  # noqa: E402

BASE = ROOT / "data" / "external" / "vcseis"
OUT = ROOT / "data" / "processed" / "features_vcseis"
REGION = {"cascade_lp": "Cascade", "cascade_vt": "Cascade",
          "ncedc_lp": "Norte de California", "ncedc_vt": "Norte de California",
          "ak_lp": "Alaska", "ak_rg": "Alaska",
          "hw12t21_lp": "Hawái", "hw12t21_rg": "Hawái"}
# Constantes del estimador, calibradas contra la duracion del analista de OVDAS.
K, HOLD, SMOOTH = 1.2, 2.0, 1.0
NEED_PRE_S = C.DUR_PADDED_PRE_S      # 10 s antes del arribo


def parse_trace_name(name: str):
    """'bucket0$3,:3,:4334' -> ('bucket0', 3)."""
    b, rest = name.split("$")
    return b, int(rest.split(",")[0])


def extract_chunk(chunk: str, limit: int | None = None,
                  sample: int | None = None, data_dir: Path | None = None,
                  seed: int = 42) -> pd.DataFrame:
    import h5py
    data_dir = data_dir or BASE
    meta = pd.read_csv(BASE / f"metadata_{chunk}.csv", low_memory=False)
    num = lambda c: pd.to_numeric(meta[c], errors="coerce")  # noqa: E731
    meta["p_sample"] = num("trace_p_arrival_sample")
    meta["sps"] = num("trace_sampling_rate_hz")
    meta = meta[(meta.sps == 100.0) & meta.p_sample.notna()]
    meta = meta[meta.p_sample >= NEED_PRE_S * 100.0]
    if sample and len(meta) > sample:
        # Muestra aleatoria y no las primeras filas: los trozos vienen
        # ordenados por evento, y encabezarlos tomaria un solo periodo.
        meta = meta.sample(sample, random_state=seed)
    if limit:
        meta = meta.head(limit)

    # La etiqueta sale de source_type y NO del nombre del trozo: los trozos
    # "_rg" traen, ademas de eventos locales, otros tipos que el catalogo
    # distingue -- en Alaska un 6,7 % son ot, re, tr o av -- y meterlos como VT
    # contamina la clase.
    LOCAL = {"le", "local_event", "e", "vt"}
    if chunk.endswith("_lp"):
        meta = meta[meta.source_type == "lp"]
        label = "LP"
    else:
        meta = meta[meta.source_type.isin(LOCAL)]
        label = "VT"
    rows, sin_ventana = [], 0
    with h5py.File(data_dir / f"waveforms_{chunk}.hdf5", "r") as f:
        g = f["data"]
        for r in meta.itertuples():
            try:
                b, i = parse_trace_name(r.trace_name)
                x = np.asarray(g[b][i, 0, :], dtype=np.float64)   # Z de ZNE
            except (KeyError, IndexError, ValueError):
                continue
            pre_s = float(r.p_sample) / 100.0
            dur = coda_duration(x, 100.0, pre_s, K, hold_s=HOLD, smooth_s=SMOOTH)
            if not np.isfinite(dur) or dur < C.MIN_DURATION_S:
                sin_ventana += 1
                continue
            plan = C.duration_window_plan(dur, 100.0)
            i0 = int(round((pre_s - plan["pre_s"]) * 100.0))
            if i0 < 0 or i0 + plan["total_samples"] > len(x):
                sin_ventana += 1
                continue
            seg = x[i0:i0 + plan["total_samples"]]
            wide = C.preprocess_segment(seg, 100.0, band=C.WIDE_FILTER_BAND)
            rq = C.preprocess_segment(seg, 100.0, band=C.DURATION_FILTER_BAND)
            w_sig = wide[plan["signal_start"]:plan["signal_end"]]
            r_sig = rq[plan["signal_start"]:plan["signal_end"]]
            if len(r_sig) != plan["n_window"]:
                sin_ventana += 1
                continue
            rqa, eps, rr = C.compute_rqa_domain(C.zscore(r_sig), C.M_TIME, C.TAU_TIME)
            sp = C.compute_spectral_wide(w_sig, 100.0)
            snr_f = C.preprocess_segment(seg, 100.0, band=C.SNR_BAND)
            noise = snr_f[plan["noise_start"]:plan["noise_end"]]
            snr = (C.compute_snr_db(noise, snr_f[plan["signal_start"]:plan["signal_end"]])
                   if len(noise) == plan["noise_end"] - plan["noise_start"] else np.nan)
            rows.append(dict(
                event_id=f"{chunk}:{r.trace_name}", volcano="vcseis",
                chunk=chunk, region=REGION.get(chunk, chunk), label=label,
                source_id=r.source_id, station=r.station_code,
                duration_s_original=dur, duration_s_used=plan["n_window"] / 100.0,
                n_window=plan["n_window"], snr_db=snr, eps_time=eps, rr_time=rr,
                fi_autores=pd.to_numeric(pd.Series([r.trace_frequency_index]),
                                         errors="coerce").iloc[0],
                DET=rqa["DET"], Wentr=rqa["Wentr"], SR_w=sp["SR_w"],
                FI_hi=sp["FI_hi"], FI_lo=sp["FI_lo"], fpeak_w=sp["fpeak_w"],
                E_vlp=sp["E_vlp"], E_lp=sp["E_lp"], E_vt=sp["E_vt"]))
    print(f"  {chunk}: {len(rows)} extraidos, {sin_ventana} sin ventana valida "
          f"(de {len(meta)} trazas con 10 s previos)")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+",
                    default=["cascade_lp", "cascade_vt", "ncedc_lp", "ncedc_vt"])
    ap.add_argument("--limit", type=int, default=None,
                    help="tope de trazas por trozo, para probar")
    ap.add_argument("--sample", type=int, default=None,
                    help="muestra aleatoria de trazas por trozo")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="donde estan los .hdf5, si no es data/external/vcseis")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for c in a.chunks:
        dd = a.data_dir or BASE
        if not (dd / f"waveforms_{c}.hdf5").exists():
            print(f"  (falta waveforms_{c}.hdf5 en {dd})")
            continue
        d = extract_chunk(c, a.limit, a.sample, dd)
        if len(d):
            d.to_parquet(OUT / f"features_{c}.parquet", index=False)
            parts.append(d)
    # El consolidado se rehace desde todos los trozos ya extraidos, no solo
    # los de esta corrida: los grandes se procesan de a uno y se borran.
    todos = sorted(p for p in OUT.glob("features_*.parquet")
                   if p.name != "features_vcseis.parquet")
    if todos:
        t = pd.concat([pd.read_parquet(p) for p in todos], ignore_index=True)
        t.to_parquet(OUT / "features_vcseis.parquet", index=False)
        print()
        for (reg, lab), g in t.groupby(["region", "label"]):
            print(f"  {reg:22s} {lab}  n={len(g):5d}  dur med {g.duration_s_original.median():5.1f} s  "
                  + "  ".join(f"{c} {g[c].median():+.2f}"
                              for c in ("SR_w", "FI_hi", "DET", "Wentr")))
        print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
