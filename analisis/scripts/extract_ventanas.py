"""
Ventanas de 10 s (1000 muestras a 100 Hz, acondicionamiento actual: 0.5-10 Hz,
z-score) de los mismos eventos que audita sesgo_experto.py, para analisis
sobre la forma de onda o su espectro (agrupamiento_fft.py).

  - OVDAS (Hudson, Copahue, Villarrica, Laguna del Maule, Nevados de Chillan):
    la misma submuestra aleatoria de hasta N_SUB eventos que sesgo_experto.py
    (misma semilla, mismas clases >= MIN_CLASE, mismo tope por clase), bajada
    del Winston con make_fig_coda_window.segment y acondicionada con
    rqa_core.preprocess_segment, como el pipeline principal.
  - Externos (todos los datasets con eventos discretos del corte manual):
    todos los eventos validos de features16_fija, leidos como extract_picks_16
    (cortar_trazas.cargar + a_100hz) desde el onset manual.

Salida: data/processed/ventanas/ventanas_<tag>.npz con X (n x 1000), item_id,
clase, sistema, snr_db. Cada dataset por separado.

Uso:
    nice -n 10 python scripts/extract_ventanas.py [--sistemas Hudson Alaska ...] [--workers 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import rqa_core as C  # noqa: E402
from ranking_transferibilidad import OVDAS  # noqa: E402
from sesgo_experto import MIN_CLASE, N_SUB, SEMILLA, TOPE_CLASE, cargar  # noqa: E402

OUT = ROOT / "data" / "processed" / "ventanas"
EXTERNOS = ["Alaska", "Cascade", "Hawái", "Norte de California", "Ruiz-Olleta", "Ruiz-Bis",
            "Decepcion09", "Decepcion95", "Kilauea", "Redoubt", "Tanaga", "Veniaminof",
            "Semisopochnoi", "Pavlof"]
IDX = C.padded_window_indices(C.SPS)


def tag(sistema):
    return sistema.lower().replace(" ", "_")


def ventana(raw):
    """Segmento crudo alineado a onset - PADDED_PRE_S -> ventana de senal z-scoreada."""
    w = C.preprocess_segment(raw, C.SPS)[IDX["signal_start"]:IDX["signal_end"]]
    if len(w) != C.WINDOW_SAMPLES:
        raise ValueError("ventana incompleta")
    return C.zscore(w)


def submuestra_ovdas(sistema):
    """Replica la submuestra de sesgo_experto.py (mismos eventos)."""
    d = cargar(sistema)
    d = d[d.clase.notna()]
    n = d.clase.value_counts()
    d = d[d.clase.isin(n[n >= MIN_CLASE].index)]
    if (n > TOPE_CLASE).any():
        d = pd.concat([g.sample(min(len(g), TOPE_CLASE), random_state=SEMILLA)
                       for _, g in d.groupby("clase")])
    d = d.reset_index(drop=True)
    rng = np.random.default_rng(SEMILLA)
    isub = np.sort(rng.choice(len(d), size=min(N_SUB, len(d)), replace=False))
    return d.iloc[isub].reset_index(drop=True)


def una_ovdas(args):
    clave, origin_time = args
    warnings.filterwarnings("ignore")
    from obspy import UTCDateTime
    from make_fig_coda_window import segment
    cfg = json.load(open(ROOT / "config" / "volcanoes.json", encoding="utf-8"))[clave]
    try:
        raw = segment(cfg, UTCDateTime(pd.Timestamp(origin_time).isoformat()),
                      C.PADDED_PRE_S, C.PADDED_POST_S, C.SPS)
        return ventana(raw)
    except Exception:                                   # noqa: BLE001
        return None


def una_externo(args):
    item, estacion, onset = args
    warnings.filterwarnings("ignore")
    import cortar_trazas as CT
    from extract_picks import a_100hz
    try:
        t = CT.cargar(dict(item, estacion=estacion))
        x = a_100hz(np.nan_to_num(t["x"]), t["sr"])
        i0 = int(round((onset - C.PADDED_PRE_S) * C.SPS))
        n = IDX["total_samples"]
        if i0 < 0 or i0 + n > len(x):
            return None
        return ventana(x[i0:i0 + n])
    except Exception:                                   # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sistemas", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ovdas = {nombre: clave for clave, (nombre, _) in OVDAS.items()}
    sistemas = a.sistemas or (list(ovdas) + EXTERNOS)
    f = pd.read_parquet(ROOT / "data/processed/features_manual/features16_fija.parquet")
    f = f[(f.motivo == "") & ~f.duplicado.fillna(False).astype(bool)].copy()
    dec = f.sistema == "Decepcion"
    f.loc[dec, "sistema"] = "Decepcion" + f.loc[dec, "item_id"].str.extract(r"dec\.(\d\d)")[0]
    import cortar_trazas as CT
    cola = pd.read_csv(CT.COLA, dtype=str, keep_default_na=False).set_index("item_id")
    picks = (pd.read_csv(CT.PICKS, dtype=str, keep_default_na=False)
             .drop_duplicates("item_id", keep="last").set_index("item_id"))
    for sis in sistemas:
        ruta = OUT / f"ventanas_{tag(sis)}.npz"
        if ruta.exists():
            print(f"{sis}: ya existe", flush=True)
            continue
        t0 = time.time()
        if sis in ovdas:
            d = submuestra_ovdas(sis)
            trabajos = [(ovdas[sis], r.origin_time) for r in d.itertuples()]
            fn = una_ovdas
        else:
            d = f[f.sistema == sis].reset_index(drop=True)
            trabajos = [(cola.loc[r.item_id].to_dict() | {"item_id": r.item_id},
                         picks.loc[r.item_id, "estacion"], float(r.onset)) for r in d.itertuples()]
            fn = una_externo
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(fn, trabajos, chunksize=20))
        ok = np.array([r is not None for r in res])
        X = np.vstack([r for r in res if r is not None]).astype(np.float32)
        np.savez_compressed(ruta, X=X, item_id=d.item_id.to_numpy()[ok].astype(str),
                            clase=d.clase.to_numpy()[ok].astype(str), sistema=sis,
                            snr_db=d.snr_db.to_numpy(float)[ok])
        print(f"{sis}: {ok.sum()}/{len(d)} ventanas [{(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
