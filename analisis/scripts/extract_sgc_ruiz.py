"""
Extraccion piloto sobre los eventos del Nevado del Ruiz que envio el SGC.

Los archivos vienen en SUDS demultiplexado (DMX), un evento por archivo con
toda la red, separados en carpetas por clase: HIBRIDO, LP, TREMOR, VT. No
traen catalogo, asi que el onset y la duracion que definen la ventana del
estudio no existen: aqui se estiman, y eso es lo que hace a esta corrida un
piloto y no una medicion comparable con OVDAS.

    onset     STA/LTA recursivo (0,5 s / 5 s, umbral 3) sobre 1-10 Hz. Un
              disparo dentro del arranque del LTA se descarta: ahi el evento
              ya venia empezado y el onset no es tal.
    duracion  coda_duration.py con las constantes calibradas contra el
              analista de OVDAS (las mismas de VCSEIS).

Solo se usan las extensiones que empiezan con R; se asume que R es Ruiz y que
las demas letras (M, T, N, I, E) son otros volcanes, y hay que confirmarlo
con el SGC. TREMOR corresponde a la clase TR del catalogo OVDAS.

Uso:
    python scripts/extract_sgc_ruiz.py --station BISZ
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dist" / "descriptores_volcanicos"))
from coda_duration import coda_duration  # noqa: E402
from descriptores import WindowError, features_from_signal  # noqa: E402

DATA = ROOT / "data" / "external" / "sgc_ruiz"
OUT = ROOT / "data" / "processed" / "features_sgc"
CLASSES = {"LP": "LP", "VT": "VT", "HIBRIDO": "HB", "TREMOR": "TR"}
K, HOLD, SMOOTH = 1.2, 2.0, 1.0
STA_S, LTA_S, ON, OFF = 0.5, 5.0, 3.0, 1.0
NOISE_NEED_S = 12.0            # coda_duration mide el ruido en [-12, -2] s
CLIP_FRAC = 0.002              # fraccion de muestras en el tope que marca saturacion


def onset_s(x: np.ndarray, sr: float) -> float:
    y = x - x.mean()
    from obspy.signal.filter import bandpass
    y = bandpass(y, 1.0, 10.0, sr, corners=4, zerophase=True)
    c = recursive_sta_lta(y, int(STA_S * sr), int(LTA_S * sr))
    on = trigger_onset(c, ON, OFF)
    if not len(on):
        return float("nan")
    t = on[0][0] / sr
    return t if t > LTA_S + 0.5 else float("nan")


def clipped(x: np.ndarray) -> bool:
    top = np.abs(x).max()
    return top > 0 and np.mean(np.abs(x) >= 0.999 * top) > CLIP_FRAC


def one(path: Path, station: str) -> dict:
    st = read(str(path), format="DMX").select(station=station)
    if not st:
        return dict(motivo="sin estacion")
    tr = st[0]
    sr = tr.stats.sampling_rate
    x = tr.data.astype(np.float64)
    if abs(sr - 100.0) > 0.5:
        return dict(motivo=f"tasa {sr:.1f} Hz")
    # Deriva del digitalizador (100,009 Hz, 100,16 Hz): menos de 1 ms por
    # segundo, se toma como 100 Hz nominal.
    sr = 100.0
    t0 = onset_s(x, sr)
    base = dict(largo_s=len(x) / sr, onset_s=t0, saturado=clipped(x))
    if not np.isfinite(t0):
        return dict(base, motivo="sin onset")
    if t0 < NOISE_NEED_S:
        return dict(base, motivo="menos de 12 s antes del onset")
    dur = coda_duration(x, sr, t0, K, hold_s=HOLD, smooth_s=SMOOTH)
    if not np.isfinite(dur):
        return dict(base, motivo="sin fin de coda")
    try:
        f = features_from_signal(x, onset_s=t0, coda_end_s=t0 + dur, sps=sr)
    except WindowError as e:
        return dict(base, duracion_s=dur, motivo=f"ventana: {e}")
    return dict(base, duracion_s=dur, motivo="", **f.as_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="BISZ")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    rows = []
    for carpeta, label in CLASSES.items():
        fs = sorted(f for f in os.listdir(DATA / carpeta)
                    if f.split(".")[-1].startswith("R"))[:a.limit]
        for f in fs:
            try:
                r = one(DATA / carpeta / f, a.station)
            except Exception as e:                      # noqa: BLE001
                r = dict(motivo=f"lectura: {type(e).__name__}")
            rows.append(dict(file=f, label=label, station=a.station, **r))
        print(f"  {label}: {len(fs)} archivos")
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"features_sgc_ruiz_{a.station}.parquet"
    d.to_parquet(p, index=False)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
