"""
Extraccion sobre el corpus etiquetado de VOISS-Net (Fee et al., 2025).

David Fee envio la lista de las 272 611 ventanas etiquetadas
(labels_generalized_filenames.txt) y un script que las traduce a pedidos
FDSN (voissnet_dataset.py); ambos quedan sin modificar en
data/external/DavidFee. Este script toma de ahi las tablas de estaciones,
redes y canales, y hace tres cosas que el original no hace:

    muestreo  por volcan y clase, nunca mezclando volcanes. La etiqueta es de
              la ventana para toda la red (a la misma hora todas las
              estaciones tienen la misma clase), asi que la unidad es la hora
              de la ventana, no la traza: se sortean horas distintas y a cada
              una se le asigna la estacion de referencia del volcan, o la
              siguiente en la lista si esa no tiene la ventana etiquetada.
    margen    VOISS-Net etiqueta ventanas fijas de 120 s sin inicio ni fin de
              evento. Se descargan 60 s antes y 120 s despues para tener ruido
              previo y cierre de coda.
    ventana   onset por STA/LTA y duracion por coda_duration.py, con las
              mismas constantes del piloto del SGC (extract_sgc_ruiz.py). El
              onset se busca solo dentro de la ventana etiquetada.

Todo se remuestrea a 100 Hz antes de medir: 50 Hz por dos (exacto, como en
Decepcion), 40 Hz por cinco medios. A 40 Hz el Nyquist cae en 20 Hz, el borde
superior de BAND_HIGH; esas filas quedan marcadas en `sps_original`.

Etna (red IV) no esta en FDSN y queda fuera; Fee ofrecio enviarlo aparte.

Uso:
    python scripts/extract_voissnet.py --plan          # sortea y guarda el plan
    python scripts/extract_voissnet.py --ampliar VT LP # agrega todas las horas
                                                       # que falten de esas clases
    python scripts/extract_voissnet.py --descargar     # baja lo que falte
    python scripts/extract_voissnet.py --extraer       # descriptores
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "external" / "DavidFee"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dist" / "descriptores_volcanicos"))
sys.path.insert(0, str(DATA))
import voissnet_dataset as V  # noqa: E402
from extract_sgc_ruiz import (K, HOLD, SMOOTH, STA_S, LTA_S, ON, OFF,  # noqa: E402
                              NOISE_NEED_S, clipped)

WAVES = DATA / "waveforms"
PLAN = DATA / "plan_muestreo.csv"
OUT = ROOT / "data" / "processed" / "features_voissnet"

N_POR_CLASE = 500
SEMILLA = 41                   # la misma de Fee
PRE_S, POST_S, LARGO_S = 60.0, 120.0, 120.0
SPS = 100.0

# Estaciones por volcan, en orden de preferencia: la primera es la de
# referencia. En Popocatepetl ZACA va al final porque paso a 40 Hz.
VOLCANES = {
    "Semisopochnoi": ["CESW", "CEPE", "CERA", "CETU", "CERB", "CEAP"],
    "Pavlof": ["PV6A", "PN7A", "PVV", "PS4A", "PS1A"],
    "Veniaminof": ["VNSS", "VNWF", "VNHG", "VNFG", "VNKR", "VNNF", "VNSW"],
    "Redoubt": ["REF", "RDN", "RSO"],
    "Tanaga": ["TASE", "TANO"],
    "Kilauea": ["PUHI", "RIMD", "HAT", "UWE", "WRM"],
    "Popocatepetl": ["NEXP", "HSHT", "BIBI", "ZACA"],
}
# Equivalencia con el catalogo OVDAS. Explosion y Noise no tienen
# equivalente, pero se guardan: la fase descriptiva conserva todo.
CLASE = {
    "Earthquake": "VT", "Long Period": "LP",
    "Broadband Tremor": "TR", "Harmonic Tremor": "TR",
    "Monochromatic Tremor": "TR", "Explosion": "EX", "Noise": "RUIDO",
}


def etiquetas() -> pd.DataFrame:
    filas = []
    for line in V.INPUT_FILE.read_text().splitlines():
        parts = Path(line.strip()).stem.split("_")
        if len(parts) == 4 and parts[-1].isdigit():
            filas.append(parts)
    d = pd.DataFrame(filas, columns=["station", "start_time", "end_time", "class_id"])
    d["label"] = d.class_id.astype(int).map(V.CLASS_LABELS)
    return d


def plan() -> pd.DataFrame:
    d = etiquetas()
    rutas = V.build_station_routes()
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for volcan, orden in VOLCANES.items():
        dv = d[d.station.isin(orden)].copy()
        dv["prioridad"] = dv.station.map({s: i for i, s in enumerate(orden)})
        # una fila por hora: la estacion de mayor prioridad que la tenga
        dv = dv.sort_values("prioridad").drop_duplicates(["start_time", "label"])
        for label, g in dv.groupby("label"):
            g = g.sort_values("start_time")
            total, n = len(g), min(N_POR_CLASE, len(g))
            g = g.iloc[np.sort(rng.choice(len(g), n, replace=False))]
            for r in g.itertuples():
                red, cliente = rutas[r.station]
                filas.append(dict(
                    volcan=volcan, station=r.station, network=red,
                    fdsn_client=cliente,
                    channel=V.resolve_channel(r.station, r.start_time),
                    label=label, clase=CLASE[label], start_time=r.start_time,
                    horas_disponibles=total))
    p = pd.DataFrame(filas)
    p["archivo"] = (p.volcan + "/" + p.station + "_" + p.start_time + "_"
                    + p.clase + ".mseed")
    p.to_csv(PLAN, index=False)
    print(p.groupby(["volcan", "label"]).size().to_string())
    print(f"-> {PLAN} ({len(p)} ventanas)")
    return p


def ampliar(clases: list[str]):
    """Agrega al plan TODAS las horas disponibles de esas clases que aun no
    esten, sin tocar lo ya sorteado: el muestreo original queda como tanda 1."""
    p = pd.read_csv(PLAN, dtype=str)
    if "tanda" not in p.columns:
        p["tanda"] = "1"
    tanda = str(int(p.tanda.astype(int).max()) + 1)
    d = etiquetas()
    rutas = V.build_station_routes()
    filas = []
    for volcan, orden in VOLCANES.items():
        dv = d[d.station.isin(orden) & d.label.map(CLASE).isin(clases)].copy()
        dv["prioridad"] = dv.station.map({s: i for i, s in enumerate(orden)})
        dv = dv.sort_values("prioridad").drop_duplicates(["start_time", "label"])
        ya = set(zip(p[p.volcan == volcan].start_time, p[p.volcan == volcan].label))
        for r in dv.sort_values("start_time").itertuples():
            if (r.start_time, r.label) in ya:
                continue
            red, cliente = rutas[r.station]
            filas.append(dict(
                volcan=volcan, station=r.station, network=red, fdsn_client=cliente,
                channel=V.resolve_channel(r.station, r.start_time),
                label=r.label, clase=CLASE[r.label], start_time=r.start_time,
                horas_disponibles=str(int((dv.label == r.label).sum())),
                archivo=f"{volcan}/{r.station}_{r.start_time}_{CLASE[r.label]}.mseed",
                tanda=tanda))
    n = pd.DataFrame(filas)
    if n.empty:
        print("No hay horas nuevas.")
        return
    pd.concat([p, n], ignore_index=True).to_csv(PLAN, index=False)
    print(n.groupby(["volcan", "label"]).size().to_string())
    print(f"-> {PLAN} (+{len(n)} ventanas, tanda {tanda})")


def bajar(r, clientes) -> str:
    from obspy import UTCDateTime
    dest = WAVES / r.archivo
    if dest.exists():
        return "ya estaba"
    t0 = UTCDateTime.strptime(r.start_time, "%Y%m%d%H%M")
    try:
        st = clientes[r.fdsn_client].get_waveforms(
            r.network, r.station, V.DEFAULT_LOCATION, r.channel,
            t0 - PRE_S, t0 + LARGO_S + POST_S)
    except Exception as e:                              # noqa: BLE001
        return f"error: {type(e).__name__}"
    if not st:
        return "vacio"
    dest.parent.mkdir(parents=True, exist_ok=True)
    st.write(str(dest), format="MSEED")
    return "ok"


def descargar(workers: int):
    from obspy.clients.fdsn import Client
    p = pd.read_csv(PLAN, dtype=str)
    clientes = {k: Client(V.FDSN_ENDPOINT[k]) for k in p.fdsn_client.unique()}
    estados = {}
    with ThreadPoolExecutor(workers) as ex:
        fut = {ex.submit(bajar, r, clientes): r.archivo for r in p.itertuples()}
        for i, f in enumerate(as_completed(fut), 1):
            estados[fut[f]] = f.result()
            if i % 250 == 0:
                print(f"  {i}/{len(p)}", flush=True)
    s = pd.Series(estados)
    print(s.str.split(":").str[0].value_counts().to_string())
    s.rename("estado").to_csv(DATA / "descarga_estado.csv", index_label="archivo")


def a_100hz(tr) -> tuple[np.ndarray, float]:
    from scipy.signal import resample_poly
    sr = tr.stats.sampling_rate
    x = tr.data.astype(np.float64)
    nominal = round(sr)                 # 99,9993 Hz de RSO se toma como 100
    if nominal == SPS:
        return x, sr
    f = Fraction(int(SPS), nominal)
    return resample_poly(x - x.mean(), f.numerator, f.denominator), sr


def onset_s(x: np.ndarray) -> tuple[float, bool]:
    """Primer disparo dentro de la ventana etiquetada, y si ya habia uno
    activo al empezarla (evento que venia de antes)."""
    from obspy.signal.filter import bandpass
    from obspy.signal.trigger import recursive_sta_lta, trigger_onset
    y = bandpass(x - x.mean(), 1.0, 10.0, SPS, corners=4, zerophase=True)
    c = recursive_sta_lta(y, int(STA_S * SPS), int(LTA_S * SPS))
    on = np.asarray(trigger_onset(c, ON, OFF), float).reshape(-1, 2) / SPS
    previo = bool(np.any((on[:, 0] < PRE_S) & (on[:, 1] >= PRE_S)))
    dentro = on[(on[:, 0] >= PRE_S) & (on[:, 0] < PRE_S + LARGO_S)]
    return (float(dentro[0, 0]) if len(dentro) else float("nan")), previo


def una(ruta: Path) -> dict:
    from obspy import read
    from coda_duration import coda_duration
    from descriptores import WindowError, features_from_signal
    st = read(str(ruta)).merge(fill_value=None)
    tr = st[0]
    if np.ma.isMaskedArray(tr.data):
        return dict(motivo="huecos en la traza")
    x, sr0 = a_100hz(tr)
    t0, previo = onset_s(x)
    base = dict(sps_original=sr0, largo_s=len(x) / SPS, onset_s=t0,
                evento_previo=previo, saturado=clipped(tr.data))
    if not np.isfinite(t0):
        return dict(base, motivo="sin onset")
    if t0 < NOISE_NEED_S:
        return dict(base, motivo="menos de 12 s antes del onset")
    dur = coda_duration(x, SPS, t0, K, hold_s=HOLD, smooth_s=SMOOTH)
    if not np.isfinite(dur):
        return dict(base, motivo="sin fin de coda")
    try:
        f = features_from_signal(x, onset_s=t0, coda_end_s=t0 + dur, sps=SPS)
    except WindowError as e:
        return dict(base, duracion_s=dur, motivo=f"ventana: {e}")
    return dict(base, duracion_s=dur, motivo="", **f.as_dict())


def _fila(r: dict) -> dict:
    warnings.filterwarnings("ignore")
    ruta = WAVES / r["archivo"]
    if not ruta.exists():
        return dict(r, motivo="no descargado")
    try:
        return dict(r, **una(ruta))
    except Exception as e:                              # noqa: BLE001
        return dict(r, motivo=f"lectura: {type(e).__name__}")


def extraer(limite: int | None, volcanes: list[str] | None, workers: int):
    from concurrent.futures import ProcessPoolExecutor
    p = pd.read_csv(PLAN, dtype=str)
    OUT.mkdir(parents=True, exist_ok=True)
    for volcan, g in p.groupby("volcan"):
        if volcanes and volcan not in volcanes:
            continue
        g = g.head(limite) if limite else g
        with ProcessPoolExecutor(workers) as ex:
            filas = list(ex.map(_fila, g.to_dict("records"), chunksize=8))
        d = pd.DataFrame(filas)
        dest = OUT / f"features_voissnet_{volcan}.parquet"
        d.to_parquet(dest, index=False)
        ok = (d.motivo == "").groupby(d.label).agg(["sum", "size"])
        print(f"{volcan}:\n{ok.to_string()}\n-> {dest}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--ampliar", nargs="+", metavar="CLASE")
    ap.add_argument("--descargar", action="store_true")
    ap.add_argument("--extraer", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--volcanes", nargs="*", default=None)
    a = ap.parse_args()
    if a.plan:
        plan()
    if a.ampliar:
        ampliar(a.ampliar)
    if a.descargar:
        descargar(a.workers)
    if a.extraer:
        extraer(a.limit, a.volcanes, a.workers)


if __name__ == "__main__":
    main()
