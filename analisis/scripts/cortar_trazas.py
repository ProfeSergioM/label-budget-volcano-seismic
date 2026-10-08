"""
Corte manual de trazas de los corpus externos: ventana de ruido y ventana del
evento, marcadas a mano, de a poco.

Ninguno de los corpus externos trae las dos ventanas que el estudio necesita.
VOISS-Net etiqueta ventanas fijas de 120 s, el SGC entrego archivos sin
catalogo, VCSEIS trae arribos P y S pero no fin de coda, y Decepcion trae
segmentos del analista pero sin ventana de ruido. Hasta ahora se estimaban con
STA/LTA y coda_duration.py; esta herramienta deja que el analista las fije.

Hay dos pasos:

    cola     arma la lista de trazas a revisar, una vez, con orden fijo:
             dentro de cada corpus se alterna entre estratos (sistema x clase)
             para que, se pare donde se pare, todos los estratos lleven un
             avance parecido.
    servir   levanta una pagina local para marcar. Retoma en la primera traza
             sin marcar; cada guardado se agrega al archivo de picks, asi que
             no hay nada que perder al cerrar.

La pagina propone el corte automatico (el mismo de los extractores) como punto
de partida, y muestra la referencia que trae cada corpus: la ventana de 120 s
de VOISS-Net, los arribos P y S de VCSEIS, el segmento del analista de
Decepcion.

Teclas (con el mouse sobre la traza):
    1 / 2    inicio / fin de la ventana de ruido, donde esta el cursor
    3 / 4    onset / fin del evento
    c        reclasifica (clase_revisada; vacio = la del corpus)
    f        lleva el onset a la linea verde del AIC (afinar_onset.py), que
             se recalcula en [onset - 3, onset + 3] s cada vez que se mueve
    Enter    guardar como bueno, si se movio alguna marca
    a        guardar como bueno aceptando el corte propuesto sin moverlo
    u        guardar como dudoso           x   descartar (pide motivo)
    n        siguiente sin guardar
    p        anterior                      v   filtrada 0,5-10 Hz / cruda
    z        zoom al evento                0   ver la traza completa

En Decepcion cada archivo trae varios segmentos etiquetados y cada uno es una
traza de la cola: la pagina muestra el propio ("ESTE") y los otros ("otro"),
y avisa si el onset cae fuera del propio.

El ruido acompana al onset (termina 2 s antes y dura 10 s) mientras no se
marque a mano con 1 o 2. En VOISS-Net la pagina avisa si el onset cae fuera
de la ventana etiquetada: ese evento no es el que el corpus clasifico.

Los estratos de EN_MUESTRA (Tanaga VT) muestran solo lo ya marcado; lo que
falta vuelve con --incluir-todo o pidiendo el sistema con --sistema.
El tremor y el ruido de VOISS-Net y del SGC no se cortan a mano
(EXCLUIDAS_POR_AHORA): no tienen onset, y su ventana fija se centra en el
tramo etiquetado (extract_continuas.py). Siguen en la cola y vuelven con
--incluir-todo, o pidiendolas con --clase. Las explosiones (EX) y las clases
de VCSEIS fuera de LP/VT (VCSEIS_OTRAS), que traen llegada P, si se cortan.

Con --repasar otra_estacion la pagina muestra en Bis (BISZ, si no BIS1) los
descartes "sin evento visible" del Ruiz; al guardar, la marca registra Bis.

Con --repasar tornillos la pagina recorre los LP ya marcados empezando por
los de coda mas lenta y monocromatica (Q de coda x concentracion,
scripts/indice_tornillo.py), para reclasificar con c.

Con --repasar la cola se reduce a las trazas ya marcadas que cumplen un
criterio (fuera_de_ventana, igual_al_auto_sin_registro, ruido_recalculado),
para volver sobre ellas.

La columna `origen` distingue un corte editado de uno aceptado tal como se
propuso ("aceptado"). Un evento de menos de 5 s no se guarda como bueno: el
paquete no le calcula descriptores, asi que la pagina ofrece descartarlo.

Los tiempos se guardan en segundos desde el inicio del tramo mostrado, junto
con la hora UTC de ese inicio cuando el corpus la trae. Los picks se versionan
en experiments/picks/ (data/ no se versiona).

Uso:
    python scripts/cortar_trazas.py cola --vcseis-n 200
    python scripts/cortar_trazas.py cola --ampliar      # suma lo nuevo al final
    python scripts/cortar_trazas.py servir --dataset voissnet --analista SM
    # en TU equipo (PowerShell), nunca en el servidor: si el tunel se abre en
    # el servidor, escucha en su propio 8765 y arma un bucle que cuelga la pagina
    #   ssh -L 8765:127.0.0.1:8765 smorales@<servidor>
    # y abrir http://localhost:8765
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import warnings
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

EXT = ROOT / "data" / "external"
PICKS_DIR = ROOT / "experiments" / "picks"
COLA = PICKS_DIR / "cola.csv"
PICKS = PICKS_DIR / "picks_manuales.csv"

# Referencia del corte automatico: las mismas constantes de los extractores.
K, HOLD, SMOOTH = 1.2, 2.0, 1.0
STA_S, LTA_S, ON, OFF = 0.5, 5.0, 3.0, 1.0
RUIDO_A, RUIDO_B = 12.0, 2.0        # ruido propuesto: [onset-12, onset-2] s
BANDA_VISTA = (0.5, 10.0)           # la banda del SNR
PRE_VOISS, POST_VOISS = 60.0, 120.0  # margen con que extract_voissnet baja
AUTO_DUR_MIN = 5.0                   # s, duracion minima del corte propuesto
MARGEN_FIN = 3.0                     # s que el paquete exige tras el fin (EDGE_PAD_S)

# Clases continuas que la pagina no muestra (en VOISS-Net y SGC); siguen en
# la cola. EX entra al corte desde 2026-10-06.
EXCLUIDAS_POR_AHORA = ("TR", "RUIDO")
CONTINUAS_EN = ("voissnet", "sgc")
# Estratos que quedan en muestra: la pagina oculta lo que falte por marcar.
# Tanaga VT ya es estable con ~1000 (make_fig_estabilidad_manual.py).
EN_MUESTRA = {("voissnet", "Tanaga", "VT")}

# Estacion de referencia del Ruiz segun el SGC (J. Londono, 2026-09-29):
# Olleta, y si falla, Bis. En los archivos de tres componentes la vertical es
# OLLZ (OLLE es la este-oeste: no correlaciona con OLLZ). En los de 2002 solo
# existe OLLE, estacion de una componente de la red antigua; se asume
# vertical, por confirmar con el SGC.
SGC_ESTACIONES = ("OLLZ", "OLLE", "BISZ", "BIS1")
SGC_CLASES = {"LP": "LP", "VT": "VT", "HIBRIDO": "HB", "TREMOR": "TR"}
VCSEIS_REGION = {"cascade": "Cascade", "ncedc": "Norte de California",
                 "ak": "Alaska", "hw12t21": "Hawái"}
VCSEIS_LOCAL = {"le", "local_event", "e", "vt"}
# Las demas clases de los trozos regionales (Alaska, Hawai): regionales,
# otros, tremor, avalanchas, explosiones. Entran a la cola desde 2026-10-06.
VCSEIS_OTRAS = {"re": "RE", "ot": "OT", "tr": "TR", "av": "AV", "ex": "EX"}
DECEPCION = EXT / "deception" / "VSR_DBs_@cortes-GRL.18"
DECEPCION_DBS = ("dec.95M.16c", "dec.09U.4c")

CAMPOS = ["item_id", "dataset", "sistema", "clase", "estacion", "t0_utc", "sps",
          "ruido_ini", "ruido_fin", "onset", "fin", "estado", "motivo",
          "auto_onset", "auto_fin", "origen", "analista", "guardado_utc",
          "clase_revisada"]


def migrar_picks():
    """Agrega `origen` a un archivo de picks anterior al ajuste. Esas marcas
    se guardaron sin registrar si se aceptaron a proposito, asi que solo se
    puede decir si quedaron iguales al corte automatico."""
    if not PICKS.exists():
        return
    p = pd.read_csv(PICKS, dtype=str, keep_default_na=False)
    if "origen" in p.columns:
        if "clase_revisada" not in p.columns:
            p["clase_revisada"] = ""
            p[CAMPOS].to_csv(PICKS, index=False)
            print(f"picks migrados: columna clase_revisada agregada a {len(p)} filas")
        return
    p["clase_revisada"] = ""
    num = lambda c: pd.to_numeric(p[c], errors="coerce")  # noqa: E731
    igual = ((num("onset") - num("auto_onset")).abs() < 0.01) & \
            ((num("fin") - num("auto_fin")).abs() < 0.01)
    p["origen"] = np.where(p.estado == "descartado", "",
                           np.where(igual, "igual_al_auto_sin_registro", "editado"))
    p[CAMPOS].to_csv(PICKS, index=False)
    print(f"picks migrados: columna origen agregada a {len(p)} filas")


# ---------------------------------------------------------------- la cola --

def _voissnet() -> list[dict]:
    plan = EXT / "DavidFee" / "plan_muestreo.csv"
    if not plan.exists():
        print("  voissnet: falta plan_muestreo.csv (extract_voissnet.py --plan)")
        return []
    p = pd.read_csv(plan, dtype=str)
    return [dict(item_id=f"voissnet:{r.archivo}", dataset="voissnet",
                 sistema=r.volcan, clase=r.clase, estacion=r.station,
                 fuente=r.archivo,
                 extra=json.dumps(dict(label=r.label, inicio=r.start_time)))
            for r in p.itertuples()]


def estacion_sgc(ruta: Path) -> str:
    """La primera de SGC_ESTACIONES que trae el archivo; OLLE solo cuenta si
    no hay OLLZ (si la hay, OLLE es una horizontal)."""
    from obspy import read
    try:
        st = read(str(ruta), format="DMX", headonly=True)
    except Exception:                                   # noqa: BLE001
        return ""
    hay = {tr.stats.station for tr in st}
    return next((e for e in SGC_ESTACIONES if e in hay), "")


SGC_ALTERNAS = ("BISZ", "BIS1")     # Bis, si Olleta presenta problemas


def estacion_sgc_alterna(ruta: Path) -> str:
    """La estacion de respaldo (Bis) que trae el archivo, o "" si no hay."""
    from obspy import read
    try:
        st = read(str(ruta), format="DMX", headonly=True)
    except Exception:                                   # noqa: BLE001
        return ""
    hay = {tr.stats.station for tr in st}
    return next((e for e in SGC_ALTERNAS if e in hay), "")


def _sgc() -> list[dict]:
    base = EXT / "sgc_ruiz"
    items = []
    for carpeta, clase in SGC_CLASES.items():
        if not (base / carpeta).exists():
            continue
        for f in sorted((base / carpeta).iterdir()):
            # R = Ruiz, por confirmar con el SGC (ver extract_sgc_ruiz.py)
            if f.suffix[1:2] == "R":
                items.append(dict(item_id=f"sgc:{carpeta}/{f.name}", dataset="sgc",
                                  sistema="Ruiz", clase=clase,
                                  estacion=estacion_sgc(f),
                                  fuente=f"{carpeta}/{f.name}", extra="{}"))
    return items


def muestra_vcseis(chunk: str, n: int, semilla: int) -> pd.DataFrame:
    """Las trazas de un trozo que entran a la cola. fetch_vcseis_muestra.py
    usa esta misma funcion para bajar solo esas trazas."""
    m = pd.read_csv(EXT / "vcseis" / f"metadata_{chunk}.csv", low_memory=False)
    m = m[pd.to_numeric(m.trace_sampling_rate_hz, errors="coerce") == 100]
    m = m[pd.to_numeric(m.trace_p_arrival_sample, errors="coerce").notna()]
    if chunk.endswith("_lp"):
        m = m[m.source_type == "lp"].assign(clase="LP")
    else:
        m = m[m.source_type.isin(VCSEIS_LOCAL)].assign(clase="VT")
    return m.sample(min(n, len(m)), random_state=semilla)


def muestra_vcseis_otras(chunk: str, n: int, semilla: int) -> pd.DataFrame:
    """Hasta n trazas por clase de VCSEIS_OTRAS en un trozo regional."""
    if chunk.endswith("_lp"):
        return pd.DataFrame()
    m = pd.read_csv(EXT / "vcseis" / f"metadata_{chunk}.csv", low_memory=False)
    m = m[pd.to_numeric(m.trace_sampling_rate_hz, errors="coerce") == 100]
    m = m[pd.to_numeric(m.trace_p_arrival_sample, errors="coerce").notna()]
    m = m[m.source_type.isin(VCSEIS_OTRAS)].assign(clase=lambda d: d.source_type.map(VCSEIS_OTRAS))
    return pd.concat([g.sample(min(n, len(g)), random_state=semilla)
                      for _, g in m.groupby("clase")]) if len(m) else m


def _vcseis_h5(chunk: str) -> tuple[Path, bool]:
    """El hdf5 completo si esta; si no, la muestra bajada por rangos."""
    full = EXT / "vcseis" / f"waveforms_{chunk}.hdf5"
    return (full, True) if full.exists() else (EXT / "vcseis" / f"muestra_{chunk}.hdf5", False)


def _vcseis(n: int, semilla: int) -> list[dict]:
    items = []
    for meta in sorted((EXT / "vcseis").glob("metadata_*.csv")):
        chunk = meta.stem.removeprefix("metadata_")
        if not _vcseis_h5(chunk)[0].exists():
            print(f"  vcseis: {chunk} sin formas de onda en el servidor, queda fuera "
                  "(fetch_vcseis_muestra.py)")
            continue
        region = VCSEIS_REGION[chunk.rsplit("_", 1)[0]]
        m = pd.concat([muestra_vcseis(chunk, n, semilla),
                       muestra_vcseis_otras(chunk, 500, semilla)])
        for r in m.itertuples():
            items.append(dict(
                item_id=f"vcseis:{chunk}:{r.trace_name}", dataset="vcseis",
                sistema=region, clase=r.clase, estacion=r.station_code,
                fuente=chunk,
                extra=json.dumps(dict(
                    trace=r.trace_name, t0=r.trace_start_time,
                    p=float(r.trace_p_arrival_sample),
                    s=float(pd.to_numeric(r.trace_s_arrival_sample, errors="coerce"))))))
    return items


def _decepcion() -> list[dict]:
    from extract_deception import parse_mlf
    items = []
    for db in DECEPCION_DBS:
        mlf = DECEPCION / f"{db}.mlf"
        if not mlf.exists():
            continue
        for f, a, b, label in parse_mlf(mlf):
            if label in ("LP", "VT") and (DECEPCION / db / "raw" / f).exists():
                items.append(dict(
                    item_id=f"decepcion:{db}/{f}:{a:.4f}-{b:.4f}",
                    dataset="decepcion", sistema="Decepcion", clase=label,
                    estacion=db, fuente=f"{db}/raw/{f}",
                    extra=json.dumps(dict(a=a, b=b))))
    return items


def _intercalar(d: pd.DataFrame, rng) -> pd.DataFrame:
    """Orden al azar dentro de cada estrato y turnos entre estratos."""
    d = d.assign(_r=rng.random(len(d))).sort_values("_r")
    d["_turno"] = d.groupby(["sistema", "clase"]).cumcount()
    d["_e"] = d.groupby(["sistema", "clase"]).ngroup()
    return d.sort_values(["_turno", "_e"]).drop(columns=["_r", "_turno", "_e"])


def ampliar_cola(vcseis_n: int, semilla: int):
    """Agrega al final de la cola lo que las fuentes tengan y la cola no, sin
    mover nada de lo que ya esta (ni el orden ni lo marcado)."""
    c = pd.read_csv(COLA, dtype=str, keep_default_na=False)
    rng = np.random.default_rng(semilla + len(c))
    partes = []
    for items in (_voissnet(), _sgc(), _vcseis(vcseis_n, semilla), _decepcion()):
        d = pd.DataFrame(items)
        if len(d):
            d = d[~d.item_id.isin(c.item_id)]
        if len(d):
            partes.append(_intercalar(d, rng))
    if not partes:
        print("La cola ya tiene todo lo disponible.")
        return
    n = pd.concat(partes, ignore_index=True)
    n.insert(0, "orden", range(len(c), len(c) + len(n)))
    pd.concat([c, n.astype(str)], ignore_index=True).to_csv(COLA, index=False)
    print(n.groupby(["dataset", "sistema", "clase"]).size().to_string())
    print(f"-> {COLA} (+{len(n)} trazas, {len(c) + len(n)} en total)")


def armar_cola(vcseis_n: int, semilla: int):
    if COLA.exists():
        sys.exit(f"{COLA} ya existe: la cola se arma una vez. Borrala a mano "
                 "si de verdad quieres rehacerla.")
    rng = np.random.default_rng(semilla)
    partes = []
    for items in (_voissnet(), _sgc(), _vcseis(vcseis_n, semilla), _decepcion()):
        if not items:
            continue
        partes.append(_intercalar(pd.DataFrame(items), rng))
    c = pd.concat(partes, ignore_index=True)
    c.insert(0, "orden", range(len(c)))
    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    c.to_csv(COLA, index=False)
    print(c.groupby(["dataset", "sistema", "clase"]).size().to_string())
    print(f"-> {COLA} ({len(c)} trazas)")


# ------------------------------------------------------------- las trazas --

def _auto_onset(x, sr, t_min, t_max):
    from obspy.signal.filter import bandpass
    from obspy.signal.trigger import recursive_sta_lta, trigger_onset
    y = bandpass(x - x.mean(), 1.0, min(10.0, 0.45 * sr), sr, corners=4,
                 zerophase=True)
    c = recursive_sta_lta(y, int(STA_S * sr), int(LTA_S * sr))
    on = np.asarray(trigger_onset(c, ON, OFF), float).reshape(-1, 2) / sr
    on = on[(on[:, 0] >= max(t_min, LTA_S + 0.5)) & (on[:, 0] < t_max)]
    return float(on[0, 0]) if len(on) else None


def _auto_fin(x, sr, onset):
    from coda_duration import coda_duration
    if onset is None:
        return None
    d = coda_duration(x, sr, onset, K, hold_s=HOLD, smooth_s=SMOOTH)
    # el estimador corta demasiado pronto en eventos debiles o emergentes:
    # la propuesta nunca baja de AUTO_DUR_MIN, sin pasarse del final
    d = max(float(d), AUTO_DUR_MIN) if np.isfinite(d) else AUTO_DUR_MIN
    return min(onset + d, len(x) / sr - MARGEN_FIN)


_SEGS = None


def _segmentos_decepcion() -> dict:
    global _SEGS
    if _SEGS is None:
        from extract_deception import parse_mlf
        _SEGS = {}
        for db in DECEPCION_DBS:
            if (DECEPCION / f"{db}.mlf").exists():
                for f, a, b, lab in parse_mlf(DECEPCION / f"{db}.mlf"):
                    _SEGS.setdefault((db, f), []).append((a, b, lab))
    return _SEGS


def cargar(r: dict) -> dict:
    """Traza, referencias del corpus y corte automatico de un item de la cola."""
    extra = json.loads(r["extra"])
    refs = []                     # (texto, inicio, fin o None)
    etiqueta = None               # ventana en que el corpus afirma el evento
    t0_utc = ""
    ds = r["dataset"]
    if ds == "voissnet":
        from obspy import read, UTCDateTime
        st = read(str(EXT / "DavidFee" / "waveforms" / r["fuente"]))
        huecos = len(st) > 1
        st.merge(fill_value=0)
        tr = st[0]
        sr, x = tr.stats.sampling_rate, tr.data.astype(np.float64)
        t0_utc = str(tr.stats.starttime)
        a = UTCDateTime.strptime(extra["inicio"], "%Y%m%d%H%M") - tr.stats.starttime
        refs.append((f"VOISS-Net: {extra['label']}", a, a + 120.0))
        etiqueta = (a, a + 120.0)
        if huecos:
            refs.append(("traza con huecos, rellenados con ceros", 0.0, None))
        onset = _auto_onset(x, sr, a, a + 120.0)
        fin = _auto_fin(x, sr, onset)
    elif ds == "sgc":
        from obspy import read
        st = read(str(EXT / "sgc_ruiz" / r["fuente"]), format="DMX")
        st = st.select(station=r["estacion"])
        if not st:
            raise ValueError(f"el archivo no trae la estacion {r['estacion']}")
        tr = st[0]
        sr, x = tr.stats.sampling_rate, tr.data.astype(np.float64)
        t0_utc = str(tr.stats.starttime)
        onset = _auto_onset(x, sr, 0.0, len(x) / sr)
        fin = _auto_fin(x, sr, onset)
    elif ds == "vcseis":
        import h5py
        from extract_vcseis import parse_trace_name
        b, i = parse_trace_name(extra["trace"])
        h5, completo = _vcseis_h5(r["fuente"])
        with h5py.File(h5, "r") as f:
            x = np.asarray(f["data"][b][i, 0, :] if completo
                           else f[f"data/{b}/{i}"][0, :], dtype=np.float64)
        x = x[:int(extra["trace"].rsplit(":", 1)[1])]    # sin el relleno del bucket
        sr = 100.0
        t0_utc = extra["t0"]
        onset = extra["p"] / sr
        refs.append(("P", onset, None))
        if np.isfinite(extra["s"]):
            refs.append(("S", extra["s"] / sr, None))
        fin = _auto_fin(x, sr, onset)
    elif ds == "ovdas":
        # Revision de etiquetas OVDAS (p. ej. sospechosos de confident learning):
        # la traza sale del Winston de la estacion de referencia, con el onset y
        # la duracion del catalogo; la clase sugerida va como referencia.
        from obspy import UTCDateTime
        from make_fig_coda_window import segment
        cfg = json.load(open(ROOT / "config" / "volcanoes.json", encoding="utf-8"))[extra["volcan"]]
        dur = max(float(extra.get("dur") or 0.0), 5.0)
        pre, post = 40.0, dur + 40.0
        onset_utc = UTCDateTime(extra["origin"])
        sr = float(cfg["sps"])
        x = segment(cfg, onset_utc, pre, post, sr)
        t0_utc = str(onset_utc - pre)
        # una sola referencia: dos textos en el mismo onset se superponen
        refs.append((f"catalogo: {r['clase']}  |  sugerida: {extra.get('sugerida', '')}"
                     f" (logistica: {extra.get('sugerida_logreg', '')})", pre, pre + dur))
        etiqueta = (pre, pre + dur)
        onset, fin = pre, pre + dur
    elif ds == "decepcion":
        sr = 50.0
        x = np.fromfile(DECEPCION / r["fuente"], dtype="<i2").astype(np.float64)
        refs.append((f"ESTE: {r['clase']}", extra["a"], extra["b"]))
        etiqueta = (extra["a"], extra["b"])
        # el archivo trae otros segmentos etiquetados: se muestran para no
        # confundirlos con el que se esta marcando
        db, f = r["fuente"].split("/raw/")
        for a, b, lab in _segmentos_decepcion().get((db, f), []):
            if lab in ("LP", "VT") and (abs(a - extra["a"]) > 1e-6
                                         or abs(b - extra["b"]) > 1e-6):
                refs.append((f"otro: {lab}", a, b))
        onset, fin = extra["a"], extra["b"]
    else:
        raise ValueError(f"dataset desconocido: {ds}")
    return dict(x=x, sr=sr, t0_utc=t0_utc, refs=refs, etiqueta=etiqueta,
                auto_onset=onset, auto_fin=fin)


def paquete(r: dict) -> dict:
    """Lo que viaja a la pagina: traza cruda y filtrada, espectrograma, marcas."""
    from scipy.signal import butter, detrend, sosfiltfilt, spectrogram
    c = cargar(r)
    x, sr = c["x"], c["sr"]
    x = detrend(np.nan_to_num(x), type="linear")
    sos = butter(4, (BANDA_VISTA[0], min(BANDA_VISTA[1], 0.45 * sr)),
                 btype="bandpass", fs=sr, output="sos")
    xf = sosfiltfilt(sos, x)
    nper = int(2 ** np.round(np.log2(2.56 * sr)))
    f, t, S = spectrogram(x, fs=sr, nperseg=nper, noverlap=3 * nper // 4)
    keep = f <= min(25.0, sr / 2)
    S = np.nan_to_num(10 * np.log10(S[keep] + 1e-20), nan=-200.0)
    lo, hi = np.percentile(S, [5, 99.5])
    auto = dict(onset=c["auto_onset"], fin=c["auto_fin"])
    if auto["onset"] is not None:
        auto.update(ruido_ini=max(0.0, auto["onset"] - RUIDO_A),
                    ruido_fin=max(0.0, auto["onset"] - RUIDO_B))
    r4 = lambda v: [round(float(u), 4) for u in v]  # noqa: E731
    return dict(item=r, sr=sr, t0_utc=c["t0_utc"], largo=len(x) / sr,
                crudo=r4(x), filtrado=r4(xf),
                spec=dict(t=r4(t), f=r4(f[keep]),
                          z=np.round(S, 1).tolist(), zmin=float(lo), zmax=float(hi)),
                refs=c["refs"], etiqueta=c["etiqueta"], auto=auto)


# -------------------------------------------------------------- el server --

class Estado:
    def __init__(self, cola: pd.DataFrame, analista: str):
        self.cola = cola.reset_index(drop=True)
        self.analista = analista
        self.picks = {}
        migrar_picks()
        if PICKS.exists():
            for p in pd.read_csv(PICKS, dtype=str, keep_default_na=False).to_dict("records"):
                self.picks[p["item_id"]] = p
        self.cache, self.lock = {}, threading.Lock()

    def item(self, i: int) -> dict:
        with self.lock:
            if i in self.cache:
                return self.cache[i]
        r = self.cola.iloc[i].to_dict()
        try:
            d = paquete(r)
        except Exception as e:                          # noqa: BLE001
            d = dict(item=r, error=f"{type(e).__name__}: {e}")
        d.update(i=i, n=len(self.cola), pick=self.picks.get(r["item_id"]))
        with self.lock:
            self.cache[i] = d
            for k in [k for k in self.cache if abs(k - i) > 3]:
                del self.cache[k]
        return d

    def precargar(self, i: int):
        if i < len(self.cola) and i not in self.cache:
            threading.Thread(target=self.item, args=(i,), daemon=True).start()

    def primera_pendiente(self) -> int:
        if getattr(self, "repasar", False):
            return 0
        for i, iid in enumerate(self.cola.item_id):
            if iid not in self.picks:
                return i
        return len(self.cola) - 1

    def avance(self) -> list[dict]:
        c = self.cola.assign(hecho=self.cola.item_id.isin(self.picks))
        g = c.groupby(["dataset", "sistema", "clase"]).hecho.agg(["sum", "size"])
        return [dict(estrato=" · ".join(k), hechos=int(v["sum"]), total=int(v["size"]))
                for k, v in g.iterrows()]

    def guardar(self, p: dict):
        r = self.cola.set_index("item_id").loc[p["item_id"]]
        fila = dict(item_id=p["item_id"], dataset=r.dataset, sistema=r.sistema,
                    clase=r.clase, estacion=r.estacion,
                    t0_utc=p.get("t0_utc", ""), sps=p.get("sps", ""),
                    analista=self.analista,
                    guardado_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        for k in ("ruido_ini", "ruido_fin", "onset", "fin", "auto_onset", "auto_fin"):
            v = p.get(k)
            fila[k] = "" if v is None else f"{float(v):.3f}"
        fila["estado"] = p["estado"]
        fila["motivo"] = p.get("motivo", "")
        fila["origen"] = p.get("origen", "")
        fila["clase_revisada"] = (p.get("clase_revisada") or "").strip().upper()
        nuevo = not PICKS.exists()
        with self.lock, PICKS.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CAMPOS)
            if nuevo:
                w.writeheader()
            w.writerow(fila)
            self.picks[p["item_id"]] = fila
            for k, d in self.cache.items():
                if d["item"]["item_id"] == p["item_id"]:
                    d["pick"] = fila


def fuera_de_ventana(picks: pd.DataFrame, cola: pd.DataFrame) -> pd.Series:
    """item_id cuyo onset marcado cae fuera de la ventana que etiqueto el
    corpus (con 2 s de tolerancia antes): los 120 s de VOISS-Net o el segmento
    del analista de Decepcion."""
    from obspy import UTCDateTime
    m = picks[picks.dataset.isin(["voissnet", "decepcion"]) & (picks.estado != "descartado")]
    m = m.merge(cola[["item_id", "extra"]], on="item_id")
    if m.empty:
        return pd.Series([], dtype=str)
    a, b = [], []
    for ds, e, t in zip(m.dataset, m.extra, m.t0_utc):
        e = json.loads(e)
        if ds == "voissnet":
            ini = UTCDateTime.strptime(e["inicio"], "%Y%m%d%H%M") - UTCDateTime(t)
            a.append(ini)
            b.append(ini + 120.0)
        else:
            a.append(e["a"])
            b.append(e["b"])
    on = pd.to_numeric(m.onset, errors="coerce").to_numpy()
    return m.item_id[(on < np.array(a) - 2.0) | (on > np.array(b))]


def ocupados_decepcion(picks: pd.DataFrame, cola: pd.DataFrame) -> dict:
    """Por item de Decepcion, los tramos del mismo archivo que no pueden ser
    ruido: los otros segmentos LP/VT del analista y los otros eventos ya
    marcados."""
    segs = _segmentos_decepcion()
    cc = cola.set_index("item_id")
    m = picks[(picks.dataset == "decepcion") & (picks.estado != "descartado")]
    ev = {}
    for q in m.itertuples():
        ev.setdefault(cc.loc[q.item_id, "fuente"], []).append(
            (q.item_id, float(q.onset), float(q.fin)))
    out = {}
    for q in m.itertuples():
        fuente = cc.loc[q.item_id, "fuente"]
        e = json.loads(cc.loc[q.item_id, "extra"])
        db, f = fuente.split("/raw/")
        t = [(a, b) for a, b, lab in segs.get((db, f), [])
             if lab in ("LP", "VT") and (abs(a - e["a"]) > 1e-6 or abs(b - e["b"]) > 1e-6)]
        t += [(a, b) for iid, a, b in ev.get(fuente, []) if iid != q.item_id]
        out[q.item_id] = t
    return out


def ruido_con_evento(picks: pd.DataFrame, cola: pd.DataFrame) -> pd.Series:
    """item_id de Decepcion cuya ventana de ruido pisa otro evento."""
    oc = ocupados_decepcion(picks, cola)
    m = picks[picks.item_id.isin(oc)].set_index("item_id")
    malos = [i for i, t in oc.items()
             if any(a < float(m.loc[i, "ruido_fin"]) and b > float(m.loc[i, "ruido_ini"])
                    for a, b in t)]
    return pd.Series(malos, dtype=str)


def fin_en_borde(picks: pd.DataFrame, cola: pd.DataFrame) -> pd.Series:
    """item_id cuyo fin marcado queda a menos de MARGEN_FIN del final de la
    traza: o la coda sigue despues y la duracion queda cortada, o no queda el
    margen de filtro que exige el paquete y el evento se pierde. VCSEIS trae
    el largo en el nombre; VOISS-Net baja 300 s; en Decepcion sale del tamano
    del archivo (int16 a 50 Hz)."""
    m = picks[picks.dataset.isin(["vcseis", "voissnet", "decepcion"])
              & (picks.estado != "descartado")]
    m = m.merge(cola[["item_id", "extra", "fuente"]], on="item_id")
    largo = []
    for ds, e, f in zip(m.dataset, m.extra, m.fuente):
        if ds == "vcseis":
            largo.append(int(json.loads(e)["trace"].rsplit(":", 1)[1]) / 100.0)
        elif ds == "voissnet":
            largo.append(PRE_VOISS + 120.0 + POST_VOISS)
        else:
            largo.append((DECEPCION / f).stat().st_size / 2 / 50.0)
    fin = pd.to_numeric(m.fin, errors="coerce").to_numpy()
    return m.item_id[fin > np.array(largo) - MARGEN_FIN]


def orden_tornillos(cola: pd.DataFrame) -> list[str]:
    """LP marcados buenos o dudosos, del mas al menos probable tornillo segun
    scripts/indice_tornillo.py. Un tornillo decae lento (Q de coda alto) y es
    monocromatico (energia concentrada en el pico), asi que el puntaje es
    Q de coda x concentracion, sobre los ajustes de coda con R2 >= 0,7; el
    resto va despues, por concentracion."""
    ruta = ROOT / "data" / "processed" / "indice_tornillo.parquet"
    if not ruta.exists():
        sys.exit("Falta el indice: corre  python scripts/indice_tornillo.py")
    d = pd.read_parquet(ruta)
    d = d[(d.motivo == "") & d.item_id.isin(cola.item_id)].copy()
    fiable = (d.r2_coda >= 0.7) & np.isfinite(d.q_coda)
    d["puntaje"] = np.where(fiable, d.q_coda * d.concentracion, -1.0)
    return d.sort_values(["puntaje", "concentracion"], ascending=False).item_id.tolist()


def a_repasar(criterio: str, cola: pd.DataFrame) -> set:
    if not PICKS.exists():
        return set()
    migrar_picks()
    p = pd.read_csv(PICKS, dtype=str, keep_default_na=False)
    p = p.drop_duplicates("item_id", keep="last")
    if criterio == "fuera_de_ventana":
        return set(fuera_de_ventana(p, cola))
    if criterio == "fin_en_borde":
        return set(fin_en_borde(p, cola))
    if criterio == "ruido_con_evento":
        return set(ruido_con_evento(p, cola))
    if criterio == "ruido_recalculado":
        return set(p.item_id[p.motivo.str.startswith("ruido recalculado")])
    return set(p.item_id[p.origen == criterio])


def servir(puerto: int, analista: str, dataset, sistema, clase, repasar=None,
           incluir_todo=False):
    warnings.filterwarnings("ignore")
    if not COLA.exists():
        sys.exit("No hay cola: corre primero  python scripts/cortar_trazas.py cola")
    cola = pd.read_csv(COLA, dtype=str, keep_default_na=False)
    for col, v in (("dataset", dataset), ("sistema", sistema), ("clase", clase)):
        if v:
            cola = cola[cola[col].isin(v)]
    if not incluir_todo and not clase:
        cola = cola[~(cola.clase.isin(EXCLUIDAS_POR_AHORA)
                      & cola.dataset.isin(CONTINUAS_EN))]
    if not incluir_todo and not sistema:
        marcadas = set(pd.read_csv(PICKS, dtype=str).item_id) if PICKS.exists() else set()
        estrato = list(zip(cola.dataset, cola.sistema, cola.clase))
        oculta = [e in EN_MUESTRA and i not in marcadas
                  for e, i in zip(estrato, cola.item_id)]
        cola = cola[~np.array(oculta, dtype=bool)]
    if repasar == "tornillos":
        orden = orden_tornillos(cola)
        cola = cola.set_index("item_id").loc[orden].reset_index()
    elif repasar == "otra_estacion":
        # descartes "sin evento visible" del Ruiz, mostrados en Bis: la cola
        # se cambia solo en memoria, y la marca guarda la estacion usada
        p = pd.read_csv(PICKS, dtype=str, keep_default_na=False)
        p = p.drop_duplicates("item_id", keep="last")
        # las ya guardadas en Bis (buenas o confirmadas sin evento) no vuelven
        ids = set(p.item_id[(p.dataset == "sgc") & (p.motivo == "sin evento visible")
                            & ~p.estacion.isin(SGC_ALTERNAS)])
        cola = cola[cola.item_id.isin(ids)].copy()
        cola["estacion"] = [estacion_sgc_alterna(EXT / "sgc_ruiz" / f) for f in cola.fuente]
        cola = cola[cola.estacion != ""]
    elif repasar:
        cola = cola[cola.item_id.isin(a_repasar(repasar, cola))]
    if cola.empty:
        sys.exit("La cola filtrada quedo vacia.")
    E = Estado(cola, analista)
    E.repasar = bool(repasar)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj, allow_nan=False, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                b = PAGINA.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif u.path == "/api/estado":
                self._json(dict(inicio=E.primera_pendiente(), n=len(E.cola),
                                avance=E.avance(), analista=E.analista))
            elif u.path == "/api/item":
                i = min(max(int(q.get("i", ["0"])[0]), 0), len(E.cola) - 1)
                d = E.item(i)
                E.precargar(i + 1)
                self._json(d)
            else:
                self._json(dict(error="no existe"), 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/pick":
                return self._json(dict(error="no existe"), 404)
            p = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            E.guardar(p)
            self._json(dict(ok=True, avance=E.avance()))

    import socket
    try:
        with socket.create_connection(("::1", puerto), timeout=1):
            sys.exit(f"Hay algo escuchando en [::1]:{puerto}, probablemente un tunel "
                     "ssh abierto en este mismo servidor. Cierralo (pgrep -af 'ssh -L') "
                     "y abre el tunel desde tu equipo.")
    except OSError:
        pass
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", puerto), H)
    except OSError:
        sys.exit(f"El puerto {puerto} esta ocupado: ya hay una pagina abierta. "
                 "Cierrala con Ctrl+C en su terminal, o busca su proceso con\n"
                 "    pgrep -af 'cortar_trazas.py servir'\n"
                 "y detenlo con  kill <pid>. Si quieres dos a la vez, usa --puerto.")
    print(f"{len(E.cola)} trazas en la cola, {sum(i in E.picks for i in E.cola.item_id)} "
          f"ya marcadas. Retoma en la {E.primera_pendiente() + 1}.")
    print(f"Abrir http://localhost:{puerto}. Desde otro equipo, abre el tunel "
          f"EN ESE EQUIPO, no aqui:\n    ssh -L {puerto}:127.0.0.1:{puerto} "
          "<usuario>@<servidor>")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"\nPicks en {PICKS}")


PAGINA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Corte de trazas</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#dcdcd6;--card:#fff;
--ruido:rgba(120,120,120,.22);--evento:rgba(230,126,34,.25);--ref:#2f6fb5}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1a;--fg:#ecece8;--mut:#a0a09a;
--line:#3a3a37;--card:#242422}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.4 system-ui,sans-serif}
header{display:flex;gap:16px;align-items:baseline;padding:10px 16px;
border-bottom:1px solid var(--line);flex-wrap:wrap}
h1{font-size:16px;margin:0}#meta{color:var(--mut)}#pos{margin-left:auto;color:var(--mut)}
main{display:grid;grid-template-columns:1fr 290px;gap:0}
#plot{height:calc(100vh - 50px)}
aside{border-left:1px solid var(--line);padding:12px 14px;overflow:auto;
height:calc(100vh - 50px);background:var(--card)}
aside h2{font-size:13px;margin:14px 0 6px;color:var(--mut);font-weight:600}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td{padding:2px 4px;border-bottom:1px solid var(--line)}td:last-child{text-align:right}
.activo td{font-weight:600}.aviso{color:#c0392b;margin:6px 0}
.guardado{color:#27ae60}.kbd{font-family:ui-monospace,monospace;background:var(--line);
padding:0 4px;border-radius:3px}#teclas td:first-child{white-space:nowrap}
#avance td{font-size:12px}#err{color:#c0392b;padding:20px}
</style></head><body>
<header><h1 id="titulo">Corte de trazas</h1><span id="meta"></span><span id="pos"></span></header>
<main><div id="plot"></div><aside>
<div id="estado"></div>
<h2>Ventanas (s desde el inicio del tramo)</h2>
<table id="marcas"></table><div id="avisos"></div>
<h2>Teclas, con el mouse sobre la traza</h2>
<table id="teclas">
<tr><td><span class=kbd>1</span> <span class=kbd>2</span></td><td>inicio / fin del ruido</td></tr>
<tr><td><span class=kbd>3</span> <span class=kbd>4</span></td><td>onset / fin del evento</td></tr>
<tr><td><span class=kbd>Enter</span></td><td>guardar como bueno (moviste algo)</td></tr>
<tr><td><span class=kbd>a</span></td><td>bueno, acepto el corte propuesto</td></tr>
<tr><td><span class=kbd>u</span></td><td>guardar como dudoso</td></tr>
<tr><td><span class=kbd>x</span></td><td>descartar, con motivo</td></tr>
<tr><td><span class=kbd>n</span> <span class=kbd>p</span></td><td>siguiente / anterior</td></tr>
<tr><td><span class=kbd>f</span></td><td>onset a la línea verde (AIC)</td></tr>
<tr><td><span class=kbd>c</span></td><td>reclasificar (p. ej. LP → TO)</td></tr>
<tr><td><span class=kbd>v</span></td><td>filtrada / cruda</td></tr>
<tr><td><span class=kbd>z</span> <span class=kbd>0</span></td><td>zoom al evento / todo</td></tr>
</table>
<h2>Avance por estrato</h2><table id="avance"></table>
</aside></main>
<script>
let D=null, M={}, M0={}, xCursor=null, crudo=false, N=0, ruidoManual=false, claseRev="", claseRev0="";
const CLASES_REV="LP VT HB TO VLP AV TR EX RUIDO OTRO";
const cambios=()=>Object.keys(NOMBRES).some(k=>M[k]!==M0[k])||claseRev!==claseRev0;
function titulo(){const it=D.item;document.getElementById("titulo").textContent=
  `${it.sistema} · ${it.clase}`+(claseRev?` → ${claseRev}`:"")}
const NOMBRES={ruido_ini:"inicio ruido",ruido_fin:"fin ruido",onset:"onset",fin:"fin evento"};
const TECLA={"1":"ruido_ini","2":"ruido_fin","3":"onset","4":"fin"};
// Picador AIC (Maeda) en [t0-3, t0+3] s, como scripts/afinar_onset.py
function aic(t0){
  if(t0==null||!D||!D.filtrado)return null;
  const y=D.filtrado,sr=D.sr,i0=Math.max(0,Math.round((t0-3)*sr)),i1=Math.min(y.length,Math.round((t0+3)*sr));
  const n=i1-i0;if(n<20)return null;
  const c1=new Float64Array(n+1),c2=new Float64Array(n+1);
  for(let i=0;i<n;i++){const v=y[i0+i];c1[i+1]=c1[i]+v;c2[i+1]=c2[i]+v*v}
  let best=Infinity,kb=-1;
  for(let k=5;k<n-5;k++){
    const m1=c1[k]/k,v1=c2[k]/k-m1*m1,r1=c1[n]-c1[k],r2=c2[n]-c2[k],m2=r1/(n-k),v2=r2/(n-k)-m2*m2;
    if(v1<=0||v2<=0)continue;const a=k*Math.log(v1)+(n-k-1)*Math.log(v2);
    if(a<best){best=a;kb=k}}
  return kb<0?null:+((i0+kb)/sr).toFixed(3)}
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const f2=v=>v==null?"—":v.toFixed(2);

async function estado(){const e=await (await fetch("/api/estado")).json();N=e.n;
  avance(e.avance);return e.inicio}
function avance(a){document.getElementById("avance").innerHTML=a.map(r=>
  `<tr><td>${r.estrato}</td><td>${r.hechos} / ${r.total}</td></tr>`).join("")}

async function ir(i){
  if(i<0||i>=N)return;
  D=await (await fetch("/api/item?i="+i)).json();
  const it=D.item;
  claseRev=claseRev0=(D.pick&&D.pick.clase_revisada)||"";titulo();
  document.getElementById("meta").textContent=`${it.dataset} · ${it.estacion} · ${it.fuente}`;
  document.getElementById("pos").textContent=`${D.i+1} de ${D.n}`;
  if(D.error){Plotly.purge("plot");document.getElementById("plot").innerHTML=
    `<div id=err>No se pudo leer la traza: ${D.error}<br>Descártala con <b>x</b> o pasa con <b>n</b>.</div>`;
    M={};panel();return}
  const p=D.pick, num=v=>v===""||v==null?null:parseFloat(v);
  M=p?{ruido_ini:num(p.ruido_ini),ruido_fin:num(p.ruido_fin),onset:num(p.onset),fin:num(p.fin)}
     :{ruido_ini:D.auto.ruido_ini??null,ruido_fin:D.auto.ruido_fin??null,
       onset:D.auto.onset??null,fin:D.auto.fin??null};
  // un ruido que no esta en [onset-12, onset-2] ya fue puesto a mano
  ruidoManual=M.onset!=null&&M.ruido_ini!=null&&
    (Math.abs(M.ruido_fin-Math.max(0,M.onset-2))>.01||Math.abs(M.ruido_ini-Math.max(0,M.onset-12))>.01);
  M0={...M};dibujar();panel();
}

function dibujar(){
  const t=Array.from({length:D.crudo.length},(_,k)=>k/D.sr);
  const y=crudo?D.crudo:D.filtrado, fg=css("--fg");
  Plotly.react("plot",[
    {x:t,y:y,type:"scattergl",mode:"lines",line:{width:1,color:fg},hoverinfo:"x",xaxis:"x",yaxis:"y"},
    {x:D.spec.t,y:D.spec.f,z:D.spec.z,type:"heatmap",colorscale:"Viridis",showscale:false,
     zmin:D.spec.zmin,zmax:D.spec.zmax,hoverinfo:"x+y",xaxis:"x",yaxis:"y2"}],
   {showlegend:false,
    margin:{l:60,r:12,t:10,b:40},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",
    font:{color:fg},xaxis:{title:"s",range:rango(),anchor:"y2"},
    yaxis:{domain:[.42,1],title:crudo?"cruda":"0,5–10 Hz",anchor:"x"},
    yaxis2:{domain:[0,.36],title:"Hz",anchor:"x"},shapes:formas(),annotations:notas(),
    dragmode:"zoom",hovermode:"x"},{responsive:true,displaylogo:false});
  const gd=document.getElementById("plot");
  gd.removeAllListeners&&gd.removeAllListeners("plotly_hover");
  gd.on("plotly_hover",e=>{xCursor=e.points[0].x});
}
let zoom=null;
function rango(){return zoom||[0,D.largo]}
function formas(){
  const ai=aic(M.onset);
  const s=[], vl=(x,c,d)=>({type:"line",xref:"x",yref:"paper",x0:x,x1:x,y0:0,y1:1,
    line:{color:c,width:1.5,dash:d||"solid"}});
  const rect=(a,b,c)=>({type:"rect",xref:"x",yref:"paper",x0:a,x1:b,y0:0,y1:1,
    fillcolor:c,line:{width:0},layer:"below"});
  if(M.ruido_ini!=null&&M.ruido_fin!=null)s.push(rect(M.ruido_ini,M.ruido_fin,css("--ruido")));
  if(M.onset!=null&&M.fin!=null)s.push(rect(M.onset,M.fin,css("--evento")));
  for(const k in M)if(M[k]!=null)s.push(vl(M[k],k.startsWith("ruido")?"#777":"#e67e22"));
  for(const [txt,a,b] of D.refs){const c=txt.startsWith("otro")?"#999":css("--ref"),d=txt.startsWith("otro")?"dot":"dash";
    s.push(vl(a,c,d));if(b!=null)s.push(vl(b,c,d))}
  if(ai!=null)s.push({type:"line",xref:"x",yref:"paper",x0:ai,x1:ai,y0:0,y1:1,
    line:{color:"#27ae60",width:1.5,dash:"dot"}});
  return s}
function notas(){return D.refs.map(([txt,a])=>({x:a,y:1,xref:"x",yref:"paper",text:txt,
  showarrow:false,xanchor:"left",yanchor:"top",
  font:{size:11,color:txt.startsWith("otro")?"#999":css("--ref")}}))}

function rms(a,b){if(a==null||b==null||b<=a)return null;
  const i0=Math.max(0,Math.round(a*D.sr)),i1=Math.min(D.filtrado.length,Math.round(b*D.sr));
  let s=0;for(let i=i0;i<i1;i++)s+=D.filtrado[i]**2;return i1>i0?Math.sqrt(s/(i1-i0)):null}
function fueraVentana(){const w=D&&D.etiqueta;
  return w&&M.onset!=null&&(M.onset<w[0]-2||M.onset>w[1])}
function avisos(){const a=[];const m=M;
  if(m.ruido_ini!=null&&m.ruido_fin!=null&&D.refs.some(([t,x,y])=>t.startsWith("otro")&&y!=null&&x<m.ruido_fin&&y>m.ruido_ini))
    a.push("el ruido se cruza con otro evento etiquetado");
  if(m.fin!=null&&m.fin>D.largo-3)
    a.push("el fin está a menos de 3 s del final de la traza: el paquete no calcula descriptores");
  if(fueraVentana())a.push("el onset está fuera de la ventana etiquetada: no es el evento que clasificó el corpus");
  if(m.ruido_ini==null||m.ruido_fin==null||m.onset==null||m.fin==null)a.push("faltan marcas");
  else{if(!(m.ruido_ini<m.ruido_fin&&m.ruido_fin<=m.onset&&m.onset<m.fin))
      a.push("el orden debe ser ruido → onset → fin");
    if(m.ruido_fin-m.ruido_ini<5)a.push("ruido de menos de 5 s");
    if(m.onset-m.ruido_fin<2)a.push("menos de 2 s entre ruido y onset");
    if(m.fin-m.onset<5)a.push("evento de menos de 5 s")}
  return a}
function panel(){
  const d=M.fin!=null&&M.onset!=null?M.fin-M.onset:null;
  const rs=rms(M.onset,M.fin), rn=rms(M.ruido_ini,M.ruido_fin);
  const snr=rs&&rn?20*Math.log10(rs/rn):null;
  const auto=D&&D.auto?D.auto:{};
  document.getElementById("marcas").innerHTML=Object.keys(NOMBRES).map(k=>
    `<tr><td>${NOMBRES[k]}</td><td>${f2(M[k])}</td></tr>`).join("")+
    `<tr><td>duración</td><td>${f2(d)}</td></tr><tr><td>SNR 0,5–10 Hz (dB)</td><td>${f2(snr)}</td></tr>`+
    `<tr><td>AIC (tecla f)</td><td>${f2(aic(M.onset))}</td></tr>`+
    `<tr><td>automático</td><td>${f2(auto.onset)} → ${f2(auto.fin)}</td></tr>`;
  document.getElementById("avisos").innerHTML=(D&&!D.error?avisos():[]).map(t=>`<div class=aviso>${t}</div>`).join("");
  const p=D&&D.pick;
  document.getElementById("estado").innerHTML=p?
    `<div class=guardado>Guardada: ${p.estado}${p.origen?", "+p.origen:""}${p.motivo?" ("+p.motivo+")":""}, ${p.guardado_utc}</div>`:
    "<div>Sin marcar. Las ventanas propuestas son las automáticas.</div>";
}

async function guardar(estadoPick,motivo,origen){
  if(estadoPick!=="descartado"&&avisos().some(a=>a==="faltan marcas"||a.startsWith("el orden")))
    {alert("Faltan marcas o el orden no es ruido → onset → fin.");return}
  if(estadoPick!=="descartado"&&M.fin-M.onset<5){
    if(confirm("El evento dura menos de 5 s y no tendrá descriptores. ¿Descartarlo?"))
      guardar("descartado","evento de menos de 5 s");
    return}
  if(estadoPick!=="descartado"&&fueraVentana()&&
     !confirm("El onset está fuera de la ventana etiquetada. ¿Guardar igual?"))return;
  if(estadoPick==="descartado")origen="";
  else if(!origen)origen=(cambios()||claseRev!==claseRev0)?"editado":estadoPick==="dudoso"?"aceptado":
    (D.pick?D.pick.origen||"editado":null);
  if(origen===null){alert("No moviste ninguna marca. Si el corte propuesto está bien, guárdalo con  a.");return}
  const body={item_id:D.item.item_id,estado:estadoPick,motivo:motivo||"",
    t0_utc:D.t0_utc||"",sps:D.sr||"",auto_onset:D.auto?D.auto.onset:null,
    auto_fin:D.auto?D.auto.fin:null,origen:origen,clase_revisada:claseRev,...(estadoPick==="descartado"&&D.error?{}:M)};
  const r=await (await fetch("/api/pick",{method:"POST",body:JSON.stringify(body)})).json();
  avance(r.avance);zoom=null;ir(D.i+1)}

document.addEventListener("keydown",e=>{
  if(!D||e.target.tagName==="INPUT")return;
  if(TECLA[e.key]&&xCursor!=null&&!D.error){const k=TECLA[e.key];M[k]=+xCursor.toFixed(3);
    if(k.startsWith("ruido"))ruidoManual=true;
    else if(k==="onset"&&!ruidoManual){M.ruido_ini=+Math.max(0,M.onset-12).toFixed(3);
      M.ruido_fin=+Math.max(0,M.onset-2).toFixed(3)}
    Plotly.relayout("plot",{shapes:formas()});panel();return}
  switch(e.key){
    case "Enter":guardar("bueno");break;
    case "a":guardar("bueno","",cambios()?"editado":"aceptado");break;
    case "u":guardar("dudoso",prompt("Motivo (opcional):")||"");break;
    case "x":{const m=prompt("Motivo del descarte:","sin evento visible");if(m!==null)guardar("descartado",m);break}
    case "n":case "ArrowRight":zoom=null;ir(D.i+1);break;
    case "p":case "ArrowLeft":zoom=null;ir(D.i-1);break;
    case "c":{const r=prompt("Clase revisada ("+CLASES_REV+"). Vacío = la del corpus ("+D.item.clase+").",claseRev);
      if(r===null)break;const v=r.trim().toUpperCase();
      if(v&&!CLASES_REV.split(" ").includes(v)){alert("Clase no reconocida: "+v);break}
      claseRev=v===D.item.clase?"":v;titulo();panel();break}
    case "f":{if(D.error)break;const ai=aic(M.onset);if(ai==null)break;M.onset=ai;
      if(!ruidoManual){M.ruido_ini=+Math.max(0,ai-12).toFixed(3);M.ruido_fin=+Math.max(0,ai-2).toFixed(3)}
      Plotly.relayout("plot",{shapes:formas()});panel();break}
    case "v":crudo=!crudo;dibujar();break;
    case "z":if(M.onset!=null){const a=(M.ruido_ini??M.onset)-5,b=(M.fin??M.onset+30)+10;
      zoom=[Math.max(0,a),Math.min(D.largo,b)];Plotly.relayout("plot",{"xaxis.range":zoom})}break;
    case "0":zoom=null;Plotly.relayout("plot",{"xaxis.range":[0,D.largo]});break;
  }});
estado().then(ir);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cola", help="arma la cola de trazas, una vez")
    c.add_argument("--vcseis-n", type=int, default=200,
                   help="trazas por region y clase de VCSEIS")
    c.add_argument("--semilla", type=int, default=41)
    c.add_argument("--ampliar", action="store_true",
                   help="agrega al final lo que falte, sin tocar lo que ya esta")
    s = sub.add_parser("servir", help="levanta la pagina para marcar")
    s.add_argument("--puerto", type=int, default=8765)
    s.add_argument("--analista", default="SM")
    s.add_argument("--dataset", nargs="*", help="voissnet sgc vcseis decepcion")
    s.add_argument("--sistema", nargs="*", help="p. ej. Kilauea Ruiz")
    s.add_argument("--clase", nargs="*", help="p. ej. LP VT")
    s.add_argument("--repasar", choices=["tornillos", "otra_estacion", "fuera_de_ventana",
                                         "ruido_con_evento",
                                         "fin_en_borde",
                                         "ruido_recalculado",
                                         "igual_al_auto_sin_registro", "aceptado"],
                   help="solo trazas ya marcadas que cumplen el criterio")
    s.add_argument("--cola", help="cola alternativa (p. ej. revision de etiquetas)")
    s.add_argument("--picks", help="archivo de marcas alternativo")
    s.add_argument("--incluir-todo", action="store_true",
                   help="muestra tambien " + ", ".join(EXCLUIDAS_POR_AHORA))
    a = ap.parse_args()
    global COLA, PICKS
    if getattr(a, "cola", None):
        COLA = Path(a.cola)
    if getattr(a, "picks", None):
        PICKS = Path(a.picks)
    if a.cmd == "cola":
        (ampliar_cola if a.ampliar else armar_cola)(a.vcseis_n, a.semilla)
    else:
        servir(a.puerto, a.analista, a.dataset, a.sistema, a.clase, a.repasar,
               a.incluir_todo)


if __name__ == "__main__":
    main()
