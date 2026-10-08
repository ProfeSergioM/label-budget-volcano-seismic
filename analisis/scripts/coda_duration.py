"""
Estimador del fin de la coda, y su validacion contra la duracion del analista.

El corpus VCSEIS entrega arribos P y S pero no fin de coda, de modo que la
ventana [onset, onset+duracion] no se puede reproducir ahi. Este modulo la
reconstruye desde la traza, con la definicion operativa con que un analista
mide la duracion: el evento termina cuando la envolvente vuelve a un multiplo
del ruido previo y se queda abajo.

    duracion = ultimo instante t en [onset, onset+T_MAX] tal que la envolvente
               suavizada supera k veces el RMS del ruido previo, exigiendo
               ademas que se mantenga por debajo durante HOLD segundos.

El estimador no se justifica por argumento sino por medicion: se calibra k
contra la duracion que escribio el analista sobre una muestra del corpus
OVDAS, donde esa duracion existe para 1,47 millones de eventos, y recien
despues se aplica a VCSEIS.

Uso:
    python scripts/coda_duration.py --validar --n 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import rqa_core as C  # noqa: E402

CONFIG_PATH = ROOT / "config" / "volcanoes.json"
CAT = ROOT / "data" / "raw" / "catalog"
OUT = ROOT / "experiments" / "reports" / "41_coda_duration.csv"
CACHE = ROOT / "data" / "interim" / "coda_validation"
VOLCANOES = ["lagmaule", "nevchillan", "copahue", "villarrica", "hudson"]

# Ventana fija que se descarga para validar: generosa por los dos lados, para
# que el estimador no quede acotado por el propio segmento.
PRE_S, POST_S = 15.0, 140.0
BAND = C.SNR_BAND          # misma banda en que la tuberia mide el SNR
SMOOTH_S = 1.0             # ventana de la envolvente RMS movil
HOLD_S = 3.0               # cuanto debe quedarse abajo para declarar el fin
NOISE_A, NOISE_B = 12.0, 2.0   # ventana de ruido, en segundos antes del onset
T_MAX = 120.0
K_GRID = [1.1, 1.2, 1.3, 1.5, 1.8, 2.2, 3.0]
HOLD_GRID = [2.0, 3.0, 5.0, 8.0]
SMOOTH_GRID = [0.5, 1.0, 2.0]


def envelope(x: np.ndarray, sr: float, smooth_s: float = SMOOTH_S) -> np.ndarray:
    """RMS movil sobre |x|, con ventana de smooth_s segundos."""
    n = max(1, int(round(smooth_s * sr)))
    c = np.convolve(x.astype(float) ** 2, np.ones(n) / n, mode="same")
    return np.sqrt(np.maximum(c, 0.0))


def coda_duration(seg: np.ndarray, sr: float, pre_s: float, k: float,
                  t_max: float = T_MAX, hold_s: float = HOLD_S,
                  smooth_s: float = SMOOTH_S) -> float:
    """Duracion estimada, en segundos, para un segmento alineado a onset-pre_s.

    Devuelve NaN si la ventana de ruido no cabe o si el evento no supera el
    umbral en ningun instante: no se fabrica un valor.
    """
    x = C.preprocess_segment(seg, sr, band=BAND)
    env = envelope(x, sr, smooth_s)
    i0 = int(round((pre_s - NOISE_A) * sr))
    i1 = int(round((pre_s - NOISE_B) * sr))
    if i0 < 0 or i1 <= i0 or i1 > len(env):
        return float("nan")
    ruido = float(np.median(env[i0:i1]))       # mediana, no media: robusta a
    if not np.isfinite(ruido) or ruido <= 0:   # un transitorio dentro del ruido
        return float("nan")

    s0 = int(round(pre_s * sr))
    s1 = min(len(env), s0 + int(round(t_max * sr)))
    if s1 - s0 < int(round(C.MIN_DURATION_S * sr)):
        return float("nan")
    arriba = env[s0:s1] > k * ruido
    if not arriba.any():
        return float("nan")

    # El evento termina la primera vez que la envolvente se queda por debajo
    # del umbral durante hold_s segundos seguidos. Se recorre HACIA ADELANTE
    # desde el onset: los valles internos mas cortos que hold_s no cortan, que
    # es lo que tampoco hace un analista.
    h = int(round(hold_s * sr))
    idx = np.flatnonzero(arriba)
    fin = idx[0]
    for i in idx[1:]:
        if i - fin > h:
            break
        fin = i
    return float((fin + 1) / sr)


def descargar(volcano: str, n: int, seed: int = 42) -> list[dict]:
    """Muestra estratificada por clase, con su segmento de ventana fija."""
    from download_and_extract import make_clients, fetch_block
    vcfg = json.load(open(CONFIG_PATH, encoding="utf-8"))[volcano]
    sr = vcfg["sps"]
    cat = pd.read_parquet(CAT / f"{volcano}_catalog.parquet",
                          columns=["fecha", "tipoevento", "duracion", "idevento",
                                   "unixtime", "estacion"])
    cat = cat[(cat.duracion >= C.MIN_DURATION_S) & (cat.duracion <= T_MAX)]
    cat = cat[cat.estacion == vcfg["ref_station"]]
    # Solo las clases del estudio, no todos los codigos del catalogo.
    from class_style import CLASS_ORDER
    cnt = cat.tipoevento.value_counts()
    clases = [c for c in CLASS_ORDER if cnt.get(c, 0) >= 100]
    m = pd.concat([cat[cat.tipoevento == c].sample(min(n, int(cnt[c])),
                                                   random_state=seed)
                   for c in clases], ignore_index=True)
    clients = make_clients(vcfg)
    out = []
    for _, e in m.iterrows():
        onset = UTCDateTime(pd.Timestamp(e.fecha).to_pydatetime())
        st = fetch_block(clients, vcfg, onset - PRE_S - 5, onset + POST_S + 5)
        if st is None:
            continue
        for tr in st:
            if tr.stats.sampling_rate != sr:
                continue
            if tr.stats.starttime <= onset - PRE_S and tr.stats.endtime >= onset + POST_S:
                d = np.asarray(tr.slice(starttime=onset - PRE_S,
                                        endtime=onset + POST_S).data, dtype=float)
                nexp = int(round((PRE_S + POST_S) * sr))
                if len(d) >= nexp:
                    out.append(dict(volcano=volcano, label=e.tipoevento,
                                    duracion_analista=float(e.duracion),
                                    seg=d[:nexp], sr=sr))
                break
    return out


def cargar_cache(volcano: str, n: int):
    """Los segmentos se descargan una vez y se guardan: el estimador se barre
    muchas veces y la descarga es lo unico caro."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{volcano}_{n}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        return [dict(volcano=volcano, label=l, duracion_analista=float(d),
                     seg=s, sr=float(r))
                for l, d, s, r in zip(z["label"], z["dur"], z["seg"], z["sr"])]
    datos = descargar(volcano, n)
    if datos:
        np.savez_compressed(
            p, label=np.array([d["label"] for d in datos]),
            dur=np.array([d["duracion_analista"] for d in datos]),
            seg=np.stack([d["seg"] for d in datos]),
            sr=np.array([d["sr"] for d in datos]))
    return datos


def descriptores_con_ventana(seg, sr, pre_s, dur):
    """Los cinco descriptores medidos sobre [onset, onset+dur] del segmento."""
    plan = C.duration_window_plan(dur, sr)
    i0 = int(round((pre_s - plan["pre_s"]) * sr))
    if i0 < 0 or i0 + plan["total_samples"] > len(seg):
        return None
    x = seg[i0:i0 + plan["total_samples"]]
    w = C.preprocess_segment(x, sr, band=C.WIDE_FILTER_BAND)
    r = C.preprocess_segment(x, sr, band=C.DURATION_FILTER_BAND)
    win_w = w[plan["signal_start"]:plan["signal_end"]]
    win_r = r[plan["signal_start"]:plan["signal_end"]]
    if len(win_r) != plan["n_window"]:
        return None
    rqa, _, _ = C.compute_rqa_domain(C.zscore(win_r), C.M_TIME, C.TAU_TIME)
    sp = C.compute_spectral_wide(win_w, sr)
    return dict(SR_w=sp["SR_w"], FI_hi=sp["FI_hi"], FI_lo=sp["FI_lo"],
                DET=rqa["DET"], Wentr=rqa["Wentr"])


def comparar_ventanas(a, k, hold, sm):
    """Compara los descriptores medidos sobre la ventana del analista contra
    los medidos sobre la ventana estimada, sobre los mismos eventos."""
    from itertools import combinations
    from scipy.stats import spearmanr
    COLS = ["SR_w", "FI_hi", "FI_lo", "DET", "Wentr"]

    def solape(u, v, nb=50):
        u, v = u[np.isfinite(u)], v[np.isfinite(v)]
        if len(u) < 5 or len(v) < 5:
            return np.nan
        lo, hi = np.percentile(np.concatenate([u, v]), [1, 99])
        if hi <= lo:
            return 1.0
        e = np.linspace(lo, hi, nb)
        hu, _ = np.histogram(u, bins=e)
        hv, _ = np.histogram(v, bins=e)
        return float(np.minimum(hu / max(hu.sum(), 1), hv / max(hv.sum(), 1)).sum())

    rows = []
    for v in VOLCANOES:
        ds = cargar_cache(v, a.n)
        A, E, lab = [], [], []
        for d in ds:
            est = coda_duration(d["seg"], d["sr"], PRE_S, k, hold_s=hold, smooth_s=sm)
            if not np.isfinite(est) or est < C.MIN_DURATION_S:
                continue
            fa = descriptores_con_ventana(d["seg"], d["sr"], PRE_S,
                                          d["duracion_analista"])
            fe = descriptores_con_ventana(d["seg"], d["sr"], PRE_S, est)
            if fa is None or fe is None:
                continue
            A.append(fa); E.append(fe); lab.append(d["label"])
        if len(A) < 40:
            print(f"  {v}: solo {len(A)} pares, se omite")
            continue
        A = pd.DataFrame(A); E = pd.DataFrame(E); lab = np.array(lab)
        clases = [c for c in pd.Series(lab).value_counts().index
                  if (lab == c).sum() >= 25]
        print(f"\n  {v}  ({len(A)} eventos, clases {clases})")
        for c in COLS:
            m = A[c].notna() & E[c].notna()
            rho = float(spearmanr(A.loc[m, c], E.loc[m, c]).statistic)
            sa = np.nanmean([solape(A.loc[lab == x, c].to_numpy(float),
                                    A.loc[lab == y, c].to_numpy(float))
                             for x, y in combinations(clases, 2)])
            se = np.nanmean([solape(E.loc[lab == x, c].to_numpy(float),
                                    E.loc[lab == y, c].to_numpy(float))
                             for x, y in combinations(clases, 2)])
            rows.append(dict(volcano=v, descriptor=c, rho=rho,
                             solape_analista=sa, solape_estimada=se))
            print(f"    {c:6s} rho {rho:+.3f}   solape entre clases: "
                  f"analista {sa:.3f} -> estimada {se:.3f}  ({se - sa:+.3f})")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validar", action="store_true")
    ap.add_argument("--descriptores", action="store_true",
                    help="compara los descriptores entre las dos ventanas")
    ap.add_argument("--k", type=float, default=1.2)
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--smooth", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=200, help="eventos por clase")
    a = ap.parse_args()
    if a.descriptores:
        print(f"=== ventana del analista contra ventana estimada "
              f"(k={a.k}, hold={a.hold} s, suavizado={a.smooth} s)")
        t = comparar_ventanas(a, a.k, a.hold, a.smooth)
        p = OUT.parent / "42_ventana_estimada.csv"
        t.to_csv(p, index=False)
        print(f"\n  rho medio {t.rho.mean():+.3f} | "
              f"cambio medio de solape {(t.solape_estimada - t.solape_analista).mean():+.3f}")
        print(f"\n-> {p}")
        return
    if not a.validar:
        print(__doc__)
        return

    datos = {v: cargar_cache(v, a.n) for v in VOLCANOES}
    for v, d in datos.items():
        print(f"{v}: {len(d)} segmentos con cobertura completa")

    # Barrido conjunto de las tres constantes. Se elige UNA combinacion para
    # los cinco sistemas: un ajuste por volcan no serviria en VCSEIS, que es
    # donde el estimador tiene que funcionar sin datos de calibracion.
    print(f"\n{'='*88}\n=== barrido: acuerdo con la duracion del analista\n{'='*88}")
    print(f"  {'suav':>5} {'k':>5} {'hold':>5}   " +
          "  ".join(f"{v[:6]:>13s}" for v in VOLCANOES) + "     global")
    mejor, rows = None, []
    for sm in SMOOTH_GRID:
        for k in K_GRID:
            for hold in HOLD_GRID:
                por_v, todo_e, todo_o = {}, [], []
                for v, ds in datos.items():
                    e = np.array([coda_duration(d["seg"], d["sr"], PRE_S, k,
                                                hold_s=hold, smooth_s=sm)
                                  for d in ds])
                    o = np.array([d["duracion_analista"] for d in ds])
                    m = np.isfinite(e)
                    if m.sum() < 20:
                        por_v[v] = (np.nan, np.nan, np.nan)
                        continue
                    from scipy.stats import spearmanr
                    rho = float(spearmanr(e[m], o[m]).statistic)
                    r = e[m] / o[m]
                    por_v[v] = (rho, float(np.median(r)),
                                float(((r > 0.7) & (r < 1.43)).mean()))
                    todo_e.append(e[m] / np.median(r))   # razon normalizada
                    todo_o.append(o[m])
                    rows.append(dict(smooth_s=sm, k=k, hold_s=hold, volcano=v,
                                     rho=rho, razon=float(np.median(r)),
                                     dentro30=por_v[v][2],
                                     medibles=float(m.mean())))
                rho_g = float(np.nanmean([por_v[v][0] for v in VOLCANOES]))
                d30_g = float(np.nanmean([por_v[v][2] for v in VOLCANOES]))
                score = rho_g + d30_g
                if mejor is None or score > mejor[0]:
                    mejor = (score, sm, k, hold, rho_g, d30_g)
                print(f"  {sm:5.1f} {k:5.2f} {hold:5.1f}   " + "  ".join(
                    f"{por_v[v][0]:+.2f}/{por_v[v][2]*100:3.0f}%" for v in VOLCANOES) +
                    f"     rho {rho_g:+.3f}  {d30_g*100:3.0f}%")
    print(f"\n  mejor combinacion: suavizado {mejor[1]} s, k {mejor[2]}, "
          f"hold {mejor[3]} s  ->  rho medio {mejor[4]:+.3f}, "
          f"dentro de +-30% {mejor[5]*100:.0f}%")
    if rows:
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
