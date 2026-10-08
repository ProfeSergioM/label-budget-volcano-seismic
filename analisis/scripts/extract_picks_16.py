"""
Los 16 descriptores del documento (13 de recurrencia en el dominio del
tiempo, fpeak, SR y FI) para los corpus externos, en las dos ventanas que
compara el estudio, con exactamente las mismas funciones que OVDAS:

    fija   10 s desde el onset (rqa_core.padded_window_indices, filtro
           FILTER_BAND), como download_and_extract.extract_event_row.
    coda   del onset al fin de coda (rqa_core.duration_window_plan, filtro
           DURATION_FILTER_BAND), como extract_duration.extract_row.

El onset y el fin son los del corte manual (experiments/picks); la traza se
lee igual que la pagina de corte (cortar_trazas.cargar, con la estacion de
la marca) y se lleva a 100 Hz antes de filtrar, porque m y tau del embedding
estan en muestras.

Es incremental, como extract_picks.py: una traza se recalcula solo si su
marca cambio. Las columnas de duplicados y fuera de ventana se toman de
extract_picks.py.

Uso:
    python scripts/extract_picks_16.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cortar_trazas as CT  # noqa: E402
import rqa_core as C  # noqa: E402
from extract_picks import SUBTIPOS, a_100hz  # noqa: E402

OUT = ROOT / "data" / "processed" / "features_manual"
DESTINO = {"fija": OUT / "features16_fija.parquet", "coda": OUT / "features16_coda.parquet"}
SPS = 100.0
META = ROOT / "data" / "processed" / "features_manual" / "features_manual.parquet"


def segmento(x: np.ndarray, onset: float, pre_s: float, post_s: float, n: int):
    i0 = int(round((onset - pre_s) * SPS))
    if i0 < 0 or i0 + n > len(x):
        return None
    return x[i0:i0 + n]


def fila(window: np.ndarray, snr_filt: np.ndarray, idx: dict) -> dict:
    xz = C.zscore(window)
    rqa, eps, rr = C.compute_rqa_domain(xz, C.M_TIME, C.TAU_TIME)
    spec, _ = C.compute_spectral_features(window, SPS)
    noise = snr_filt[idx["noise_start"]:idx["noise_end"]]
    sig = snr_filt[idx["signal_start"]:idx["signal_end"]]
    snr = (C.compute_snr_db(noise, sig)
           if len(noise) == idx["noise_end"] - idx["noise_start"] else float("nan"))
    return dict(snr_db=snr, eps_time=eps, rr_time=rr, **rqa, **spec)


def fija(x: np.ndarray, onset: float) -> dict:
    idx = C.padded_window_indices(SPS)
    raw = segmento(x, onset, C.PADDED_PRE_S, C.PADDED_POST_S, idx["total_samples"])
    if raw is None:
        raise C.ExclusionError("la traza no cubre el segmento de la ventana fija")
    w = C.preprocess_segment(raw, SPS)[idx["signal_start"]:idx["signal_end"]]
    if len(w) != C.WINDOW_SAMPLES:
        raise C.ExclusionError("ventana fija incompleta")
    return dict(fila(w, C.preprocess_segment(raw, SPS, band=C.SNR_BAND), idx),
                n_window=len(w))


def coda(x: np.ndarray, onset: float, fin: float) -> dict:
    plan = C.duration_window_plan(fin - onset, SPS)
    raw = segmento(x, onset, plan["pre_s"], plan["post_s"], plan["total_samples"])
    if raw is None:
        raise C.ExclusionError("la traza no cubre el segmento de la ventana hasta la coda")
    w = C.preprocess_segment(raw, SPS, band=C.DURATION_FILTER_BAND)[
        plan["signal_start"]:plan["signal_end"]]
    if len(w) != plan["n_window"]:
        raise C.ExclusionError("ventana hasta la coda incompleta")
    return dict(fila(w, C.preprocess_segment(raw, SPS, band=C.SNR_BAND), plan),
                n_window=len(w), duration_s_used=len(w) / SPS)


def una(args) -> dict:
    f, item = args
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    base = dict(item_id=f["item_id"], guardado_utc=f["guardado_utc"])
    try:
        t = CT.cargar(item)
    except Exception as e:                              # noqa: BLE001
        m = f"lectura: {type(e).__name__}"
        return {"fija": dict(base, motivo=m), "coda": dict(base, motivo=m)}
    x = a_100hz(np.nan_to_num(t["x"]), t["sr"])
    on, fn = float(f["onset"]), float(f["fin"])
    out = {}
    for nombre, calc in (("fija", lambda: fija(x, on)), ("coda", lambda: coda(x, on, fn))):
        try:
            out[nombre] = dict(base, motivo="", **calc())
        except C.ExclusionError as e:
            out[nombre] = dict(base, motivo=f"ventana: {e}")
        except Exception as e:                          # noqa: BLE001
            out[nombre] = dict(base, motivo=f"calculo: {type(e).__name__}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todo", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    picks = pd.read_csv(CT.PICKS, dtype=str, keep_default_na=False)
    picks = picks.drop_duplicates("item_id", keep="last")
    picks = picks[picks.estado.isin(["bueno", "dudoso"])]
    items = pd.read_csv(CT.COLA, dtype=str, keep_default_na=False).set_index("item_id")
    vigente = picks.set_index("item_id").guardado_utc

    previo = {}
    for k, ruta in DESTINO.items():
        p = pd.read_parquet(ruta) if ruta.exists() and not a.todo else pd.DataFrame()
        if len(p):
            p = p[p.item_id.map(vigente) == p.guardado_utc]
        previo[k] = p
    hechos = set(previo["fija"].item_id) & set(previo["coda"].item_id) if len(previo["fija"]) else set()
    pend = picks[~picks.item_id.isin(hechos)]
    print(f"{len(picks)} marcas; {len(hechos)} ya calculadas, {len(pend)} por calcular", flush=True)

    trabajos = [(f, items.loc[f["item_id"]].to_dict() | {"item_id": f["item_id"]}
                 | ({"estacion": f["estacion"]} if f.get("estacion") else {}))
                for f in pend.to_dict("records")]
    nuevas = {"fija": [], "coda": []}
    if trabajos:
        with ProcessPoolExecutor(a.workers) as ex:
            for k, r in enumerate(ex.map(una, trabajos, chunksize=8), 1):
                nuevas["fija"].append(r["fija"])
                nuevas["coda"].append(r["coda"])
                if k % 500 == 0:
                    print(f"  {k}/{len(trabajos)}", flush=True)

    # sistema, clase, duplicados y demas marcas salen de extract_picks.py
    meta = pd.read_parquet(META)[["item_id", "dataset", "sistema", "clase", "clase_corpus",
                                  "subtipo", "estacion", "onset", "fin", "duplicado",
                                  "fuera_de_ventana", "ruido_pisa_evento"]]
    for k, ruta in DESTINO.items():
        d = pd.concat([previo[k], pd.DataFrame(nuevas[k])], ignore_index=True)
        d = d[d.item_id.isin(picks.item_id)]
        d = meta.merge(d.drop(columns=[c for c in meta.columns if c != "item_id" and c in d]),
                       on="item_id", how="right")
        d.to_parquet(ruta, index=False)
        ok = (d.motivo == "") & ~d.duplicado.fillna(False).astype(bool)
        t = d.assign(ok=ok).groupby(["sistema", "clase"]).ok.sum()
        print(f"\nventana {k}: {int(ok.sum())} eventos unicos con descriptores")
        print(t.to_string())
        malos = d.loc[d.motivo != "", "motivo"].value_counts()
        if len(malos):
            print(malos.to_string())
        print(f"-> {ruta}")


if __name__ == "__main__":
    main()
