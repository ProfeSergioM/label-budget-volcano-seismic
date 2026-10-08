"""
Los 16 descriptores con el corte AUTOMATICO en lugar del manual.

Mismas funciones y ventanas que extract_picks_16.py (fija 10 s desde el
onset; coda del onset al fin), misma lectura de la traza (cortar_trazas.cargar
con la estacion de la marca) y mismo remuestreo a 100 Hz, pero con
auto_onset y auto_fin de experiments/picks/picks_manuales.csv (la propuesta
que vio el analista: onset del catalogo o STA/LTA, fin por envolvente).

Eventos: los mismos que quedan validos con el corte manual (estado bueno o
dudoso, sin duplicados, clase LP o VT, sin motivo de exclusion en alguna de
las dos ventanas manuales). Si falta auto_onset el evento queda con motivo;
si falta auto_fin, solo la ventana coda queda con motivo.

Uso (fija primero; coda es sensibilidad):
    nice -n 15 python scripts/extract_picks_auto.py --ventana fija --workers 1
    nice -n 15 python scripts/extract_picks_auto.py --ventana coda --workers 1
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
from extract_picks import a_100hz  # noqa: E402
from extract_picks_16 import DESTINO as MANUAL, coda, fija  # noqa: E402

OUT = ROOT / "data" / "processed" / "features_manual"
DESTINO = {"fija": OUT / "features16_auto_fija.parquet",
           "coda": OUT / "features16_auto_coda.parquet"}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def una(args) -> dict:
    f, item, ventana = args
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    base = dict(item_id=f["item_id"], guardado_utc=f["guardado_utc"],
                auto_onset=_num(f["auto_onset"]), auto_fin=_num(f["auto_fin"]))
    on, fn = base["auto_onset"], base["auto_fin"]
    if np.isnan(on) or (ventana == "coda" and (np.isnan(fn) or fn <= on)):
        return dict(base, motivo="sin corte automatico")
    try:
        t = CT.cargar(item)
    except Exception as e:                              # noqa: BLE001
        return dict(base, motivo=f"lectura: {type(e).__name__}")
    x = a_100hz(np.nan_to_num(t["x"]), t["sr"])
    try:
        r = fija(x, on) if ventana == "fija" else coda(x, on, fn)
        return dict(base, motivo="", **r)
    except C.ExclusionError as e:
        return dict(base, motivo=f"ventana: {e}")
    except Exception as e:                              # noqa: BLE001
        return dict(base, motivo=f"calculo: {type(e).__name__}")


def eventos() -> pd.DataFrame:
    """Marcas de los eventos LP/VT validos con el corte manual."""
    ok = set()
    for ruta in MANUAL.values():
        m = pd.read_parquet(ruta)
        ok |= set(m.loc[(m.motivo == "") & ~m.duplicado.fillna(False).astype(bool)
                        & m.clase.isin(["LP", "VT"]), "item_id"])
    picks = pd.read_csv(CT.PICKS, dtype=str, keep_default_na=False)
    picks = picks.drop_duplicates("item_id", keep="last")
    picks = picks[picks.estado.isin(["bueno", "dudoso"]) & picks.item_id.isin(ok)]
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ventana", choices=["fija", "coda"], required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    picks = eventos()
    ruta = DESTINO[a.ventana]
    previo = pd.read_parquet(ruta) if ruta.exists() else pd.DataFrame()
    if len(previo):
        vig = picks.set_index("item_id")
        previo = previo[previo.item_id.isin(vig.index)]
        previo = previo[previo.item_id.map(vig.guardado_utc) == previo.guardado_utc]
    pend = picks[~picks.item_id.isin(set(previo.item_id) if len(previo) else set())]
    print(f"ventana {a.ventana}: {len(picks)} eventos; {len(picks) - len(pend)} ya calculados, "
          f"{len(pend)} por calcular", flush=True)

    items = pd.read_csv(CT.COLA, dtype=str, keep_default_na=False).set_index("item_id")
    trabajos = [(f, items.loc[f["item_id"]].to_dict() | {"item_id": f["item_id"]}
                 | ({"estacion": f["estacion"]} if f.get("estacion") else {}), a.ventana)
                for f in pend.to_dict("records")]
    meta = pd.read_parquet(MANUAL["fija"])[["item_id", "dataset", "sistema", "clase",
                                            "clase_corpus", "subtipo", "estacion", "onset",
                                            "fin", "duplicado"]]

    def guardar(nuevas):
        d = pd.concat([previo, pd.DataFrame(nuevas)], ignore_index=True)
        d = meta.merge(d.drop(columns=[c for c in meta.columns if c != "item_id" and c in d]),
                       on="item_id", how="right")
        d.to_parquet(ruta, index=False)
        return d

    nuevas = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(una, trabajos, chunksize=8), 1):
            nuevas.append(r)
            if k % 500 == 0:
                print(f"  {k}/{len(trabajos)}", flush=True)
                guardar(nuevas)          # guardado parcial: el calculo es incremental
    d = guardar(nuevas)
    print(d.assign(ok=d.motivo == "").groupby(["sistema", "clase"]).ok.agg(["sum", "size"])
          .to_string())
    malos = d.loc[d.motivo != "", "motivo"].value_counts()
    if len(malos):
        print(malos.to_string())
    print(f"-> {ruta}")


if __name__ == "__main__":
    main()
