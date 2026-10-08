"""
Los 16 descriptores, en la ventana fija, para las clases de los catalogos de
validacion que no tienen onset ni fin de coda: tremor y ruido de VOISS-Net,
tremor del SGC (Ruiz) y NTR de Isla Decepcion.

Sin onset, la ventana fija de 10 s se centra en el tramo etiquetado:
VOISS-Net, la ventana de 120 s; SGC, el registro completo; Decepcion, el
segmento [a, b] del archivo de etiquetas. El "onset" que se pasa a
extract_picks_16.fija es ese centro menos 5 s, de modo que la ventana, el
filtro y los descriptores son los mismos que en LP y VT. No hay ventana hasta
la coda. La SNR no se guarda: su ventana de ruido cae dentro de la propia
senal continua.

Decisiones (2026-10-06, con Sergio): ventana fija centrada, sin corte manual,
para las clases continuas; las explosiones (EX) si se cortan a mano.

Uso:
    python scripts/extract_continuas.py [--workers 1]
"""
from __future__ import annotations

import argparse
import json
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
from extract_picks_16 import fija  # noqa: E402

DESTINO = ROOT / "data" / "processed" / "features_manual" / "features16_continuas.parquet"
MEDIA = C.WINDOW_SAMPLES / 100.0 / 2          # 5 s


def items() -> list[dict]:
    c = pd.read_csv(CT.COLA, dtype=str, keep_default_na=False)
    c = c[c.clase.isin(["TR", "RUIDO"]) & c.dataset.isin(["voissnet", "sgc"])]
    out = c.to_dict("records")
    from extract_deception import parse_mlf
    for db in CT.DECEPCION_DBS:
        mlf = CT.DECEPCION / f"{db}.mlf"
        if not mlf.exists():
            continue
        for f, a, b, lab in parse_mlf(mlf):
            if lab == "NTR" and (CT.DECEPCION / db / "raw" / f).exists():
                out.append(dict(item_id=f"decepcion:{db}/{f}:{a:.4f}-{b:.4f}",
                                dataset="decepcion", sistema="Decepcion", clase=lab,
                                estacion=db, fuente=f"{db}/raw/{f}",
                                extra=json.dumps(dict(a=a, b=b))))
    return out


def sistema(r: dict) -> str:
    if r["dataset"] == "sgc":
        return "Ruiz-Olleta" if r["estacion"] in ("OLLZ", "OLLE") else "Ruiz-Bis"
    if r["dataset"] == "decepcion":
        return "Decepcion" + r["estacion"][4:6]            # dec.95M -> 95
    return r["sistema"]


def una(r: dict) -> dict:
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    base = dict(item_id=r["item_id"], dataset=r["dataset"], sistema=sistema(r),
                clase=r["clase"], estacion=r["estacion"],
                clase_corpus=json.loads(r["extra"]).get("label", r["clase"]))
    try:
        t = CT.cargar(r)
    except Exception as e:                              # noqa: BLE001
        return dict(base, motivo=f"lectura: {type(e).__name__}")
    if t["etiqueta"] is not None:
        a, b = t["etiqueta"]
    else:                                               # SGC: el registro entero
        a, b = 0.0, len(t["x"]) / t["sr"]
    centro = (a + b) / 2
    x = a_100hz(np.nan_to_num(t["x"]), t["sr"])
    base.update(centro=centro, largo_etiqueta=b - a,
                ventana_excede_etiqueta=bool(b - a < 2 * MEDIA))
    try:
        f = fija(x, centro - MEDIA)
    except C.ExclusionError as e:
        return dict(base, motivo=f"ventana: {e}")
    except Exception as e:                              # noqa: BLE001
        return dict(base, motivo=f"calculo: {type(e).__name__}")
    f["snr_db"] = np.nan
    return dict(base, motivo="", **f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    trabajo = items()
    print(f"{len(trabajo)} tramos", flush=True)
    filas = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(una, trabajo, chunksize=16), 1):
            filas.append(r)
            if k % 500 == 0:
                print(f"  {k}/{len(trabajo)}", flush=True)
    d = pd.DataFrame(filas)
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(DESTINO, index=False)
    ok = d.motivo == ""
    print(d.assign(ok=ok).groupby(["sistema", "clase"]).ok.agg(["sum", "size"]).to_string())
    print(d.loc[~ok, "motivo"].value_counts().to_string())
    print(f"-> {DESTINO}")


if __name__ == "__main__":
    main()
