"""
Resume la curva de presupuesto de etiquetas en un CSV chico.

presupuesto_etiquetas.py deja un parquet por trabajo (fuente__destino__rep)
en experiments/reports/presupuesto/ del proyecto de analisis. Este script los
junta y promedia sobre repeticiones, por destino, fuente, modelo, seleccion y
n (etiquetas del destino). Cada destino queda por separado.

Uso (desde la raiz del proyecto completo, o indicando las rutas):
    python scripts/resumir_presupuesto.py \
        --entrada experiments/reports/presupuesto \
        --salida  <repo del paper>/analisis/resultados/presupuesto_resumen.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CLAVES = ["destino", "fuente", "modelo", "seleccion", "n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default="experiments/reports/presupuesto")
    ap.add_argument("--salida", default="presupuesto_resumen.csv")
    a = ap.parse_args()
    archivos = sorted(Path(a.entrada).glob("*.parquet"))
    if not archivos:
        raise SystemExit(f"sin parquets en {a.entrada}")
    d = pd.concat([pd.read_parquet(f) for f in archivos], ignore_index=True)
    r = (d.groupby(CLAVES)
         .agg(gm=("gm", "mean"), gm_sd=("gm", "std"),
              rec_LP=("rec_LP", "mean"), rec_VT=("rec_VT", "mean"),
              n_reps=("rep", "nunique"))
         .reset_index())
    r.to_csv(a.salida, index=False)
    print(f"{len(archivos)} trabajos, {len(r)} filas -> {a.salida}")


if __name__ == "__main__":
    main()
