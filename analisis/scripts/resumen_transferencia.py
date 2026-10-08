"""
Resumen de las matrices de transferencia (transferencia_pares.parquet).

Escalones de cada par fuente -> destino:
  0 diagonal         la fuente evaluada en sus eventos fuera del sorteo
  1 estacion         mismo volcan, otra estacion (Ruiz-Olleta <-> Ruiz-Bis)
  2 misma agencia    otro volcan etiquetado por la misma agencia
  3 entre agencias   distinta agencia que etiqueta
La agencia es la que dibuja las etiquetas: OVDAS, SGC, AVO (VCSEIS Alaska),
PNSN (Cascade), HVO (Hawai), NCEDC (Norte de California), el equipo de
VOISS-Net y el catalogo de Isla Decepcion.

Tres agregaciones, para que el orden de los escalones no dependa de cuantos
destinos tenga cada fuente (auditoria): mediana por par, mediana de las
medianas por fuente y mediana de las medianas por destino. Gm solo donde el
destino tiene todas las clases de la matriz; los destinos de una sola clase
van aparte, con su recall. El escalado "sistema" se resume solo sobre
destinos con todas las clases (fuera de ellos es un artefacto de la
composicion).

Uso:
    python scripts/resumen_transferencia.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "reports"
AGENCIA = {"Laguna del Maule": "OVDAS", "Nevados de Chillan": "OVDAS", "Copahue": "OVDAS",
           "Villarrica": "OVDAS", "Hudson": "OVDAS", "Ruiz-Olleta": "SGC", "Ruiz-Bis": "SGC",
           "Alaska": "AVO", "Cascade": "PNSN", "Hawái": "HVO", "Norte de California": "NCEDC",
           "Kilauea": "VOISS-Net", "Redoubt": "VOISS-Net", "Tanaga": "VOISS-Net",
           "Pavlof": "VOISS-Net", "Semisopochnoi": "VOISS-Net", "Veniaminof": "VOISS-Net",
           "Decepcion09": "Decepcion", "Decepcion95": "Decepcion"}
CLAVE = ["matriz", "ventana", "escalado", "modelo"]


def escalon(f, d):
    if f == d:
        return "0 diagonal"
    if {f, d} == {"Ruiz-Olleta", "Ruiz-Bis"}:
        return "1 estacion"
    return "2 misma agencia" if AGENCIA[f] == AGENCIA[d] else "3 entre agencias"


def main():
    d = pd.read_parquet(OUT / "transferencia_pares.parquet")
    g = d.groupby(CLAVE + ["fuente", "destino"], as_index=False).mean(numeric_only=True)
    sd = d.groupby(CLAVE + ["fuente", "destino"]).gm.std().rename("gm_sd").reset_index()
    g = g.merge(sd, on=CLAVE + ["fuente", "destino"])
    g["escalon"] = [escalon(f, t) for f, t in zip(g.fuente, g.destino)]
    diag = g[g.escalon == "0 diagonal"].set_index(CLAVE + ["destino"]).gm
    g["diag_destino"] = [diag.get(tuple(r[c] for c in CLAVE) + (r["destino"],), np.nan)
                         for _, r in g.iterrows()]
    g["perdida"] = g.gm - g.diag_destino
    g.to_csv(OUT / "transferencia_resumen_pares.csv", index=False)

    con_gm = g[g.gm.notna()]
    filas = []
    for k, h in con_gm.groupby(CLAVE + ["escalon"]):
        filas.append(dict(zip(CLAVE + ["escalon"], k), n_pares=len(h),
                          gm_por_par=h.gm.median(),
                          gm_por_fuente=h.groupby("fuente").gm.median().median(),
                          gm_por_destino=h.groupby("destino").gm.median().median(),
                          perdida_por_par=h.perdida.median()))
    r = pd.DataFrame(filas)
    r.to_csv(OUT / "transferencia_resumen_escalones.csv", index=False)

    una = g[g.gm.isna() & (g.escalon != "0 diagonal")].copy()
    una["recall"] = una[["rec_LP", "rec_VT"]].bfill(axis=1).iloc[:, 0]
    u = una[(una.matriz == "M2") & (una.escalado == "fuente")].pivot_table(
        index="destino", columns=["ventana", "modelo"], values="recall", aggfunc="median")
    u.to_csv(OUT / "transferencia_resumen_una_clase.csv")

    m1 = g[g.matriz == "M1"].groupby(CLAVE + ["escalon"])[
        ["gm", "rec_LP", "rec_VT", "rec_HB"]].median()
    m1.to_csv(OUT / "transferencia_resumen_m1.csv")

    pd.set_option("display.width", 220)
    print(r.round(3).to_string(index=False))
    print("\nDestinos de una sola clase (M2, escalado fuente), recall mediano:")
    print(u.round(2).to_string())
    print("\nM1:")
    print(m1.round(2).to_string())


if __name__ == "__main__":
    main()
