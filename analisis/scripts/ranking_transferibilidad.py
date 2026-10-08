"""
Ranking de transferibilidad de los 16 descriptores (Methods, "Descriptor
transferability"): en cada sistema con LP y VT sobre el criterio de soporte
(50 eventos), el AUC de LP contra VT por descriptor; el AUC se orienta por el
signo de su mediana entre sistemas, y el ranking usa la mediana del AUC
orientado menos su rango intercuartil.

Las dos ventanas se comparan sobre los mismos eventos: en OVDAS, los de la
ventana hasta la coda (Hudson completo, el resto en sus muestras), y en los
catalogos de validacion, los que tienen descriptores en ambas ventanas.
Nunca se juntan sistemas: cada AUC es de un sistema.

Uso:
    python scripts/ranking_transferibilidad.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "reports"
F16 = ["DET", "L", "Lmax", "DIV", "Lentr", "LAM", "TT", "Vmax", "Ventr", "W",
       "Wmax", "Wdiv", "Wentr", "fpeak", "SR", "FI"]
SOPORTE = 50
MAX_POR_CLASE = 5000        # el AUC no depende del tamano; acota Villarrica
SEMILLA = 41
OVDAS = {"lagmaule": ("Laguna del Maule", "_pool10"),
         "nevchillan": ("Nevados de Chillan", "_pool10"),
         "copahue": ("Copahue", "_pool10"),
         "villarrica": ("Villarrica", "_pool100"),
         "hudson": ("Hudson", "")}


def ovdas() -> dict[str, pd.DataFrame]:
    out = {"fija": [], "coda": []}
    for key, (nombre, pool) in OVDAS.items():
        c = pd.read_parquet(ROOT / f"data/processed/features_dur/features_{key}{pool}.parquet")
        f = pd.read_parquet(ROOT / f"data/processed/features/features_{key}.parquet")
        ids = set(c.event_id) & set(f.event_id)
        for w, d in (("fija", f), ("coda", c)):
            d = d[d.event_id.isin(ids)].drop_duplicates("event_id")
            out[w].append(d.assign(sistema=nombre, origen="OVDAS", clase=d.label)
                          [["sistema", "origen", "clase"] + F16])
    return {w: pd.concat(v, ignore_index=True) for w, v in out.items()}


def validacion() -> dict[str, pd.DataFrame]:
    base = ROOT / "data/processed/features_manual"
    d = {w: pd.read_parquet(base / f"features16_{w}.parquet") for w in ("fija", "coda")}
    ok = {w: set(x.loc[(x.motivo == "") & ~x.duplicado.astype(bool), "item_id"])
          for w, x in d.items()}
    ids = ok["fija"] & ok["coda"]
    out = {}
    for w, x in d.items():
        x = x[x.item_id.isin(ids)].copy()
        dec = x.sistema == "Decepcion"
        x.loc[dec, "sistema"] = "Decepcion" + x.loc[dec, "item_id"].str.extract(r"dec\.(\d\d)")[0]
        out[w] = x.assign(origen=x.dataset)[["sistema", "origen", "clase"] + F16]
    return out


def aucs(d: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for (sis, org), g in d.groupby(["sistema", "origen"]):
        n = g.clase.value_counts()
        if n.get("LP", 0) < SOPORTE or n.get("VT", 0) < SOPORTE:
            continue
        g = pd.concat([h.sample(min(len(h), MAX_POR_CLASE), random_state=SEMILLA)
                       for c, h in g[g.clase.isin(["LP", "VT"])].groupby("clase")])
        for f in F16:
            v = g[[f, "clase"]].dropna()
            if v.clase.nunique() < 2:
                continue
            filas.append(dict(sistema=sis, origen=org, descriptor=f,
                              n_lp=int((v.clase == "LP").sum()), n_vt=int((v.clase == "VT").sum()),
                              auc=roc_auc_score(v.clase == "LP", v[f])))
    return pd.DataFrame(filas)


def ranking(a: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for f, g in a.groupby("descriptor"):
        signo = 1 if np.median(g.auc) >= 0.5 else -1
        o = g.auc if signo > 0 else 1 - g.auc
        q1, med, q3 = np.percentile(o, [25, 50, 75])
        filas.append(dict(descriptor=f, sentido="LP>VT" if signo > 0 else "LP<VT",
                          auc_mediana=med, iqr=q3 - q1, puntaje=med - (q3 - q1),
                          auc_min=o.min(), sistema_min=g.sistema.iloc[int(np.argmin(o))],
                          invertidos=int((o < 0.5).sum()), n_sistemas=len(g)))
    return pd.DataFrame(filas).sort_values("puntaje", ascending=False).reset_index(drop=True)


def main():
    ov, va = ovdas(), validacion()
    todo_a, todo_r = [], []
    for w in ("fija", "coda"):
        a = aucs(pd.concat([ov[w], va[w]], ignore_index=True)).assign(ventana=w)
        r = ranking(a).assign(ventana=w)
        todo_a.append(a)
        todo_r.append(r)
        print(f"\n=== ventana {w}: {a.sistema.nunique()} sistemas ===")
        print(", ".join(sorted(a.sistema.unique())))
        print(r.drop(columns="ventana").round(3).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(todo_a).to_csv(OUT / "transferibilidad_auc.csv", index=False)
    pd.concat(todo_r).to_csv(OUT / "transferibilidad_ranking.csv", index=False)
    print(f"\n-> {OUT / 'transferibilidad_ranking.csv'}")


if __name__ == "__main__":
    sys.exit(main())
