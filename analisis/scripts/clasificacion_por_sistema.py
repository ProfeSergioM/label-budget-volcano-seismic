"""
Funciona el clasificador dentro de cada sistema, con todas sus clases?

En cada sistema, todas las clases con al menos SOPORTE eventos (globales y
locales: HB, AV, VLP, TO, tremor, EX, ruido, NTR, RE, OT...) entran a un
clasificador multiclase sobre los 16 descriptores, evaluado con validacion
cruzada estratificada de 5 pliegues, repetida REPS veces sobre sorteos de
hasta N_MAX eventos por clase. Dos modelos: regresion logistica L2
(estandarizada con el pliegue de entrenamiento) y XGBoost con una
configuracion fija moderada (sin ajuste por sistema, para que sea un piso).

Metricas por sistema: exactitud balanceada, Gm de los recalls, recall y
precision por clase y matriz de confusion acumulada. Cada sistema por
separado; nunca se juntan volcanes.

Uso:
    python scripts/clasificacion_por_sistema.py [--ventana fija|coda]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from heterogeneidad_clases import datos  # noqa: E402
from ranking_transferibilidad import F16, SOPORTE  # noqa: E402

OUT = ROOT / "experiments" / "reports"
N_MAX, REPS, SEMILLA = 300, 5, 41


def modelo(tipo, seed):
    if tipo == "log":
        return LogisticRegression(C=1.0, max_iter=5000)
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, tree_method="hist", n_jobs=1,
                         random_state=seed, verbosity=0)


def cv(g: pd.DataFrame, tipo: str, seed: int):
    clases = sorted(g.clase.unique())
    x = g[F16].to_numpy(float)
    y = np.array([clases.index(c) for c in g.clase])
    pred = np.zeros_like(y)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        xa, xb = x[a], x[b]
        if tipo == "log":
            med = np.nanmedian(xa, 0)
            xa, xb = np.where(np.isnan(xa), med, xa), np.where(np.isnan(xb), med, xb)
            mu, sd = xa.mean(0), xa.std(0) + 1e-9
            xa, xb = (xa - mu) / sd, (xb - mu) / sd
        pred[b] = modelo(tipo, seed).fit(xa, y[a]).predict(xb)
    return clases, confusion_matrix(y, pred, labels=range(len(clases)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ventana", default="fija", choices=["fija", "coda"])
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    d = datos(a.ventana)
    filas, por_clase, conf = [], [], []
    for (org, sis), g in d.groupby(["origen", "sistema"]):
        n = g.clase.value_counts()
        ok = n[n >= SOPORTE].index
        if len(ok) < 2:
            continue
        for tipo in ("log", "xgb"):
            cm_tot = None
            for r in range(REPS):
                s = pd.concat([g[g.clase == c].sample(min(N_MAX, n[c]), random_state=SEMILLA + r)
                               for c in ok])
                clases, cm = cv(s, tipo, SEMILLA + r)
                cm_tot = cm if cm_tot is None else cm_tot + cm
            rec = np.diag(cm_tot) / cm_tot.sum(1)
            prec = np.diag(cm_tot) / np.maximum(cm_tot.sum(0), 1)
            filas.append(dict(origen=org, sistema=sis, modelo=tipo, n_clases=len(clases),
                              clases=" ".join(clases), bal_acc=float(rec.mean()),
                              gm=float(np.prod(rec) ** (1 / len(rec))),
                              azar=1 / len(clases)))
            for c, rc, pc in zip(clases, rec, prec):
                por_clase.append(dict(origen=org, sistema=sis, modelo=tipo, clase=c,
                                      n=int(n[c]), recall=float(rc), precision=float(pc)))
            for i, cv_ in enumerate(clases):
                for j, cp_ in enumerate(clases):
                    conf.append(dict(origen=org, sistema=sis, modelo=tipo, verdadera=cv_,
                                     predicha=cp_, n=int(cm_tot[i, j]),
                                     fraccion_fila=float(cm_tot[i, j] / cm_tot[i].sum())))
    r, pc = pd.DataFrame(filas), pd.DataFrame(por_clase)
    pd.set_option("display.width", 200)
    print(r.round(3).to_string(index=False))
    print(pc[pc.modelo == "xgb"].pivot_table(index=["origen", "sistema"], columns="clase",
                                             values="recall").round(2).to_string())
    r.to_csv(OUT / f"clasificacion_por_sistema_{a.ventana}.csv", index=False)
    pc.to_csv(OUT / f"clasificacion_por_sistema_clases_{a.ventana}.csv", index=False)
    pd.DataFrame(conf).to_csv(OUT / f"clasificacion_por_sistema_confusion_{a.ventana}.csv",
                              index=False)


if __name__ == "__main__":
    main()
