"""
Confident learning entre datasets: las etiquetas de un dataset destino vistas
con el criterio aprendido en otro dataset (fuente).

sesgo_experto.py audita cada sistema con su propio criterio (probabilidades
fuera de muestra del mismo sistema). Aqui, para cada par fuente -> destino:
  - el destino aporta hasta N_DESTINO eventos por clase (al azar);
  - "propio": probabilidades fuera de muestra del destino (5 pliegues), como
    en sesgo_experto.py;
  - "fuente": probabilidades de un modelo entrenado con hasta N_FUENTE
    eventos por clase de la fuente;
  - en ambos casos cleanlab (find_label_issues, prune_by_noise_rate) marca
    las etiquetas del destino que no calzan con ese criterio, y
    estimate_joint da la matriz conjunta etiqueta dada x etiqueta estimada.
Clases: LP, VT y HB, las que tengan al menos SOPORTE eventos en ambos
sistemas (HB solo entre Hudson, Copahue, Ruiz-Olleta y Ruiz-Bis); pares con
al menos dos clases comunes. Los umbrales por clase de cleanlab absorben un
corrimiento global de las probabilidades; lo que queda por sobre lo
"propio" es desacuerdo de criterio o corrimiento propio de esa clase.

Mismo XGBoost de configuracion fija que sesgo_experto.py, pesos balanceados,
descriptores sin escalar (XGBoost es invariante a la escala). Mismos eventos
y ventana fija que transferencia.py. Cada destino se reporta por separado.

Uso:
    nice -n 10 python scripts/sesgo_entre_datasets.py [--reps 3] [--workers 3]
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
import transferencia as T  # noqa: E402
from ranking_transferibilidad import F16, SOPORTE  # noqa: E402

OUT = ROOT / "experiments" / "reports"
CLASES = ["LP", "VT", "HB"]
N_FUENTE = 1000
N_DESTINO = 2000
SEMILLA = 41
DATOS = {}


def xgb(seed):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, tree_method="hist", n_jobs=1,
                         random_state=seed, verbosity=0)


def pesos(y):
    u, c = np.unique(y, return_counts=True)
    w = dict(zip(u, len(y) / (len(u) * c)))
    return np.array([w[k] for k in y])


def tope(g, n, seed):
    return pd.concat([h.sample(min(len(h), n), random_state=seed) for _, h in g.groupby("clase")])


def una(args):
    fuente, destino, clases, rep = args
    clases = sorted(clases)          # searchsorted necesita el orden
    warnings.filterwarnings("ignore")
    from cleanlab.count import estimate_joint
    from cleanlab.filter import find_label_issues
    from sklearn.model_selection import StratifiedKFold
    d = DATOS["fija"]
    seed = SEMILLA + rep
    src = tope(d[(d.sistema == fuente) & d.clase.isin(clases)], N_FUENTE, seed)
    dst = tope(d[(d.sistema == destino) & d.clase.isin(clases)], N_DESTINO, seed)
    y = np.searchsorted(clases, dst.clase.to_numpy())
    x = dst[F16].to_numpy(float)
    ys = np.searchsorted(clases, src.clase.to_numpy())
    probs = {"fuente": xgb(seed).fit(src[F16].to_numpy(float), ys, sample_weight=pesos(ys))
             .predict_proba(x)}
    p = np.zeros((len(y), len(clases)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        p[te] = xgb(seed).fit(x[tr], y[tr], sample_weight=pesos(y[tr])).predict_proba(x[te])
    probs["propio"] = p
    filas, conj = [], []
    for criterio, p in probs.items():
        m = find_label_issues(y, p, filter_by="prune_by_noise_rate")
        j = estimate_joint(y, p)
        pred = p.argmax(1)
        for i, c in enumerate(clases):
            mc = y == i
            fila = dict(fuente=fuente, destino=destino, rep=rep, criterio=criterio,
                        clases="-".join(clases), clase=c, n=int(mc.sum()),
                        recall=float(np.mean(pred[mc] == i)),
                        frac_sospechosos=float(m[mc].mean()))
            for k, c2 in enumerate(clases):
                fila[f"sugiere_{c2}"] = int((m & mc & (pred == k)).sum())
                conj.append(dict(fuente=fuente, destino=destino, rep=rep, criterio=criterio,
                                 clases="-".join(clases), etiqueta=c, estimada=c2,
                                 p=float(j[i, k])))
            filas.append(fila)
    return filas, conj


def init(datos):
    DATOS.update(datos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    datos = T.cargar()
    d = datos["fija"]
    n = d[d.clase.isin(CLASES)].groupby(["sistema", "clase"]).size().unstack(fill_value=0)
    trabajos = []
    for dst in n.index:
        for src in n.index:
            if src == dst:
                continue
            comunes = [c for c in CLASES if n.loc[src, c] >= SOPORTE and n.loc[dst, c] >= SOPORTE]
            if len(comunes) >= 2:
                trabajos += [(src, dst, comunes, r) for r in range(a.reps)]
    print(f"{len(trabajos)} trabajos", flush=True)
    filas, conj = [], []
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(datos,)) as ex:
        for k, (f, c) in enumerate(ex.map(una, trabajos), 1):
            filas += f
            conj += c
            if k % 20 == 0:
                print(f"  {k}/{len(trabajos)}", flush=True)
    f = pd.DataFrame(filas)
    f.to_csv(OUT / "sesgo_entre_datasets_pares.csv", index=False)
    pd.DataFrame(conj).to_csv(OUT / "sesgo_entre_datasets_conjunta.csv", index=False)
    # resumen por destino y clase: mediana sobre fuentes (promedio sobre reps)
    r = (f.groupby(["destino", "clases", "clase", "criterio", "fuente"])
         [["frac_sospechosos", "recall"]].mean()
         .groupby(["destino", "clases", "clase", "criterio"]).median().unstack("criterio"))
    r.columns = [f"{a}_{b}" for a, b in r.columns]
    r = r.reset_index()
    r.to_csv(OUT / "sesgo_entre_datasets_resumen.csv", index=False)
    print(r.round(3).to_string())


if __name__ == "__main__":
    main()
