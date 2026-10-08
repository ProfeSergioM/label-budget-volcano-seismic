"""
Cuantas etiquetas del volcan nuevo hacen falta (enfoque del paper desde
2026-10-08): curva de presupuesto de etiquetas y aprendizaje activo.

Para cada par fuente -> destino de la matriz LP-VT (ventana fija; mismos
eventos y descriptores que transferencia.py), y para cada repeticion:
  - el destino se parte en una prueba fija (la mitad de cada clase, hasta
    TOPE_PRUEBA por clase) y un pool de candidatos a etiquetar (el resto, con
    la composicion natural del destino);
  - se entrena con los 100 por clase de la fuente mas n eventos etiquetados
    del destino, n en PRESUPUESTO (total, no por clase), elegidos
      * al azar del pool (seleccion "azar"), o
      * por aprendizaje activo (seleccion "activo"): partiendo del modelo de
        la fuente, en lotes de LOTE se etiqueta el evento del pool con menor
        margen |p(LP) - 0.5| y se reentrena;
  - referencias: el mismo modelo solo con n eventos del destino al azar
    ("solo_destino") y el modelo local con hasta 100 por clase del pool
    ("local"), el techo que el presupuesto busca alcanzar; ambos con la
    configuracion que transferencia.py ajusto para el propio destino.
Tres clasificadores: regresion logistica y XGBoost con la configuracion que
transferencia.py ajusto para cada fuente, y TabPFN v2 con su configuracion
por defecto salvo un ensamble de 4 (n_estimators=4, por costo en CPU). El aprendizaje activo corre con logistica y XGBoost; TabPFN
solo con seleccion al azar. Pesos de clase balanceados en logistica y
XGBoost. Metricas sobre la prueba fija: recall de LP y VT y Gm.

Cada destino se reporta por separado (nunca se juntan volcanes); la mediana
sobre fuentes se calcula despues, por destino.

Uso:
    python scripts/presupuesto_etiquetas.py [--workers 3] [--reps 5]
"""
from __future__ import annotations

import argparse
import ast
import os
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

OUT = ROOT / "experiments" / "reports" / "presupuesto"
CLASES = ["LP", "VT"]
PRESUPUESTO = [0, 10, 20, 50, 100, 200]
LOTE = 10
TOPE_PRUEBA = 300
SEMILLA = 41
DATOS = {}


# ---------------------------------------------------------------- modelos --

def pesos(y):
    u, c = np.unique(y, return_counts=True)
    w = {k: len(y) / (len(u) * v) for k, v in zip(u, c)}
    return np.array([w[k] for k in y])


_TABPFN = {}


def tabpfn_cacheado(seed):
    """Un TabPFN por proceso: cargar los pesos en cada ajuste es lo lento."""
    if "m" not in _TABPFN:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        _TABPFN["m"] = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, device="cpu", random_state=seed, n_estimators=4)
    _TABPFN["m"].random_state = seed
    return _TABPFN["m"]


class Clf:
    def __init__(self, tipo, conf, seed):
        self.tipo, self.conf, self.seed = tipo, conf, seed

    def fit(self, x, y):
        yi = (y == "VT").astype(int)
        if len(np.unique(yi)) < 2:
            self.const = yi[0]
            return self
        self.const = None
        self.med = np.nanmedian(x, 0)
        if self.tipo == "xgb":
            from xgboost import XGBClassifier
            self.m = XGBClassifier(**self.conf, tree_method="hist", n_jobs=1,
                                   random_state=self.seed, verbosity=0)
            self.m.fit(x, yi, sample_weight=pesos(yi))
        elif self.tipo == "log":
            from sklearn.linear_model import LogisticRegression
            xx = np.where(np.isnan(x), self.med, x)
            self.mu, self.sd = xx.mean(0), xx.std(0) + 1e-9
            self.m = LogisticRegression(C=self.conf, max_iter=5000)
            self.m.fit((xx - self.mu) / self.sd, yi, sample_weight=pesos(yi))
        else:
            self.m = tabpfn_cacheado(self.seed)
            self.m.fit(np.where(np.isnan(x), self.med, x), yi)
        return self

    def proba_vt(self, x):
        if self.const is not None:
            return np.full(len(x), float(self.const))
        if self.tipo == "log":
            xx = (np.where(np.isnan(x), self.med, x) - self.mu) / self.sd
            return self.m.predict_proba(xx)[:, 1]
        if self.tipo == "tabpfn":
            return self.m.predict_proba(np.where(np.isnan(x), self.med, x))[:, 1]
        return self.m.predict_proba(x)[:, 1]


def puntuar(p_vt, y):
    pred = np.where(p_vt >= 0.5, "VT", "LP")
    r = {f"rec_{c}": float(np.mean(pred[y == c] == c)) for c in CLASES}
    r["gm"] = float(np.sqrt(r["rec_LP"] * r["rec_VT"]))
    return r


# ---------------------------------------------------------------- trabajo --

def particion(g, rep):
    """Prueba fija (mitad de cada clase, hasta TOPE_PRUEBA) y pool."""
    prueba, pool = [], []
    for c in CLASES:
        h = g[g.clase == c].sample(frac=1, random_state=SEMILLA + rep)
        k = min(len(h) // 2, TOPE_PRUEBA)
        prueba.append(h.iloc[:k])
        pool.append(h.iloc[k:])
    return pd.concat(prueba), pd.concat(pool).sample(frac=1, random_state=SEMILLA + 100 + rep)


def una(args):
    fuente, destino, rep, confs, confs_dst = args
    warnings.filterwarnings("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    d = DATOS["fija"]
    src = d[(d.sistema == fuente) & d.clase.isin(CLASES)]
    dst = d[(d.sistema == destino) & d.clase.isin(CLASES)]
    tr_src = T.sorteo(src, CLASES, T.N_POR_CLASE, SEMILLA + rep)
    prueba, pool = particion(dst, rep)
    xp, yp = prueba[F16].to_numpy(float), prueba.clase.to_numpy()
    xs, ys = tr_src[F16].to_numpy(float), tr_src.clase.to_numpy()
    xq, yq = pool[F16].to_numpy(float), pool.clase.to_numpy()
    filas = []

    def fila(modelo, seleccion, n, m):
        filas.append(dict(fuente=fuente, destino=destino, rep=rep, modelo=modelo,
                          seleccion=seleccion, n=n, **puntuar(m.proba_vt(xp), yp)))

    for tipo in ("log", "xgb", "tabpfn"):
        conf = confs.get(tipo)
        # fuente + n al azar (pool en orden fijo: los primeros n)
        for n in PRESUPUESTO:
            n = min(n, len(pool))
            x = np.vstack([xs, xq[:n]])
            y = np.concatenate([ys, yq[:n]])
            fila(tipo, "azar", n, Clf(tipo, conf, SEMILLA + rep).fit(x, y))
        # solo destino, n al azar
        for n in PRESUPUESTO[1:]:
            n = min(n, len(pool))
            fila(tipo, "solo_destino", n,
                 Clf(tipo, confs_dst.get(tipo), SEMILLA + rep).fit(xq[:n], yq[:n]))
        # local: hasta 100 por clase del pool, estratificado
        loc = pd.concat([pool[pool.clase == c].iloc[:T.N_POR_CLASE] for c in CLASES])
        fila(tipo, "local", len(loc), Clf(tipo, confs_dst.get(tipo), SEMILLA + rep).fit(
            loc[F16].to_numpy(float), loc.clase.to_numpy()))
        # aprendizaje activo (logistica y XGBoost)
        if tipo == "tabpfn":
            continue
        elegidos = np.zeros(len(pool), bool)
        m = Clf(tipo, conf, SEMILLA + rep).fit(xs, ys)
        fila(tipo, "activo", 0, m)
        for n in range(LOTE, max(PRESUPUESTO) + 1, LOTE):
            libres = np.flatnonzero(~elegidos)
            if not len(libres):
                break
            margen = np.abs(m.proba_vt(xq[libres]) - 0.5)
            elegidos[libres[np.argsort(margen)[:LOTE]]] = True
            m = Clf(tipo, conf, SEMILLA + rep).fit(
                np.vstack([xs, xq[elegidos]]), np.concatenate([ys, yq[elegidos]]))
            if n in PRESUPUESTO:
                fila(tipo, "activo", n, m)
    return filas


def init(datos):
    DATOS.update(datos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--solo", nargs="*", help="limita a estos destinos (prueba)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    datos = T.cargar()
    d = datos["fija"]
    n = d[d.clase.isin(CLASES)].groupby(["sistema", "clase"]).size().unstack(fill_value=0)
    sistemas = sorted(n[(n.LP >= SOPORTE) & (n.VT >= SOPORTE)].index)
    pares = pd.read_parquet(ROOT / "experiments/reports/transferencia_pares.parquet")
    conf = (pares[(pares.matriz == "M2") & (pares.ventana == "fija") & (pares.rep == 0)
                  & (pares.escalado == "fuente")].drop_duplicates("fuente")
            .set_index("fuente"))
    trabajos = []
    for dst in sistemas:
        if a.solo and dst not in a.solo:
            continue
        for src in sistemas:
            if src == dst:
                continue
            confs, confs_dst = (dict(xgb=ast.literal_eval(conf.loc[s, "conf_xgb"]),
                                     log=float(conf.loc[s, "conf_log"])) for s in (src, dst))
            for r in range(a.reps):
                ruta = OUT / f"{dst}__{src}__{r}.parquet".replace(" ", "_")
                if not ruta.exists():
                    trabajos.append((src, dst, r, confs, confs_dst))
    print(f"{len(sistemas)} sistemas con LP y VT; {len(trabajos)} trabajos", flush=True)
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(datos,)) as ex:
        for k, filas in enumerate(ex.map(una, trabajos), 1):
            f = pd.DataFrame(filas)
            r = f.iloc[0]
            f.to_parquet(OUT / f"{r.destino}__{r.fuente}__{r.rep}.parquet".replace(" ", "_"),
                         index=False)
            if k % 10 == 0:
                print(f"  {k}/{len(trabajos)}", flush=True)
    todo = pd.concat([pd.read_parquet(p) for p in OUT.glob("*.parquet")], ignore_index=True)
    todo.to_parquet(OUT.parent / "presupuesto_etiquetas.parquet", index=False)
    print(f"-> {OUT.parent / 'presupuesto_etiquetas.parquet'} ({len(todo)} filas)")


if __name__ == "__main__":
    main()
