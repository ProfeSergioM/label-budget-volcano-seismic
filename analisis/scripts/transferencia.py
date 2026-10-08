"""
Matrices de transferencia (Methods, "Classification and transfer design").

Dos matrices por ventana:
  M1  LP-VT-HB. Fuentes: los sistemas con LP, VT y HB sobre el soporte
      (Copahue, Hudson, Ruiz-Olleta). Destinos: los sistemas con HB sobre el
      soporte (los tres mas Ruiz-Bis).
  M2  LP-VT. Fuentes: los sistemas con LP y VT sobre el soporte (cinco OVDAS,
      cuatro regiones VCSEIS, Ruiz-Olleta). Destinos: todo sistema con LP o VT
      sobre el soporte, incluidos los de una sola clase.

Dos clasificadores por fuente: XGBoost (Optuna TPE, N_TRIALS ensayos, CV
estratificada de 5 pliegues sobre un sorteo independiente de la fuente,
espacio de busqueda del documento) y regresion logistica L2 (C en una grilla
logaritmica 1e-3..1e3, misma CV). La configuracion elegida de una fuente
queda fija para sus REPS sorteos de entrenamiento (N_POR_CLASE por clase) y
todos sus destinos.

Dos escalados: "fuente" (XGBoost en escala nativa; la logistica estandariza
con media y varianza de la fuente) y "sistema" (cada sistema, fuente y
destino, con la mediana y el IQR de todos sus eventos, sin etiquetas; la
logistica ademas estandariza con la fuente). Para XGBoost el escalado por
sistema es monotono dentro de la fuente, asi que el ajuste de Optuna vale
para ambos.

Diagonal: en cada sorteo, el modelo se evalua en los eventos de la fuente que
no entraron al entrenamiento (submuestreo aleatorio repetido). Fuera de la
diagonal se evalua sobre todos los eventos del destino, con un tope de
TOPE_DESTINO por clase (muestra fija) para acotar el calculo.

Metricas: recall por clase y, si el destino tiene todas las clases de la
matriz, su media geometrica Gm. Importancia por permutacion (exactitud
balanceada, escalado "fuente", sorteos 0..N_PI-1) en cada par.

Mismos eventos en ambas ventanas, como en ranking_transferibilidad.py.
Cada sistema por separado; nunca se juntan volcanes.

Uso:
    python scripts/transferencia.py [--workers 3] [--trials 200] [--reps 20]
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
from ranking_transferibilidad import F16, OVDAS, SOPORTE  # noqa: E402

OUT = ROOT / "experiments" / "reports" / "transferencia"
N_POR_CLASE, REPS, N_TRIALS, N_PI = 100, 20, 200, 3
TOPE_DESTINO, TOPE_SISTEMA = 5000, 20000
SEMILLA = 41
GRILLA_C = np.logspace(-3, 3, 13)


# ------------------------------------------------------------------ datos --

def cargar() -> dict[str, pd.DataFrame]:
    """Eventos con descriptores en ambas ventanas, con sistema, clase y SNR."""
    out = {"fija": [], "coda": []}
    rng = np.random.default_rng(SEMILLA)
    for key, (nombre, pool) in OVDAS.items():
        c = pd.read_parquet(ROOT / f"data/processed/features_dur/features_{key}{pool}.parquet")
        f = pd.read_parquet(ROOT / f"data/processed/features/features_{key}.parquet")
        ids = sorted(set(c.event_id) & set(f.event_id))
        if len(ids) > TOPE_SISTEMA * 3:        # hoy solo Copahue: muestra fija, igual en ambas ventanas
            ids = list(rng.choice(ids, TOPE_SISTEMA * 3, replace=False))
        ids = set(ids)
        for w, d in (("fija", f), ("coda", c)):
            d = d[d.event_id.isin(ids)].drop_duplicates("event_id")
            out[w].append(pd.DataFrame({"id": d.event_id.astype(str), "sistema": nombre,
                                        "origen": "OVDAS", "clase": d.label.values,
                                        "snr_db": d.snr_db.values, **{k: d[k].values for k in F16}}))
    base = ROOT / "data/processed/features_manual"
    v = {w: pd.read_parquet(base / f"features16_{w}.parquet") for w in ("fija", "coda")}
    ok = [set(x.loc[(x.motivo == "") & ~x.duplicado.fillna(False).astype(bool), "item_id"])
          for x in v.values()]
    ids = ok[0] & ok[1]
    for w, x in v.items():
        x = x[x.item_id.isin(ids)].copy()
        dec = x.sistema == "Decepcion"
        x.loc[dec, "sistema"] = "Decepcion" + x.loc[dec, "item_id"].str.extract(r"dec\.(\d\d)")[0]
        out[w].append(pd.DataFrame({"id": x.item_id.values, "sistema": x.sistema.values,
                                    "origen": x.dataset.values, "clase": x.clase.values,
                                    "snr_db": x.snr_db.values, **{k: x[k].values for k in F16}}))
    return {w: pd.concat(d, ignore_index=True) for w, d in out.items()}


def escalar_sistema(d: pd.DataFrame) -> pd.DataFrame:
    """Mediana e IQR de todos los eventos de cada sistema, sin etiquetas."""
    partes = []
    for _, g in d.groupby("sistema"):
        x = g[F16].astype(float)
        iqr = (x.quantile(.75) - x.quantile(.25)).replace(0, 1.0)
        partes.append(g.assign(**((x - x.median()) / iqr)))
    return pd.concat(partes)


def soporte(d: pd.DataFrame, clases) -> dict[str, list[str]]:
    n = d[d.clase.isin(clases)].groupby(["sistema", "clase"]).size()
    out = {}
    for (s, c), k in n.items():
        if k >= SOPORTE:
            out.setdefault(s, []).append(c)
    return out


# --------------------------------------------------------------- modelos --

def _xgb(params, seed):
    from xgboost import XGBClassifier
    return XGBClassifier(**params, tree_method="hist", n_jobs=1, random_state=seed,
                         eval_metric="mlogloss", verbosity=0)


def _logistica(C):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=5000)


class Modelo:
    """Ajuste con la imputacion y la estandarizacion que pide cada clasificador."""

    def __init__(self, tipo, conf, clases, seed=0):
        self.tipo, self.conf, self.clases, self.seed = tipo, conf, list(clases), seed

    def fit(self, x, y):
        yi = np.array([self.clases.index(c) for c in y])
        if self.tipo == "xgb":
            self.m = _xgb(self.conf, self.seed).fit(x, yi)
        else:
            self.med = np.nanmedian(x, axis=0)
            x = np.where(np.isnan(x), self.med, x)
            self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
            self.m = _logistica(self.conf).fit((x - self.mu) / self.sd, yi)
        return self

    def predict(self, x):
        if self.tipo != "xgb":
            x = (np.where(np.isnan(x), self.med, x) - self.mu) / self.sd
        return np.array(self.clases)[self.m.predict(x)]


def bal_cv(tipo, conf, x, y, clases, seed):
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    sc = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        p = Modelo(tipo, conf, clases, seed).fit(x[a], y[a]).predict(x[b])
        sc.append(balanced_accuracy_score(y[b], p))
    return float(np.mean(sc))


def ajustar_xgb(x, y, clases, n_trials, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(t):
        conf = dict(n_estimators=t.suggest_int("n_estimators", 200, 2000),
                    max_depth=t.suggest_int("max_depth", 3, 12),
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    subsample=t.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                    min_child_weight=t.suggest_float("min_child_weight", 1, 50, log=True),
                    reg_alpha=t.suggest_float("reg_alpha", 1e-8, 10, log=True),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-8, 10, log=True))
        return bal_cv("xgb", conf, x, y, clases, seed)

    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials)
    return st.best_params, st.best_value


def ajustar_logistica(x, y, clases, seed):
    sc = [bal_cv("log", C, x, y, clases, seed) for C in GRILLA_C]
    i = int(np.argmax(sc))
    return float(GRILLA_C[i]), sc[i]


def sorteo(g: pd.DataFrame, clases, n, seed) -> pd.DataFrame:
    return pd.concat([g[g.clase == c].sample(min(n, (g.clase == c).sum()), random_state=seed)
                      for c in clases])


# ------------------------------------------------------------------ trabajo --

def puntuar(m, g, clases, minimo=SOPORTE):
    """Recall por clase presente (con al menos `minimo` eventos) y Gm si estan
    todas. En la diagonal minimo=1: el soporte ya se exigio a la fuente, y tras
    sacar los 100 de entrenamiento puede quedar menos de 50 (Ruiz-Olleta LP)."""
    p = m.predict(g[F16].to_numpy(float))
    r = {f"rec_{c}": float(np.mean(p[g.clase.to_numpy() == c] == c))
         for c in clases if (g.clase == c).sum() >= minimo}
    r["gm"] = (float(np.prod(list(r.values())) ** (1 / len(clases)))
               if len(r) == len(clases) else np.nan)
    return r


def importancia(m, g, clases, seed, minimo=SOPORTE):
    from sklearn.metrics import balanced_accuracy_score
    n = g.clase.value_counts()
    g = g[g.clase.isin([c for c in clases if n.get(c, 0) >= minimo])]
    g = pd.concat([h.sample(min(len(h), 1000), random_state=seed) for _, h in g.groupby("clase")])
    if g.clase.nunique() < 2:
        return {}
    x, y = g[F16].to_numpy(float), g.clase.to_numpy()
    base = balanced_accuracy_score(y, m.predict(x))
    rng = np.random.default_rng(seed)
    out = {}
    for j, f in enumerate(F16):
        caidas = []
        for _ in range(3):
            xp = x.copy()
            xp[:, j] = rng.permutation(xp[:, j])
            caidas.append(base - balanced_accuracy_score(y, m.predict(xp)))
        out[f"pi_{f}"] = float(np.mean(caidas))
    return out


def trabajo(args):
    matriz, ventana, fuente, clases, destinos, n_trials, reps = args
    warnings.filterwarnings("ignore")
    destino_ruta = OUT / f"{matriz}_{ventana}_{fuente.replace(' ', '_')}.parquet"
    if destino_ruta.exists():
        return str(destino_ruta)
    d = DATOS[ventana]
    dd = {"fuente": d, "sistema": DATOS_SIS[ventana]}
    src = d[(d.sistema == fuente) & d.clase.isin(clases)]
    # ajuste sobre un sorteo independiente (semilla propia), escalado "fuente"
    t = sorteo(src, clases, N_POR_CLASE, SEMILLA + 999)
    xt, yt = t[F16].to_numpy(float), t.clase.to_numpy()
    p_xgb, cv_xgb = ajustar_xgb(xt, yt, clases, n_trials, SEMILLA)
    c_log, cv_log = ajustar_logistica(xt, yt, clases, SEMILLA)
    confs = {"xgb": p_xgb, "log": c_log}
    filas = []
    for esc, base in dd.items():
        b_src = base[(base.sistema == fuente) & base.clase.isin(clases)]
        dest_g = {}
        for s in destinos:
            g = base[(base.sistema == s) & base.clase.isin(clases)]
            dest_g[s] = pd.concat([h.sample(min(len(h), TOPE_DESTINO), random_state=SEMILLA)
                                   for _, h in g.groupby("clase")])
        if esc == "sistema":
            # el escalado "sistema" de la logistica parte de los mismos descriptores
            # escalados; la conf de C se reajusta sobre el sorteo de ajuste escalado
            ts = b_src.loc[t.index]
            confs_esc = dict(confs, log=ajustar_logistica(ts[F16].to_numpy(float),
                                                          ts.clase.to_numpy(), clases, SEMILLA)[0])
            conf_log_sis = confs_esc["log"]
        else:
            confs_esc = confs
        for r in range(reps):
            tr = sorteo(b_src, clases, N_POR_CLASE, SEMILLA + r)
            for tipo in ("xgb", "log"):
                m = Modelo(tipo, confs_esc[tipo], clases, SEMILLA + r).fit(
                    tr[F16].to_numpy(float), tr.clase.to_numpy())
                for s, g in dest_g.items():
                    minimo = 1 if s == fuente else SOPORTE
                    if s == fuente:
                        g = b_src.drop(index=tr.index)
                        g = pd.concat([h.sample(min(len(h), TOPE_DESTINO), random_state=SEMILLA)
                                       for _, h in g.groupby("clase")])
                    fila = dict(matriz=matriz, ventana=ventana, escalado=esc, modelo=tipo,
                                fuente=fuente, destino=s, rep=r, **puntuar(m, g, clases, minimo),
                                **{f"n_{c}": int((g.clase == c).sum()) for c in clases},
                                snr_mediana=float(np.nanmedian(g.snr_db)),
                                snr_p25=float(np.nanpercentile(g.snr_db, 25)),
                                snr_p75=float(np.nanpercentile(g.snr_db, 75)))
                    if esc == "fuente" and r < N_PI:
                        fila.update(importancia(m, g, clases, SEMILLA + r, minimo))
                    filas.append(fila)
    res = pd.DataFrame(filas).assign(cv_ajuste_xgb=cv_xgb, cv_ajuste_log=cv_log,
                                     conf_xgb=str(p_xgb), conf_log=c_log,
                                     conf_log_sistema=conf_log_sis)
    res.to_parquet(destino_ruta, index=False)
    return str(destino_ruta)


DATOS, DATOS_SIS = {}, {}


def init(datos, datos_sis, out):
    global OUT
    OUT = out
    DATOS.update(datos)
    DATOS_SIS.update(datos_sis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--solo", nargs="*", help="limita a estas fuentes (prueba)")
    ap.add_argument("--prueba", action="store_true", help="escribe en transferencia_prueba/")
    a = ap.parse_args()
    global OUT
    if a.prueba:
        OUT = OUT.parent / "transferencia_prueba"
    OUT.mkdir(parents=True, exist_ok=True)
    datos = cargar()
    datos_sis = {w: escalar_sistema(d) for w, d in datos.items()}
    trabajos = []
    for w, d in datos.items():
        for matriz, clases in (("M1", ["LP", "VT", "HB"]), ("M2", ["LP", "VT"])):
            sop = soporte(d, clases)
            fuentes = [s for s, cs in sop.items() if set(clases) <= set(cs)]
            if matriz == "M1":
                destinos = [s for s, cs in sop.items() if "HB" in cs]
            else:
                destinos = list(sop)
            for f in sorted(fuentes):
                if a.solo and f not in a.solo:
                    continue
                trabajos.append((matriz, w, f, clases, sorted(destinos), a.trials, a.reps))
    print(f"{len(trabajos)} trabajos (matriz, ventana, fuente)", flush=True)
    for t in trabajos:
        print(f"  {t[0]} {t[1]} {t[2]} -> {len(t[4])} destinos", flush=True)
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(datos, datos_sis, OUT)) as ex:
        for k, ruta in enumerate(ex.map(trabajo, trabajos), 1):
            print(f"[{k}/{len(trabajos)}] {ruta}", flush=True)
    todo = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("M*_*.parquet"))])
    todo.to_parquet(OUT.parent / "transferencia_pares.parquet", index=False)
    print(f"-> {OUT.parent / 'transferencia_pares.parquet'} ({len(todo)} filas)")


if __name__ == "__main__":
    main()
