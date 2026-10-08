"""
Verificaciones del diseno de transferencia M2 (LP-VT), ventana fija.

Reutiliza la carga, los modelos y las configuraciones ya elegidas por
transferencia.py (conf_xgb, conf_log, conf_log_sistema guardadas en
experiments/reports/transferencia_pares.parquet); no se reajusta nada.

1. Sorteo temporal. Para cada fuente y sorteo r, los 100 eventos por clase
   se eligen (a) al azar, exactamente como en transferencia.py (semilla
   SEMILLA + r), o (b) repartidos en el tiempo: la clase se ordena por fecha
   y se toma un evento cada N/100 posiciones, con un desfase inicial uniforme
   en [0, N/100) sorteado con la semilla SEMILLA + r (muestreo sistematico).
   El modelo usa la misma semilla SEMILLA + r en ambos casos. Los destinos y
   su muestra son los de transferencia.py (tope 5000 por clase, muestra fija);
   en la diagonal se evalua en los eventos de la fuente fuera del sorteo.
   Fecha: OVDAS, origin_time; validacion, t0_utc + onset (picks manuales).

2. Proporcion del destino sin etiquetas. Con el modelo del sorteo al azar,
   cada destino se evalua con todos sus eventos LP y VT (composicion natural,
   sin tope). Sobre las 200 muestras de entrenamiento, una CV estratificada de
   5 pliegues da (i) la matriz de confusion de la fuente para BBSE (Lipton et
   al., 2018) y (ii) las probabilidades fuera de pliegue con las que se ajusta
   una calibracion de Platt. La proporcion LP del destino se estima con BBSE
   (prediccion dura del modelo) y con el EM de Saerens et al. (2002) sobre las
   probabilidades calibradas; las probabilidades calibradas se reponderan con
   el prior estimado (el de entrenamiento es 50/50) y se clasifica por maximo.
   Se reportan tambien el modelo original, la calibracion sin reponderar y la
   reponderacion con el prior real (oraculo, cota superior).

Salidas en experiments/reports/verif_*.csv. Cada sistema por separado.

Uso:
    nice python scripts/verificaciones_transferencia.py [--reps 5] [--solo Hudson ...]
"""
from __future__ import annotations

import argparse
import ast
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ranking_transferibilidad import F16, OVDAS, SOPORTE  # noqa: E402
from transferencia import (N_POR_CLASE, SEMILLA, TOPE_DESTINO, Modelo,  # noqa: E402
                           cargar, escalar_sistema, puntuar, soporte, sorteo)

REP = ROOT / "experiments" / "reports"
CLASES = ["LP", "VT"]


# ----------------------------------------------------------------- fechas --

def fechas() -> pd.Series:
    """Fecha de cada evento (id -> Timestamp UTC)."""
    partes = []
    for key in OVDAS:
        f = pd.read_parquet(ROOT / f"data/processed/features/features_{key}.parquet",
                            columns=["event_id", "origin_time"])
        partes.append(pd.Series(pd.to_datetime(f.origin_time, utc=True, errors="coerce").values,
                                index=f.event_id.astype(str)))
    p = pd.read_csv(ROOT / "experiments/picks/picks_manuales.csv",
                    usecols=["item_id", "t0_utc", "guardado_utc"])
    p = p.sort_values("guardado_utc").drop_duplicates("item_id", keep="last")
    v = pd.read_parquet(ROOT / "data/processed/features_manual/features16_fija.parquet",
                        columns=["item_id", "onset"]).drop_duplicates("item_id")
    v = v.merge(p, on="item_id", how="left")
    t = pd.to_datetime(v.t0_utc, utc=True, errors="coerce") + pd.to_timedelta(pd.to_numeric(v.onset, errors="coerce"), unit="s")
    partes.append(pd.Series(t.values, index=v.item_id))
    s = pd.concat(partes)
    return s[~s.index.duplicated()]


def sorteo_temporal(g: pd.DataFrame, n, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    partes = []
    for c in CLASES:
        h = g[(g.clase == c) & g.fecha.notna()].sort_values(["fecha", "id"])
        N = len(h)
        if N <= n:
            partes.append(h)
            continue
        paso = N / n
        idx = np.floor(rng.uniform(0, paso) + paso * np.arange(n)).astype(int)
        partes.append(h.iloc[np.minimum(idx, N - 1)])
    return pd.concat(partes)


# ------------------------------------------------------------ proporciones --

def proba_lp(m: Modelo, x):
    if m.tipo != "xgb":
        x = (np.where(np.isnan(x), m.med, x) - m.mu) / m.sd
    return m.m.predict_proba(x)[:, 0]          # columna 0 = LP


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(z, y_lp):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(z.reshape(-1, 1), y_lp)
    return lambda zz: lr.predict_proba(np.asarray(zz).reshape(-1, 1))[:, 1]


def em_saerens(p, prior_tr=0.5, tol=1e-8, it=1000):
    q = prior_tr
    for _ in range(it):
        a = p * q / prior_tr
        b = (1 - p) * (1 - q) / (1 - prior_tr)
        nq = float(np.mean(a / (a + b)))
        if abs(nq - q) < tol:
            return nq
        q = nq
    return q


def reponderar(p, q, prior_tr=0.5):
    a = p * q / prior_tr
    b = (1 - p) * (1 - q) / (1 - prior_tr)
    return a / (a + b)


def metricas(pred_lp: np.ndarray, y: np.ndarray, minimo):
    """Exactitud y Gm (recall por clase con al menos `minimo` eventos)."""
    pred = np.where(pred_lp, "LP", "VT")
    r = {c: float(np.mean(pred[y == c] == c)) for c in CLASES if (y == c).sum() >= minimo}
    gm = float(np.sqrt(r["LP"] * r["VT"])) if len(r) == 2 else np.nan
    return float(np.mean(pred == y)), gm


# ------------------------------------------------------------------ main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--solo", nargs="*")
    ap.add_argument("--sufijo", default="")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    from sklearn.model_selection import StratifiedKFold

    t0 = time.time()
    d = cargar()["fija"]
    d = d.assign(fecha=d.id.map(fechas()))
    bases = {"fuente": d, "sistema": escalar_sistema(d)}
    sop = soporte(d, CLASES)
    fuentes = sorted(s for s, cs in sop.items() if set(CLASES) <= set(cs))
    destinos = sorted(sop)
    tp = pd.read_parquet(REP / "transferencia_pares.parquet")
    tp = tp[(tp.matriz == "M2") & (tp.ventana == "fija")]
    confs = tp.groupby("fuente")[["conf_xgb", "conf_log", "conf_log_sistema"]].first()
    ref = tp.set_index(["escalado", "modelo", "fuente", "destino", "rep"]).gm

    # fechas disponibles por fuente
    cob = d[d.sistema.isin(fuentes) & d.clase.isin(CLASES)].groupby(["sistema", "clase"]).agg(
        n=("id", "size"), sin_fecha=("fecha", lambda s: int(s.isna().sum())),
        desde=("fecha", "min"), hasta=("fecha", "max"))
    cob.to_csv(REP / f"verif_fechas_fuentes{a.sufijo}.csv")
    print(cob.to_string(), flush=True)

    filas_t, filas_p = [], []
    for fuente in fuentes:
        if a.solo and fuente not in a.solo:
            continue
        cf = confs.loc[fuente]
        for esc, base in bases.items():
            conf = {"xgb": ast.literal_eval(cf.conf_xgb),
                    "log": float(cf.conf_log if esc == "fuente" else cf.conf_log_sistema)}
            b = base[base.clase.isin(CLASES)]
            b_src = b[b.sistema == fuente]
            dest_tope, dest_nat = {}, {}
            for s in destinos:
                g = b[b.sistema == s]
                dest_tope[s] = pd.concat([h.sample(min(len(h), TOPE_DESTINO), random_state=SEMILLA)
                                          for _, h in g.groupby("clase")])
                dest_nat[s] = g
            for r in range(a.reps):
                sorteos = {"azar": sorteo(b_src, CLASES, N_POR_CLASE, SEMILLA + r),
                           "temporal": sorteo_temporal(b_src, N_POR_CLASE, SEMILLA + r)}
                for tipo in ("xgb", "log"):
                    for modo, tr in sorteos.items():
                        xtr, ytr = tr[F16].to_numpy(float), tr.clase.to_numpy()
                        m = Modelo(tipo, conf[tipo], CLASES, SEMILLA + r).fit(xtr, ytr)
                        span = (tr.fecha.max() - tr.fecha.min()).days
                        for s in destinos:
                            minimo = 1 if s == fuente else SOPORTE
                            g = dest_tope[s]
                            if s == fuente:
                                g = b_src.drop(index=tr.index)
                                g = pd.concat([h.sample(min(len(h), TOPE_DESTINO), random_state=SEMILLA)
                                               for _, h in g.groupby("clase")])
                            sc = puntuar(m, g, CLASES, minimo)
                            filas_t.append(dict(escalado=esc, modelo=tipo, fuente=fuente, destino=s,
                                                rep=r, sorteo=modo, rec_LP=sc.get("rec_LP", np.nan),
                                                rec_VT=sc.get("rec_VT", np.nan), gm=sc["gm"],
                                                dias_entrenamiento=span,
                                                gm_ref=(ref.get((esc, tipo, fuente, s, r), np.nan)
                                                        if modo == "azar" else np.nan)))
                        if modo != "azar":
                            continue
                        # ---- verificacion 2 sobre el modelo del sorteo al azar
                        y_lp = (ytr == "LP").astype(int)
                        oof_p, oof_pred = np.zeros(len(ytr)), np.zeros(len(ytr), bool)
                        for i_a, i_b in StratifiedKFold(5, shuffle=True, random_state=SEMILLA + r
                                                        ).split(xtr, ytr):
                            mm = Modelo(tipo, conf[tipo], CLASES, SEMILLA + r).fit(xtr[i_a], ytr[i_a])
                            oof_p[i_b] = proba_lp(mm, xtr[i_b])
                            oof_pred[i_b] = mm.predict(xtr[i_b]) == "LP"
                        cal = platt(logit(oof_p), y_lp)
                        tpr = oof_pred[y_lp == 1].mean()      # P(pred LP | LP)
                        fpr = oof_pred[y_lp == 0].mean()      # P(pred LP | VT)
                        prior_tr = y_lp.mean()
                        for s in destinos:
                            g = dest_nat[s]
                            minimo = SOPORTE
                            if s == fuente:
                                g, minimo = b_src.drop(index=tr.index), 1
                            x, y = g[F16].to_numpy(float), g.clase.to_numpy()
                            q_real = float(np.mean(y == "LP"))
                            p_raw = proba_lp(m, x)
                            pred_raw = m.predict(x) == "LP"
                            p_cal = cal(logit(p_raw))
                            q_cc = float(pred_raw.mean())
                            q_bbse = (float(np.clip((q_cc - fpr) / (tpr - fpr), 0, 1))
                                      if tpr - fpr > 1e-6 else np.nan)
                            q_em = em_saerens(p_cal, prior_tr)
                            fila = dict(escalado=esc, modelo=tipo, fuente=fuente, destino=s, rep=r,
                                        n_LP=int((y == "LP").sum()), n_VT=int((y == "VT").sum()),
                                        q_real=q_real, q_cc=q_cc, q_pcc=float(p_cal.mean()),
                                        q_bbse=q_bbse, q_em=q_em, tpr_src=tpr, fpr_src=fpr)
                            variantes = {"original": pred_raw, "calibrado": p_cal >= .5,
                                         "bbse": (reponderar(p_cal, q_bbse, prior_tr) >= .5
                                                  if np.isfinite(q_bbse) else p_cal >= .5),
                                         "em": reponderar(p_cal, q_em, prior_tr) >= .5,
                                         "oraculo": reponderar(p_cal, q_real, prior_tr) >= .5}
                            for k, v in variantes.items():
                                fila[f"acc_{k}"], fila[f"gm_{k}"] = metricas(v, y, minimo)
                            filas_p.append(fila)
                print(f"  {fuente} {esc} rep {r} ({time.time() - t0:.0f} s)", flush=True)

    T = pd.DataFrame(filas_t)
    P = pd.DataFrame(filas_p)
    T.to_csv(REP / f"verif_temporal_pares{a.sufijo}.csv", index=False)
    P.to_csv(REP / f"verif_prior_pares{a.sufijo}.csv", index=False)

    # ---- resumenes
    k = ["escalado", "modelo", "fuente", "destino"]
    w = T.pivot_table(index=k + ["rep"], columns="sorteo", values="gm").reset_index()
    w["dgm"] = w.temporal - w.azar
    rt = w.groupby(k).agg(gm_azar=("azar", "mean"), gm_temporal=("temporal", "mean"),
                          dgm_media=("dgm", "mean"), dgm_de=("dgm", "std")).reset_index()
    rt["diagonal"] = rt.fuente == rt.destino
    rt.to_csv(REP / f"verif_temporal_resumen{a.sufijo}.csv", index=False)
    fuera = rt[~rt.diagonal & rt.dgm_media.notna()]
    pd_ = fuera.groupby(["escalado", "modelo", "destino"]).agg(
        n_fuentes=("fuente", "size"), gm_azar_mediana=("gm_azar", "median"),
        gm_temporal_mediana=("gm_temporal", "median"), dgm_mediana=("dgm_media", "median"),
        dgm_min=("dgm_media", "min"), dgm_max=("dgm_media", "max"),
        absdgm_max=("dgm_media", lambda s: s.abs().max())).reset_index()
    pd_.to_csv(REP / f"verif_temporal_destino{a.sufijo}.csv", index=False)
    pf = fuera.groupby(["escalado", "modelo", "fuente"]).agg(
        n_destinos=("destino", "size"), dgm_mediana=("dgm_media", "median"),
        dgm_min=("dgm_media", "min"), dgm_max=("dgm_media", "max")).reset_index()
    pf.to_csv(REP / f"verif_temporal_fuente{a.sufijo}.csv", index=False)

    for q in ("cc", "pcc", "bbse", "em"):
        P[f"err_{q}"] = P[f"q_{q}"] - P.q_real
    var = ["original", "calibrado", "bbse", "em", "oraculo"]
    agg = {"n_LP": ("n_LP", "first"), "n_VT": ("n_VT", "first"), "q_real": ("q_real", "first"),
           "n_fuentes": ("fuente", "nunique")}
    for q in ("cc", "bbse", "em"):
        agg[f"abserr_{q}_mediana"] = (f"err_{q}", lambda s: s.abs().median())
        agg[f"err_{q}_mediana"] = (f"err_{q}", "median")
    for v in var:
        agg[f"acc_{v}"] = (f"acc_{v}", "median")
        agg[f"gm_{v}"] = (f"gm_{v}", "median")
    Pf = P[P.fuente != P.destino]
    Pf.groupby(["escalado", "modelo", "destino"]).agg(**agg).reset_index().to_csv(
        REP / f"verif_prior_destino{a.sufijo}.csv", index=False)
    Pf.groupby(["escalado", "modelo", "fuente", "destino"])[
        [c for c in P.columns if c.startswith(("q_", "err_", "acc_", "gm_"))]].mean().reset_index().to_csv(
        REP / f"verif_prior_resumen{a.sufijo}.csv", index=False)
    print(f"listo en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
