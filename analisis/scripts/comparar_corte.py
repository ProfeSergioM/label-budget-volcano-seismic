"""
Cuanto cambian los resultados si los eventos de validacion se cortan solo con
la propuesta automatica (auto_onset, auto_fin) en vez del corte del analista.

Mismo conjunto de eventos en ambos cortes: LP y VT validos con el corte
manual y con el automatico en la ventana pedida (sin duplicados). Cada
sistema por separado; nunca se juntan volcanes (Decepcion se separa por
campana, como en transferencia.py).

(a) desplazamiento auto - manual del onset y del fin: mediana, IQR y
    fraccion con |delta| > 1 s; ademas la fraccion de marcas en que el
    analista acepto la propuesta sin editarla.
(b) FI, DET y SR: correlacion de Spearman entre ambos cortes, diferencia
    mediana (auto - manual) y esa diferencia en unidades del IQR manual.
(c) desempeno LP-VT:
    - dentro del sistema: CV estratificada 5 pliegues x 5 sorteos de hasta
      100 eventos por clase; regresion logistica y XGBoost con la
      configuracion fija de clasificacion_por_sistema.py. Los sorteos y los
      pliegues son los mismos para ambos cortes (comparacion pareada); Gm de
      la matriz de confusion acumulada.
    - como destino de transferencia desde las cinco fuentes OVDAS (que no
      cambian): la configuracion de cada fuente es la que eligio
      transferencia.py (M2, ventana pedida), REPS sorteos de entrenamiento de
      100 por clase con las semillas de transferencia.py; escalados "fuente"
      y "sistema" (en el escalado por sistema el destino se escala con la
      mediana y el IQR de sus eventos LP/VT del conjunto comun, en cada
      corte con sus propios descriptores).

Salidas: experiments/reports/corte_auto_{desplazamiento,descriptores,
dentro,transferencia}_<ventana>.csv

Uso:
    python scripts/comparar_corte.py [--ventana fija|coda] [--reps 20]
"""
from __future__ import annotations

import argparse
import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cortar_trazas as CT  # noqa: E402
from clasificacion_por_sistema import cv as cv_fija  # noqa: E402
from ranking_transferibilidad import F16, OVDAS, SOPORTE  # noqa: E402
import transferencia as TR  # noqa: E402

OUT = ROOT / "experiments" / "reports"
BASE = ROOT / "data" / "processed" / "features_manual"
CLASES = ["LP", "VT"]
N_CLASE, REPS_CV, SEMILLA = 100, 5, 41
TRES = ["FI", "DET", "SR"]


def sistema(x: pd.DataFrame) -> pd.Series:
    s = x.sistema.copy()
    dec = s == "Decepcion"
    s[dec] = "Decepcion" + x.loc[dec, "item_id"].str.extract(r"dec\.(\d\d)")[0]
    return s


def pares(ventana: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptores manual y auto del mismo conjunto de eventos, alineados."""
    man = pd.read_parquet(BASE / f"features16_{ventana}.parquet")
    aut = pd.read_parquet(BASE / f"features16_auto_{ventana}.parquet")
    ok_m = man[(man.motivo == "") & ~man.duplicado.fillna(False).astype(bool)
               & man.clase.isin(CLASES)]
    ok_a = aut[aut.motivo == ""]
    ids = sorted(set(ok_m.item_id) & set(ok_a.item_id))
    man = ok_m.set_index("item_id").loc[ids].reset_index()
    aut = ok_a.set_index("item_id").loc[ids].reset_index()
    man["sistema"] = aut["sistema"] = sistema(man).values
    excluidos = (man.shape[0], len(set(ok_m.item_id) - set(ok_a.item_id)))
    print(f"ventana {ventana}: {excluidos[0]} eventos comunes; {excluidos[1]} validos con el "
          f"corte manual quedan fuera con el automatico", flush=True)
    return man, aut


# ------------------------------------------------------------- (a) y (b) --

def desplazamiento(man: pd.DataFrame, ventana: str) -> pd.DataFrame:
    p = pd.read_csv(CT.PICKS, dtype=str, keep_default_na=False).drop_duplicates(
        "item_id", keep="last").set_index("item_id")
    p = p.loc[man.item_id]
    num = lambda c: pd.to_numeric(p[c], errors="coerce").to_numpy()  # noqa: E731
    d = pd.DataFrame({"sistema": man.sistema.values, "clase": man.clase.values,
                      "d_onset": num("auto_onset") - num("onset"),
                      "d_fin": num("auto_fin") - num("fin"),
                      "aceptado": p.origen.isin(["aceptado", "lote_auto",
                                                 "igual_al_auto_sin_registro"]).values})
    filas = []
    for (s, c), g in pd.concat([d, d.assign(clase="LP+VT")]).groupby(["sistema", "clase"]):
        f = dict(ventana=ventana, sistema=s, clase=c, n=len(g),
                 frac_propuesta_aceptada=float(g.aceptado.mean()))
        for k in ("d_onset", "d_fin"):
            v = g[k].dropna()
            f.update({f"{k}_mediana": float(v.median()),
                      f"{k}_q25": float(v.quantile(.25)), f"{k}_q75": float(v.quantile(.75)),
                      f"{k}_iqr": float(v.quantile(.75) - v.quantile(.25)),
                      f"{k}_frac_abs_gt1s": float((v.abs() > 1).mean()),
                      f"{k}_n": int(len(v))})
        filas.append(f)
    return pd.DataFrame(filas)


def descriptores(man, aut, ventana) -> pd.DataFrame:
    filas = []
    for s in sorted(man.sistema.unique()):
        for c in CLASES + ["LP+VT"]:
            k = (man.sistema == s) & ((man.clase == c) if c != "LP+VT" else True)
            if k.sum() < 10:
                continue
            for f in TRES:
                a, b = man.loc[k, f].astype(float), aut.loc[k, f].astype(float)
                v = a.notna() & b.notna()
                iqr = a[v].quantile(.75) - a[v].quantile(.25)
                dif = (b[v] - a[v]).median()
                filas.append(dict(ventana=ventana, sistema=s, clase=c, descriptor=f,
                                  n=int(v.sum()), spearman=float(spearmanr(a[v], b[v])[0]),
                                  mediana_manual=float(a[v].median()),
                                  mediana_auto=float(b[v].median()),
                                  dif_mediana=float(dif),
                                  dif_mediana_en_iqr=float(dif / iqr) if iqr else np.nan,
                                  abs_dif_mediana_en_iqr=float((b[v] - a[v]).abs().median() / iqr)
                                  if iqr else np.nan))
    return pd.DataFrame(filas)


# ------------------------------------------------------------------- (c) --

def dentro(man, aut, ventana) -> pd.DataFrame:
    filas = []
    for s in sorted(man.sistema.unique()):
        g = man[man.sistema == s]
        n = g.clase.value_counts()
        if not all(n.get(c, 0) >= SOPORTE for c in CLASES):
            continue
        for tipo in ("log", "xgb"):
            cms = {"manual": None, "auto": None}
            gms = {"manual": [], "auto": []}
            for r in range(REPS_CV):
                idx = pd.concat([g[g.clase == c].sample(min(N_CLASE, n[c]),
                                                        random_state=SEMILLA + r)
                                 for c in CLASES]).index
                for corte, d in (("manual", man), ("auto", aut)):
                    clases, cm = cv_fija(d.loc[idx], tipo, SEMILLA + r)
                    cms[corte] = cm if cms[corte] is None else cms[corte] + cm
                    rec = np.diag(cm) / cm.sum(1)
                    gms[corte].append(float(np.sqrt(np.prod(rec))))
            f = dict(ventana=ventana, sistema=s, modelo=tipo, n_LP=int(n["LP"]),
                     n_VT=int(n["VT"]))
            for corte, cm in cms.items():
                rec = np.diag(cm) / cm.sum(1)
                f.update({f"gm_{corte}": float(np.sqrt(np.prod(rec))),
                          f"rec_LP_{corte}": float(rec[0]), f"rec_VT_{corte}": float(rec[1]),
                          f"gm_{corte}_sd_sorteos": float(np.std(gms[corte]))})
            dd = np.array(gms["auto"]) - np.array(gms["manual"])
            f.update(dgm=f["gm_auto"] - f["gm_manual"], dgm_min_sorteo=float(dd.min()),
                     dgm_max_sorteo=float(dd.max()))
            filas.append(f)
    return pd.DataFrame(filas)


def escalar(g: pd.DataFrame) -> pd.DataFrame:
    x = g[F16].astype(float)
    iqr = (x.quantile(.75) - x.quantile(.25)).replace(0, 1.0)
    return g.assign(**((x - x.median()) / iqr))


def transferencia(man, aut, ventana, reps) -> pd.DataFrame:
    datos = TR.cargar()[ventana]
    ovdas = datos[datos.origen == "OVDAS"]
    ovdas_sis = TR.escalar_sistema(ovdas)
    t = pd.read_parquet(OUT / "transferencia_pares.parquet")
    t = t[(t.matriz == "M2") & (t.ventana == ventana)].drop_duplicates("fuente").set_index("fuente")
    fuentes = [nombre for nombre, _ in OVDAS.values()]
    destinos = {}
    for s in sorted(man.sistema.unique()):
        n = man[man.sistema == s].clase.value_counts()
        cl = [c for c in CLASES if n.get(c, 0) >= SOPORTE]
        if not cl:
            continue
        for corte, d in (("manual", man), ("auto", aut)):
            g = d[(d.sistema == s) & d.clase.isin(cl)]
            g = pd.concat([h.sample(min(len(h), TR.TOPE_DESTINO), random_state=SEMILLA)
                           for _, h in g.groupby("clase")])
            destinos[(s, corte, "fuente")] = g
            destinos[(s, corte, "sistema")] = escalar(d[(d.sistema == s)
                                                       & d.clase.isin(CLASES)]).loc[g.index]
    filas = []
    for fu in fuentes:
        conf = {"xgb": ast.literal_eval(t.loc[fu, "conf_xgb"]),
                "log": {"fuente": float(t.loc[fu, "conf_log"]),
                        "sistema": float(t.loc[fu, "conf_log_sistema"])}}
        for esc, base in (("fuente", ovdas), ("sistema", ovdas_sis)):
            src = base[(base.sistema == fu) & base.clase.isin(CLASES)]
            for r in range(reps):
                tr = TR.sorteo(src, CLASES, TR.N_POR_CLASE, SEMILLA + r)
                for tipo in ("xgb", "log"):
                    c = conf["xgb"] if tipo == "xgb" else conf["log"][esc]
                    m = TR.Modelo(tipo, c, CLASES, SEMILLA + r).fit(
                        tr[F16].to_numpy(float), tr.clase.to_numpy())
                    for (s, corte, e), g in destinos.items():
                        if e != esc:
                            continue
                        filas.append(dict(ventana=ventana, escalado=esc, modelo=tipo, fuente=fu,
                                          destino=s, corte=corte, rep=r,
                                          **TR.puntuar(m, g, CLASES),
                                          **{f"n_{k}": int((g.clase == k).sum())
                                             for k in CLASES}))
        print(f"  fuente {fu} lista", flush=True)
    return pd.DataFrame(filas)


def resumen_transferencia(r: pd.DataFrame) -> pd.DataFrame:
    """Por destino (y modelo, escalado): media sobre fuentes y sorteos de cada
    corte y la diferencia pareada auto - manual."""
    met = [c for c in ("gm", "rec_LP", "rec_VT") if c in r]
    k = ["ventana", "escalado", "modelo", "fuente", "destino", "rep"]
    w = r.pivot_table(index=k, columns="corte", values=met)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    for m in met:
        if f"{m}_auto" in w:
            w[f"d{m}"] = w[f"{m}_auto"] - w[f"{m}_manual"]
    por_fuente = w.groupby(k[:-1]).mean(numeric_only=True).drop(columns="rep").reset_index()
    g = por_fuente.groupby(k[:3] + ["destino"])
    out = g.mean(numeric_only=True)
    for m in met:
        if f"d{m}" in por_fuente:
            out[f"d{m}_min_fuente"] = g[f"d{m}"].min()
            out[f"d{m}_max_fuente"] = g[f"d{m}"].max()
    return out.reset_index(), por_fuente


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ventana", default="fija", choices=["fija", "coda"])
    ap.add_argument("--reps", type=int, default=TR.REPS)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 220)
    man, aut = pares(a.ventana)
    v = a.ventana

    d = desplazamiento(man, v)
    d.to_csv(OUT / f"corte_auto_desplazamiento_{v}.csv", index=False)
    print(d[d.clase == "LP+VT"][["sistema", "n", "frac_propuesta_aceptada", "d_onset_mediana",
                                 "d_onset_iqr", "d_onset_frac_abs_gt1s", "d_fin_mediana",
                                 "d_fin_iqr", "d_fin_frac_abs_gt1s"]].round(2).to_string(index=False))

    f = descriptores(man, aut, v)
    f.to_csv(OUT / f"corte_auto_descriptores_{v}.csv", index=False)
    print(f[f.clase == "LP+VT"].pivot_table(index="sistema", columns="descriptor",
                                             values=["spearman", "dif_mediana_en_iqr"])
          .round(2).to_string())

    w = dentro(man, aut, v)
    w.to_csv(OUT / f"corte_auto_dentro_{v}.csv", index=False)
    if len(w):
        print(w[["sistema", "modelo", "n_LP", "n_VT", "gm_manual", "gm_auto", "dgm",
                 "dgm_min_sorteo", "dgm_max_sorteo"]].round(3).to_string(index=False))

    r = transferencia(man, aut, v, a.reps)
    r.to_csv(OUT / f"corte_auto_transferencia_sorteos_{v}.csv", index=False)
    res, pf = resumen_transferencia(r)
    res.to_csv(OUT / f"corte_auto_transferencia_{v}.csv", index=False)
    pf.to_csv(OUT / f"corte_auto_transferencia_por_fuente_{v}.csv", index=False)
    cols = [c for c in ("escalado", "modelo", "destino", "gm_manual", "gm_auto", "dgm",
                        "dgm_min_fuente", "dgm_max_fuente", "drec_LP", "drec_VT") if c in res]
    print(res[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
