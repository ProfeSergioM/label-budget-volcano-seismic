"""
Hay, aparte de LP y VT, otra clase que se distinga en todos los centros?

Para cada clase candidata C, un detector "C contra las clases globales del
mismo sistema" (regresion logistica sobre los 16 descriptores, ventana fija)
se entrena en un sistema y se evalua en cada otro sistema que tenga C. En el
destino se mide el recall de C y, si el destino tiene LP o VT, la fraccion de
esos eventos que el detector no confunde con C (especificidad); el puntaje es
la media de ambas (exactitud balanceada). La diagonal es validacion cruzada
de 5 pliegues dentro del sistema.

LP contra VT, con el mismo procedimiento, es la referencia de una clase
global. Una candidata es global si su puntaje se sostiene entre catalogos
(centros distintos) cerca de su diagonal, como LP-VT.

Cada sistema entra con su propia etiqueta: el tremor de cada catalogo es su
propia clase local; el cruce solo prueba si un detector entrenado con el
tremor de un centro reconoce el tremor de otro, sin juntarlos al entrenar.

Dos escalados de los descriptores: "fuente" (media y varianza del sistema de
entrenamiento, como en el diseno de transferencia) y "sistema" (mediana e IQR
de todos los eventos de cada sistema, sin etiquetas).

Uso:
    python scripts/candidatos_globales.py [logistica|arboles]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from heterogeneidad_clases import datos  # noqa: E402
from ranking_transferibilidad import F16  # noqa: E402

OUT = ROOT / "experiments" / "reports"
SOPORTE, N_TRAIN, REPS, TOPE = 50, 200, 5, 3000
SEMILLA = 41
FAMILIAS = {"HB": ["HB"], "TR": ["TR", "TRh", "TRm", "TRb"], "AV": ["AV"],
            "VLP": ["VLP"], "EX": ["EX"], "RUIDO": ["RUIDO"], "NTR": ["NTR"],
            "TO": ["TO"], "RE": ["RE"], "OT": ["OT"]}


def escalar_sistema(d: pd.DataFrame) -> pd.DataFrame:
    """Mediana e IQR de todos los eventos de cada sistema (sin etiquetas)."""
    out = []
    for _, g in d.groupby("sis"):
        x = g[F16].astype(float)
        x = x.fillna(x.median())
        iqr = (x.quantile(.75) - x.quantile(.25)).replace(0, 1.0)
        out.append(g.assign(**((x - x.median()) / iqr).clip(-20, 20)))
    return pd.concat(out)


def X(g):
    return g[F16].astype(float).to_numpy()


MODELO = "logistica"


def ajustar(tr: pd.DataFrame, y, modo):
    """y: etiqueta de cada evento ("C", "LP" o "VT"). Multiclase, para que una
    clase intermedia (HB entre LP y VT) tenga su propia region; devuelve si
    el evento se predice como C."""
    x = X(tr)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isnan(x), med, x)
    mu, sd = (x.mean(0), x.std(0) + 1e-9) if modo == "fuente" else (0.0, 1.0)
    if MODELO == "logistica":
        m = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           class_weight="balanced", random_state=SEMILLA)
    m.fit((x - mu) / sd, y)
    return lambda g: m.predict((np.where(np.isnan(X(g)), med, X(g)) - mu) / sd) == "C"


def puntaje(pred_c, es_c):
    rec = pred_c[es_c].mean() if es_c.any() else np.nan
    esp = (~pred_c[~es_c]).mean() if (~es_c).any() else np.nan
    return rec, esp, np.nanmean([rec, esp])


def sistemas_con(d, clases):
    n = d.groupby(["sis", "clase"]).size()
    out = {}
    for (s, c), k in n.items():
        if c in clases and k >= SOPORTE:
            out[s] = c
    return out


def evaluar(d, positivas: dict, rng, modo) -> pd.DataFrame:
    """positivas: sistema -> etiqueta que cuenta como C en ese sistema."""
    filas = []
    glob = {s: g for s, g in d[d.clase.isin(["LP", "VT"])].groupby("sis")}
    for src, c_src in positivas.items():
        g_src = d[d.sis == src]
        pos = g_src[g_src.clase == c_src]
        neg = g_src[g_src.clase.isin(["LP", "VT"]) & (g_src.clase != c_src)]
        neg = neg.groupby("clase").filter(lambda h: len(h) >= SOPORTE)
        if len(neg) == 0:
            continue
        for tgt, c_tgt in positivas.items():
            res = []
            for r in range(REPS):
                s = int(rng.integers(1e9))
                p = pos.sample(min(N_TRAIN, len(pos)), random_state=s)
                n_ = pd.concat([h.sample(min(N_TRAIN, len(h)), random_state=s)
                                for _, h in neg.groupby("clase")])
                if tgt == src:
                    # validacion cruzada dentro del sistema
                    t = pd.concat([p, n_])
                    y = (t.clase == c_src).to_numpy()
                    lab = np.where(y, "C", t.clase.to_numpy())
                    pred = np.zeros(len(t), bool)
                    for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(t, lab):
                        pred[b] = ajustar(t.iloc[a], lab[a], modo)(t.iloc[b])
                    res.append(puntaje(pred, y))
                    continue
                f = ajustar(pd.concat([p, n_]), np.r_[np.full(len(p), "C"), n_.clase.to_numpy()], modo)
                g_t = d[d.sis == tgt]
                t = pd.concat([g_t[g_t.clase == c_tgt], glob.get(tgt, g_t.iloc[:0])])
                t = t[t.clase.isin([c_tgt, "LP", "VT"])]
                t = t.groupby("clase").filter(lambda h: len(h) >= SOPORTE)
                y = (t.clase == c_tgt).to_numpy()
                res.append(puntaje(f(t).astype(bool), y))
            rec, esp, bal = np.nanmean(np.array(res, float), axis=0)
            filas.append(dict(fuente=src, destino=tgt, rec=rec, esp=esp, bal=bal))
    return pd.DataFrame(filas)


def catalogo(s: str) -> str:
    return s.split(":", 1)[0]


def resumir(m: pd.DataFrame) -> dict:
    m = m.assign(mismo=m.fuente.map(catalogo) == m.destino.map(catalogo))
    dia = m[m.fuente == m.destino]
    intra = m[(m.fuente != m.destino) & m.mismo]
    inter = m[~m.mismo]
    return dict(n_sistemas=m.destino.nunique(), n_catalogos=m.destino.map(catalogo).nunique(),
                diagonal=dia.bal.median(), mismo_catalogo=intra.bal.median(),
                entre_catalogos=inter.bal.median(),
                rec_entre=inter.rec.median(), esp_entre=inter.esp.median(),
                n_pares_entre=len(inter))


def main():
    global MODELO
    MODELO = sys.argv[1] if len(sys.argv) > 1 else "logistica"
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(SEMILLA)
    d = datos("fija")
    cat = d.origen.replace({"OVDAS": "OVDAS"})
    d = d.assign(sis=cat + ":" + d.sistema)
    d = pd.concat([g.sample(min(len(g), TOPE), random_state=SEMILLA)
                   for _, g in d.groupby(["sis", "clase"])])
    filas, mats = [], []
    for modo, dd in (("fuente", d), ("sistema", escalar_sistema(d))):
        # referencia: VT contra LP
        m = evaluar(dd, sistemas_con(dd, ["VT"]), rng, modo)
        filas.append(dict(candidata="VT vs LP (referencia)", escalado=modo, **resumir(m)))
        mats.append(m.assign(candidata="VT", escalado=modo))
        for fam, clases in FAMILIAS.items():
            pos = sistemas_con(dd, clases)
            if len(pos) < 2:
                continue
            m = evaluar(dd, pos, rng, modo)
            if m.empty:
                continue
            filas.append(dict(candidata=fam, escalado=modo, **resumir(m)))
            mats.append(m.assign(candidata=fam, escalado=modo))
        print(f"escalado {modo} listo", flush=True)
    r = pd.DataFrame(filas)
    pd.set_option("display.width", 220)
    print(r.round(3).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / f"candidatos_globales_{MODELO}.csv", index=False)
    pd.concat(mats).to_csv(OUT / f"candidatos_globales_pares_{MODELO}.csv", index=False)


if __name__ == "__main__":
    main()
