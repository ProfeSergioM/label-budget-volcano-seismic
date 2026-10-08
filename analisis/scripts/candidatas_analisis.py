"""
Revision a fondo de las clases candidatas a globales (aparte de LP y VT),
con foco en HB y en el tremor, sobre los mismos datos y el mismo detector
multiclase de candidatos_globales.py (C contra LP y VT del mismo sistema,
regresion logistica sobre los 16 descriptores, ventana fija).

Diferencias con candidatos_globales.py, que corrigen o desglosan lo que alli
queda agregado:
  - cada unidad positiva es una pareja sistema-clase; los subtipos de tremor
    de VOISS-Net (TRh, TRm, TRb) van cada uno por separado (alli se elige uno
    solo por sistema, el ultimo en orden alfabetico con soporte);
  - la especificidad se desglosa en LP y VT (alli se juntan, y en destinos sin
    LP ni VT el puntaje es solo el recall);
  - matriz de confusion completa por par: a que clase van los C (LP o VT) y
    que fraccion de LP y de VT se lleva C;
  - AUC de la probabilidad de C contra LP y contra VT en el destino, que no
    depende del umbral ni de la calibracion entre sistemas;
  - la referencia LP-VT se evalua solo entre sistemas que tienen LP y VT, y
    ademas restringida a los sistemas con HB, para comparar la caida entre
    centros de HB con la de LP-VT en los mismos volcanes.

Ademas, una prueba univariada (AUC por descriptor y sistema de HB contra LP,
HB contra VT y LP contra VT, y de cada tremor contra LP y VT), con el sentido
comparado entre centros como en ranking_transferibilidad.py, y la posicion de
la mediana de cada clase en el eje LP->VT de cada descriptor (0 = mediana de
LP, 1 = mediana de VT), por sistema.

Dos tablas de lectura directa, para no sobreinterpretar los resumenes:
  - candidatas_analisis_contraste: HB frente a la referencia LP-VT en el
    mismo par de volcanes (misma fuente y mismo destino), con metricas de
    umbral (bal_macro, recall, LP_a_C) y con AUC por separado. Entre centros
    las metricas de umbral caen para ambas (HB y LP-VT no se distinguen);
    solo el AUC separa LP-VT (que conserva el orden) de HB (que no).
  - candidatas_analisis_tremor_detector: tremor entre catalogos desglosado
    por detector de origen (sistema-clase) y catalogo de destino, con el
    rango (min, mediana, max) de cada metrica; los detectores de un mismo
    catalogo no se comportan igual (p. ej. Ruiz-Bis y Ruiz-Olleta).

Cada sistema va por separado; los modelos se entrenan en un solo sistema y
nunca se juntan clases de distintos catalogos.

Uso:
    python scripts/candidatas_analisis.py [logistica|arboles]
Salidas: experiments/reports/candidatas_analisis_*.csv (conteos, univariado,
consistencia, posicion, pares, resumen, contraste, tremor_detector) (sufijo _arboles en
las tablas multivariadas si se pide ese modelo).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from candidatos_globales import escalar_sistema  # noqa: E402
from heterogeneidad_clases import datos  # noqa: E402
from ranking_transferibilidad import F16  # noqa: E402

OUT = ROOT / "experiments" / "reports"
SOPORTE, N_TRAIN, REPS, TOPE = 50, 200, 5, 3000
SEMILLA = 41
TREMORES = ["TR", "TRh", "TRm", "TRb"]
MODELO = "logistica"


def catalogo(s: str) -> str:
    return s.split(":", 1)[0]


def cargar() -> pd.DataFrame:
    """Igual que candidatos_globales.py: ventana fija, tope de 3000 por
    sistema-clase con la misma semilla."""
    d = datos("fija")
    d = d.assign(sis=d.origen + ":" + d.sistema)
    return pd.concat([g.sample(min(len(g), TOPE), random_state=SEMILLA)
                      for _, g in d.groupby(["sis", "clase"])])


def conteos(d: pd.DataFrame) -> dict:
    return d.groupby(["sis", "clase"]).size().to_dict()


def unidades(n: dict, clases: list[str]) -> list[tuple[str, str]]:
    return sorted((s, c) for (s, c), k in n.items() if c in clases and k >= SOPORTE)


def globales(n: dict, s: str, excluir: str = "") -> list[str]:
    return [c for c in ("LP", "VT") if c != excluir and n.get((s, c), 0) >= SOPORTE]


# ---------------------------------------------------------------- detector

def ajustar(tr: pd.DataFrame, y: np.ndarray, modo: str):
    """Multiclase C/LP/VT; devuelve (prediccion, probabilidad de C)."""
    x = tr[F16].astype(float).to_numpy()
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
    ic = list(m.classes_).index("C")

    def f(g):
        z = g[F16].astype(float).to_numpy()
        z = (np.where(np.isnan(z), med, z) - mu) / sd
        return m.predict(z), m.predict_proba(z)[:, ic]
    return f


def metricas(pred, pc, lab) -> dict:
    """lab: etiqueta verdadera en el destino ("C", "LP" o "VT")."""
    r = {}
    es_c = lab == "C"
    r["n_C"] = int(es_c.sum())
    r["rec"] = np.mean(pred[es_c] == "C")
    r["C_a_LP"] = np.mean(pred[es_c] == "LP")
    r["C_a_VT"] = np.mean(pred[es_c] == "VT")
    esp = []
    for g in ("LP", "VT"):
        k = lab == g
        r[f"n_{g}"] = int(k.sum())
        if k.any():
            r[f"{g}_a_C"] = np.mean(pred[k] == "C")
            esp.append(1 - r[f"{g}_a_C"])
            r[f"auc_{g}"] = roc_auc_score(np.r_[np.ones(es_c.sum()), np.zeros(k.sum())],
                                          np.r_[pc[es_c], pc[k]])
        else:
            r[f"{g}_a_C"] = r[f"auc_{g}"] = np.nan
    r["bal_macro"] = np.mean([r["rec"], np.mean(esp)]) if esp else np.nan
    return r


def evaluar(d: pd.DataFrame, n: dict, fuentes, destinos, modo, rng) -> pd.DataFrame:
    """fuentes/destinos: listas de (sistema, clase positiva)."""
    filas = []
    for src, c_src in fuentes:
        g_src = d[d.sis == src]
        neg_cl = globales(n, src, c_src)
        if not neg_cl:
            continue
        pos = g_src[g_src.clase == c_src]
        neg = {c: g_src[g_src.clase == c] for c in neg_cl}
        for tgt, c_tgt in destinos:
            res = []
            for _ in range(REPS):
                s = int(rng.integers(1e9))
                p = pos.sample(min(N_TRAIN, len(pos)), random_state=s)
                ng = pd.concat([h.sample(min(N_TRAIN, len(h)), random_state=s) for h in neg.values()])
                t = pd.concat([p, ng])
                lab = np.where(t.clase == c_src, "C", t.clase.to_numpy())
                if (tgt, c_tgt) == (src, c_src):
                    pred = np.empty(len(t), object)
                    pc = np.zeros(len(t))
                    for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(t, lab):
                        pred[b], pc[b] = ajustar(t.iloc[a], lab[a], modo)(t.iloc[b])
                    res.append(metricas(pred, pc, lab))
                    continue
                f = ajustar(t, lab, modo)
                g_t = d[d.sis == tgt]
                # destino: C del destino y sus LP y VT con soporte (sin duplicar
                # cuando C es VT)
                cl_t = [c_tgt] + globales(n, tgt, c_tgt)
                tt = g_t[g_t.clase.isin(cl_t)]
                lab_t = np.where(tt.clase == c_tgt, "C", tt.clase.to_numpy())
                pred, pc = f(tt)
                res.append(metricas(pred, pc, lab_t))
            r = pd.DataFrame(res).mean(numeric_only=True).to_dict()
            par = ("diagonal" if (src, c_src) == (tgt, c_tgt) else
                   "mismo_sistema_otra_clase" if src == tgt else
                   "mismo_catalogo" if catalogo(src) == catalogo(tgt) else "entre_catalogos")
            filas.append(dict(fuente=src, clase_fuente=c_src, destino=tgt, clase_destino=c_tgt,
                              par=par, neg_fuente="+".join(neg_cl),
                              neg_destino="+".join(globales(n, tgt, c_tgt)), **r))
    return pd.DataFrame(filas)


def multivariado(d, n, rng) -> pd.DataFrame:
    hb = unidades(n, ["HB"])
    tr = unidades(n, TREMORES)
    # referencia: VT contra LP, solo sistemas con LP y VT (destino comparable)
    ref = [(s, "VT") for s, c in unidades(n, ["VT"]) if n.get((s, "LP"), 0) >= SOPORTE]
    hb_sis = {s for s, _ in hb}
    mats = []
    for modo, dd in (("fuente", d), ("sistema", escalar_sistema(d))):
        mats.append(evaluar(dd, n, ref, ref, modo, rng).assign(candidata="VT_ref", escalado=modo))
        print(f"  {modo}: referencia lista", flush=True)
        mats.append(evaluar(dd, n, hb, hb, modo, rng).assign(candidata="HB", escalado=modo))
        mats.append(evaluar(dd, n, tr, tr, modo, rng).assign(candidata="TR", escalado=modo))
        print(f"  {modo}: HB y tremor listos", flush=True)
    m = pd.concat(mats, ignore_index=True)
    m["ref_en_sistemas_hb"] = (m.candidata == "VT_ref") & m.fuente.isin(hb_sis) & m.destino.isin(hb_sis)
    return m


def resumir(m: pd.DataFrame) -> pd.DataFrame:
    filas = []
    grupos = [("VT_ref", m.candidata == "VT_ref"),
              ("VT_ref (solo sistemas con HB)", m.ref_en_sistemas_hb),
              ("HB", m.candidata == "HB"),
              ("TR (SGC<->VOISS-Net)", (m.candidata == "TR")),
              ]
    for nombre, sel in grupos:
        for (esc, par), g in m[sel].groupby(["escalado", "par"]):
            filas.append(dict(candidata=nombre, escalado=esc, par=par, n_pares=len(g),
                              bal_macro=g.bal_macro.median(), rec=g.rec.median(),
                              C_a_LP=g.C_a_LP.median(), C_a_VT=g.C_a_VT.median(),
                              LP_a_C=g.LP_a_C.median(), VT_a_C=g.VT_a_C.median(),
                              auc_LP=g.auc_LP.median(), auc_VT=g.auc_VT.median(),
                              auc_LP_min=g.auc_LP.min(), auc_VT_min=g.auc_VT.min()))
    return pd.DataFrame(filas)


def contraste(m: pd.DataFrame) -> pd.DataFrame:
    """HB y referencia LP-VT lado a lado en el mismo par fuente->destino
    (solo pares donde existen ambas). Metricas de umbral y AUC por separado."""
    cols = ["bal_macro", "rec", "LP_a_C", "auc_LP"]
    hb = m[m.candidata == "HB"][["escalado", "par", "fuente", "destino"] + cols]
    rf = m[m.ref_en_sistemas_hb][["escalado", "par", "fuente", "destino"] + cols]
    t = hb.merge(rf, on=["escalado", "par", "fuente", "destino"], suffixes=("_HB", "_LPVT"))
    return t.sort_values(["escalado", "par", "fuente", "destino"]).reset_index(drop=True)


def tremor_detector(m: pd.DataFrame) -> pd.DataFrame:
    """Tremor entre catalogos por detector de origen y catalogo de destino:
    rango de lo que el detector hace con el tremor (rec) y con el LP del
    destino (LP_a_C), y el AUC tremor-LP."""
    t = m[(m.candidata == "TR") & (m.par == "entre_catalogos")].copy()
    t["detector"] = t.fuente + "/" + t.clase_fuente
    t["catalogo_destino"] = t.destino.map(catalogo)
    filas = []
    for (esc, det, cat), g in t.groupby(["escalado", "detector", "catalogo_destino"]):
        r = dict(escalado=esc, detector=det, catalogo_destino=cat, n_destinos=len(g),
                 n_con_LP=int(g.LP_a_C.notna().sum()))
        for c in ("rec", "LP_a_C", "auc_LP", "bal_macro"):
            r.update({f"{c}_min": g[c].min(), f"{c}_med": g[c].median(), f"{c}_max": g[c].max()})
        filas.append(r)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- univariado

def univariado(d: pd.DataFrame, n: dict) -> pd.DataFrame:
    """AUC por descriptor y sistema; positivo = primera clase del par."""
    pares = []
    for s, c in unidades(n, ["HB"] + TREMORES):
        for g in globales(n, s):
            pares.append((s, c, g))
    for s, _ in unidades(n, ["VT"]):
        if n.get((s, "LP"), 0) >= SOPORTE:
            pares.append((s, "LP", "VT"))
    filas = []
    for s, a, b in pares:
        g = d[(d.sis == s) & d.clase.isin([a, b])]
        for f in F16:
            v = g[[f, "clase"]].dropna()
            if v.clase.nunique() < 2:
                continue
            filas.append(dict(sis=s, centro=catalogo(s), clase_a=a, clase_b=b, descriptor=f,
                              n_a=int((v.clase == a).sum()), n_b=int((v.clase == b).sum()),
                              auc=roc_auc_score(v.clase == a, v[f])))
    return pd.DataFrame(filas)


def familia(c: str) -> str:
    return "TR*" if c in TREMORES else c


def consistencia(u: pd.DataFrame) -> pd.DataFrame:
    """Como ranking_transferibilidad: AUC orientado por el signo de la mediana
    entre sistemas, puntaje = mediana - IQR, y sentido por centro."""
    filas = []
    u = u.assign(comparacion=u.clase_a.map(familia) + "_vs_" + u.clase_b)
    for (cmp_, f), g in u.groupby(["comparacion", "descriptor"]):
        signo = 1 if np.median(g.auc) >= 0.5 else -1
        o = g.auc if signo > 0 else 1 - g.auc
        q1, med, q3 = np.percentile(o, [25, 50, 75])
        por_centro = {c: h.auc for c, h in g.groupby("centro")}
        sent = {c: ("+" if (a > .5).all() else "-" if (a < .5).all() else "mixto")
                for c, a in por_centro.items()}
        filas.append(dict(comparacion=cmp_, descriptor=f, n_sistemas=len(g),
                          n_centros=len(por_centro),
                          sentido="A>B" if signo > 0 else "A<B",
                          auc_mediana=med, iqr=q3 - q1, puntaje=med - (q3 - q1),
                          auc_min=o.min(), sistema_min=g.sis.iloc[int(np.argmin(o.to_numpy()))],
                          invertidos=int((o < .5).sum()),
                          sentido_por_centro=" ".join(f"{c}:{v}" for c, v in sorted(sent.items())),
                          mismo_sentido_centros=len(set(sent.values())) == 1 and "mixto" not in sent.values()))
    return (pd.DataFrame(filas).sort_values(["comparacion", "puntaje"], ascending=[True, False])
            .reset_index(drop=True))


def posicion(d: pd.DataFrame, n: dict) -> pd.DataFrame:
    """Mediana de cada clase en el eje LP->VT de cada descriptor, por sistema
    con LP y VT; solo descriptores donde LP y VT se separan (|AUC-0.5|>0.15)."""
    filas = []
    for s, c in unidades(n, ["HB"] + TREMORES):
        if len(globales(n, s)) < 2:
            continue
        g = d[d.sis == s]
        for f in F16:
            lp, vt, x = (g.loc[g.clase == k, f].dropna() for k in ("LP", "VT", c))
            auc = roc_auc_score(np.r_[np.ones(len(lp)), np.zeros(len(vt))], np.r_[lp, vt])
            if abs(auc - .5) <= .15:
                continue
            filas.append(dict(sis=s, clase=c, descriptor=f, auc_lp_vt=auc,
                              med_LP=lp.median(), med_VT=vt.median(), med_C=x.median(),
                              pos=(x.median() - lp.median()) / (vt.median() - lp.median())))
    return pd.DataFrame(filas)


def main():
    global MODELO
    MODELO = sys.argv[1] if len(sys.argv) > 1 else "logistica"
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    rng = np.random.default_rng(SEMILLA)
    d = cargar()
    n = conteos(d)
    OUT.mkdir(parents=True, exist_ok=True)
    suf = "" if MODELO == "logistica" else f"_{MODELO}"

    if MODELO == "logistica":
        cn = (pd.Series(n).rename("n").rename_axis(["sis", "clase"]).reset_index())
        cn = cn[cn.clase.isin(["LP", "VT", "HB"] + TREMORES)]
        cn.pivot(index="sis", columns="clase", values="n").fillna(0).astype(int) \
          .to_csv(OUT / "candidatas_analisis_conteos.csv")
        u = univariado(d, n)
        u.to_csv(OUT / "candidatas_analisis_univariado.csv", index=False)
        c = consistencia(u)
        c.to_csv(OUT / "candidatas_analisis_consistencia.csv", index=False)
        print("=== consistencia univariada (5 mejores por comparacion) ===")
        print(c.groupby("comparacion").head(5).round(3).to_string(index=False))
        p = posicion(d, n)
        p.to_csv(OUT / "candidatas_analisis_posicion.csv", index=False)
        print("\n=== posicion en el eje LP->VT (mediana por sistema-clase) ===")
        print(p.groupby(["sis", "clase"]).pos.describe()[["count", "25%", "50%", "75%"]].round(2))

    print(f"\n=== detector multiclase ({MODELO}) ===", flush=True)
    m = multivariado(d, n, rng)
    m.to_csv(OUT / f"candidatas_analisis_pares{suf}.csv", index=False)
    r = resumir(m)
    r.to_csv(OUT / f"candidatas_analisis_resumen{suf}.csv", index=False)
    print(r.round(3).to_string(index=False))
    ct = contraste(m)
    ct.to_csv(OUT / f"candidatas_analisis_contraste{suf}.csv", index=False)
    print("\n=== HB frente a LP-VT en el mismo par de volcanes ===")
    print(ct.round(2).to_string(index=False))
    print(ct.groupby(["escalado", "par"])[[c for c in ct.columns if c.endswith(("_HB", "_LPVT"))]]
          .median().round(3).to_string())
    td = tremor_detector(m)
    td.to_csv(OUT / f"candidatas_analisis_tremor_detector{suf}.csv", index=False)
    print("\n=== tremor entre catalogos por detector de origen ===")
    print(td[td.escalado == "fuente"].round(2).to_string(index=False))
    cols = ["fuente", "clase_fuente", "destino", "clase_destino", "neg_fuente", "rec", "C_a_LP",
            "C_a_VT", "LP_a_C", "VT_a_C", "bal_macro", "auc_LP", "auc_VT"]
    for cand in ("HB", "TR"):
        print(f"\n--- {cand}, escalado fuente ---")
        print(m[(m.candidata == cand) & (m.escalado == "fuente")][cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
