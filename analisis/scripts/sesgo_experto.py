"""
Sesgo del experto en un sistema: agrupamiento no supervisado (HDBSCAN, con
DBSCAN de referencia) ajustado con un indice de validez interna, y deteccion
de etiquetas probablemente erroneas con confident learning (cleanlab).

Mientras se evalua contra etiquetas, se premia reproducir al experto. Aqui las
etiquetas no intervienen en el ajuste: los parametros de HDBSCAN y DBSCAN se
eligen con Optuna (TPE) maximizando DBCV (density-based clustering
validation, Moulavi et al. 2014), y la concordancia con las etiquetas (ARI,
NMI, pureza, tabla de contingencia clase x grupo) se mide solo despues.

Representacion: los 16 descriptores, escalados dentro del sistema con la
mediana y el IQR (recortados a +-10) y reducidos por PCA al 90 % de la
varianza (principal); como variante, el plano transferible FI-DET-SR escalado
igual. DBCV usa la distancia de nucleo de Moulavi et al., que eleva las
distancias a la dimension del espacio: con los 16 descriptores crudos el
indice queda dominado por la dimension, de ahi la reduccion.

DBCV es O(n^2) en memoria y tiempo: el agrupamiento se hace sobre una
submuestra aleatoria simple (sin estratificar, para que las etiquetas no
entren en el diseno) de hasta N_SUB eventos del sistema.

Confident learning: probabilidades fuera de muestra (5 pliegues
estratificados) de XGBoost (configuracion fija) y de regresion logistica
sobre los 16 descriptores escalados, con todos los eventos del sistema;
cleanlab.filter.find_label_issues marca las etiquetas sospechosas. Se
cruzan los sospechosos con los grupos de HDBSCAN: los eventos fuera de la
submuestra reciben el grupo por mayoria de sus 15 vecinos mas cercanos en la
submuestra (incluido el ruido como un grupo mas).

Cada sistema va por separado; nunca se juntan volcanes.

Uso:
    nice -n 10 python scripts/sesgo_experto.py [--sistema Hudson] [--n-sub 4000]
        [--ensayos 80]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ranking_transferibilidad import F16, OVDAS  # noqa: E402

OUT = ROOT / "experiments" / "reports"
SEMILLA = 41
N_SUB = 4000
ENSAYOS = 80
K_VECINOS = 15
TOP_REVISION = 200
FIDETSR = ["FI", "DET", "SR"]


# ----------------------------------------------------------------------------
# datos
# ----------------------------------------------------------------------------
def cargar(sistema: str) -> pd.DataFrame:
    """Eventos de un sistema en la ventana fija (misma lectura que datos('fija'))."""
    clave = {nombre: k for k, (nombre, _) in OVDAS.items()}.get(sistema)
    if clave is not None:
        d = pd.read_parquet(ROOT / f"data/processed/features/features_{clave}.parquet")
        d = d.drop_duplicates("event_id").reset_index(drop=True)
        out = pd.DataFrame(dict(origen="OVDAS", sistema=sistema, item_id=d.event_id.astype(str),
                                clase=d.label.replace({"LV": "VLP"}),
                                dur=d.duration_s_original, snr_db=d.snr_db,
                                origin_time=d.origin_time.astype(str), station=d.station))
        return out.join(d[F16])
    from heterogeneidad_clases import datos
    d = datos("fija")
    d = d[d.sistema == sistema].reset_index(drop=True)
    if d.empty:
        raise SystemExit(f"sistema desconocido: {sistema}")
    return d


def escalar(x: pd.DataFrame) -> np.ndarray:
    x = x.astype(float)
    x = x.fillna(x.median())
    iqr = (x.quantile(0.75) - x.quantile(0.25)).replace(0, 1.0)
    return ((x - x.median()) / iqr).clip(-10, 10).fillna(0.0).to_numpy()


# ----------------------------------------------------------------------------
# DBCV (Moulavi et al. 2014)
# ----------------------------------------------------------------------------
def dbcv(D: np.ndarray, etiquetas: np.ndarray, dim: int) -> float:
    """Indice DBCV a partir de la matriz de distancias completa D.

    Distancia de nucleo de cada punto dentro de su grupo:
        core(x) = ( sum_{y != x} (1/d(x,y))^dim / (n_i - 1) )^(-1/dim)
    Distancia de alcanzabilidad mutua: max(core(a), core(b), d(a,b)).
    DSC = arista maxima entre nodos internos del arbol de expansion minima
    del grupo; DSPC = alcanzabilidad mutua minima entre nodos internos de dos
    grupos. V(C) = (min DSPC - DSC) / max(min DSPC, DSC), promediado con peso
    |C| / N, donde N incluye el ruido (etiqueta -1), que asi penaliza.
    Con menos de dos grupos el indice no esta definido y se devuelve -1.
    """
    grupos = [g for g in np.unique(etiquetas) if g != -1]
    if len(grupos) < 2:
        return -1.0
    N = len(etiquetas)
    core = np.zeros(N)
    internos, dsc, tam = {}, {}, {}
    for g in grupos:
        idx = np.flatnonzero(etiquetas == g)
        n = len(idx)
        tam[g] = n
        if n < 2:
            internos[g] = idx
            dsc[g] = 0.0
            core[idx] = np.inf
            continue
        Dg = np.maximum(D[np.ix_(idx, idx)], 1e-12)
        logs = -dim * np.log(Dg)
        np.fill_diagonal(logs, -np.inf)
        # core = exp( -(1/dim) * (logsumexp(logs) - log(n-1)) )
        core[idx] = np.exp(-(logsumexp(logs, axis=1) - np.log(n - 1)) / dim)
        mr = np.maximum(np.maximum(core[idx][:, None], core[idx][None, :]), Dg)
        np.fill_diagonal(mr, 0.0)
        mst = minimum_spanning_tree(mr).tocoo()
        grado = np.bincount(np.r_[mst.row, mst.col], minlength=n)
        interno = grado > 1
        if interno.sum() == 0:
            interno[:] = True
        ok = interno[mst.row] & interno[mst.col]
        dsc[g] = float(mst.data[ok].max()) if ok.any() else float(mst.data.max())
        internos[g] = idx[interno]
    valor = 0.0
    for g in grupos:
        a = internos[g]
        dspc = np.inf
        for h in grupos:
            if h == g:
                continue
            b = internos[h]
            mr = np.maximum(np.maximum(core[a][:, None], core[b][None, :]), D[np.ix_(a, b)])
            dspc = min(dspc, float(mr.min()))
        den = max(dspc, dsc[g])
        v = (dspc - dsc[g]) / den if den > 0 and np.isfinite(den) else -1.0
        valor += tam[g] / N * v
    return float(valor)


# ----------------------------------------------------------------------------
# agrupamiento con Optuna
# ----------------------------------------------------------------------------
def ajustar(Z: np.ndarray, D: np.ndarray, metodo: str, ensayos: int, semilla: int):
    import optuna
    from sklearn.cluster import DBSCAN, HDBSCAN
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dim = Z.shape[1]
    n = len(Z)
    # escala de distancias para eps: cuantiles de la distancia al k-esimo vecino
    kd = np.sort(D, axis=1)[:, 1:51]
    eps_lo, eps_hi = float(np.quantile(kd[:, 2], 0.05)), float(np.quantile(kd[:, 49], 0.95))
    # epsilon de HDBSCAN: hasta la mediana de la distancia al 10.o vecino;
    # valores mayores fusionan todo hasta la raiz (un solo grupo)
    eps_sel = float(np.median(kd[:, 9]))

    def construir(p):
        if metodo == "HDBSCAN":
            return HDBSCAN(min_cluster_size=p["min_cluster_size"], min_samples=p["min_samples"],
                           cluster_selection_epsilon=p["cluster_selection_epsilon"],
                           metric="precomputed", copy=True)
        return DBSCAN(eps=p["eps"], min_samples=p["min_samples"], metric="precomputed", n_jobs=1)

    def agrupar(p):
        # sklearn falla en epsilon_search cuando el epsilon sube hasta la raiz
        # del arbol; ese ensayo se trata como "todo ruido" (DBCV -1)
        try:
            return construir(p).fit_predict(D)
        except (TypeError, ValueError):
            fallos.append(p)
            return np.full(n, -1)

    filas, fallos = [], []

    def objetivo(trial):
        if metodo == "HDBSCAN":
            p = dict(min_cluster_size=trial.suggest_int("min_cluster_size", 10, max(11, n // 8), log=True),
                     min_samples=trial.suggest_int("min_samples", 1, 100, log=True),
                     cluster_selection_epsilon=trial.suggest_float("cluster_selection_epsilon", 0.0, eps_sel))
        else:
            p = dict(eps=trial.suggest_float("eps", eps_lo, eps_hi, log=True),
                     min_samples=trial.suggest_int("min_samples", 3, 100, log=True))
        et = agrupar(p)
        v = dbcv(D, et, dim)
        ng = int(len(set(et)) - (-1 in et))
        filas.append(dict(metodo=metodo, ensayo=trial.number, **p, fallo_sklearn=bool(fallos and fallos[-1] is p), n_grupos=ng,
                          frac_ruido=float(np.mean(et == -1)), dbcv=v))
        return v

    est = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=semilla))
    est.optimize(objetivo, n_trials=ensayos)
    et = agrupar(est.best_params)
    return et, est.best_params, pd.DataFrame(filas)


def concordancia(et: np.ndarray, clase: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    m = et != -1
    r = dict(ARI_todos=adjusted_rand_score(clase, et), NMI_todos=normalized_mutual_info_score(clase, et))
    if m.sum() > 1:
        r.update(ARI_sin_ruido=adjusted_rand_score(clase[m], et[m]),
                 NMI_sin_ruido=normalized_mutual_info_score(clase[m], et[m]))
    return r


def pureza(et: np.ndarray, clase: np.ndarray, etiqueta: str) -> pd.DataFrame:
    """Por clase: fraccion en su grupo dominante (sin ruido) y pureza de ese grupo."""
    t = pd.crosstab(pd.Series(clase, name="clase"), pd.Series(et, name="grupo"))
    filas = []
    for c in t.index:
        fila = t.loc[c]
        sin = fila.drop(-1, errors="ignore")
        g = sin.idxmax() if sin.sum() > 0 else -1
        filas.append(dict(agrupamiento=etiqueta, clase=c, n=int(fila.sum()),
                          frac_ruido=float(fila.get(-1, 0) / fila.sum()),
                          grupo_dominante=int(g),
                          frac_en_grupo_dominante=float(fila.get(g, 0) / fila.sum()),
                          pureza_grupo_dominante=float(t[g].loc[c] / t[g].sum()) if g != -1 else np.nan))
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# confident learning
# ----------------------------------------------------------------------------
def probabilidades(X: np.ndarray, y: np.ndarray, semilla: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from xgboost import XGBClassifier
    cv = StratifiedKFold(5, shuffle=True, random_state=semilla)
    modelos = {
        "xgb": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", n_jobs=1,
                             random_state=semilla, verbosity=0),
        "logreg": LogisticRegression(C=1.0, max_iter=2000),
    }
    return {k: cross_val_predict(m, X, y, cv=cv, method="predict_proba", n_jobs=1)
            for k, m in modelos.items()}


# ----------------------------------------------------------------------------
def figura(sistema, Z, sub, et_h, trials, cont, susp, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from class_style import CLASS_COLOR
    col = lambda c: CLASS_COLOR.get(c, "#888888")  # noqa: E731
    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))

    a = ax[0, 0]
    for m, mk in [("HDBSCAN", "o"), ("DBSCAN", "s")]:
        t = trials[(trials.metodo == m) & (trials.representacion == "PCA16")]
        s = a.scatter(t.n_grupos.clip(upper=60), t.dbcv, c=t.frac_ruido, cmap="viridis", vmin=0, vmax=1,
                      marker=mk, s=28, edgecolor="k", lw=0.3, label=m)
    a.set(xlabel="numero de grupos (recortado a 60)", ylabel="DBCV",
          title="a) Ensayos de Optuna (PCA de los 16); color = fraccion de ruido")
    a.legend()
    fig.colorbar(s, ax=a, label="fraccion de ruido")

    rng = np.random.default_rng(0)
    orden = rng.permutation(len(sub))
    a = ax[0, 1]
    paleta = plt.get_cmap("tab20")
    cg = np.array(["#d0d0d0" if g == -1 else matplotlib.colors.to_hex(paleta(g % 20)) for g in et_h])
    a.scatter(Z[orden, 0], Z[orden, 1], c=cg[orden], s=5, lw=0)
    a.set(xlabel="CP1", ylabel="CP2", title="b) Submuestra: grupos HDBSCAN (gris = ruido)")
    a = ax[0, 2]
    a.scatter(Z[orden, 0], Z[orden, 1], c=[col(c) for c in sub.clase.to_numpy()[orden]], s=5, lw=0)
    for c in sorted(sub.clase.unique()):
        a.scatter([], [], c=col(c), label=f"{c} ({(sub.clase == c).sum()})")
    a.legend(markerscale=2)
    a.set(xlabel="CP1", ylabel="CP2", title="c) Submuestra: etiqueta del experto")

    a = ax[1, 0]
    fr = cont.div(cont.sum(axis=1), axis=0)
    im = a.imshow(fr.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    a.set_xticks(range(fr.shape[1]), [("ruido" if g == -1 else str(g)) for g in fr.columns], rotation=90)
    a.set_yticks(range(fr.shape[0]), fr.index)
    for i in range(fr.shape[0]):
        for j in range(fr.shape[1]):
            if cont.iloc[i, j] > 0 and fr.shape[1] <= 25:
                a.text(j, i, int(cont.iloc[i, j]), ha="center", va="center", fontsize=7,
                       color="w" if fr.iloc[i, j] > 0.5 else "k")
    a.set(xlabel="grupo HDBSCAN", title="d) Clase x grupo (fraccion de la clase; n en celdas)")
    fig.colorbar(im, ax=a)

    a = ax[1, 1]
    s2 = susp[susp.modelo == "xgb"]
    clases = sorted(set(s2.clase) | set(s2.clase_sugerida))
    base = np.zeros(len(clases))
    for cs in clases:
        v = np.array([((s2.clase == c) & (s2.clase_sugerida == cs)).sum() for c in clases])
        a.bar(clases, v, bottom=base, color=col(cs), label=f"sugiere {cs}")
        base += v
    a.legend()
    a.set(xlabel="clase del experto", ylabel="eventos sospechosos",
          title="e) cleanlab (XGBoost): sospechosos por clase y clase sugerida")

    a = ax[1, 2]
    todos = susp.attrs["todos"]
    bins = np.linspace(np.nanquantile(todos.snr_db, 0.01), np.nanquantile(todos.snr_db, 0.99), 40)
    a.hist(todos.snr_db[~todos.sospechoso_xgb], bins=bins, density=True, histtype="step", color="k", label="resto")
    a.hist(todos.snr_db[todos.sospechoso_xgb], bins=bins, density=True, histtype="step", color="#c0392b",
           lw=1.8, label="sospechosos (XGBoost)")
    a.set(xlabel="SNR (dB)", ylabel="densidad", title="f) SNR de sospechosos frente al resto")
    a.legend()
    fig.suptitle(f"Sesgo del experto - {sistema}", fontsize=14)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sistema", default="Hudson")
    ap.add_argument("--n-sub", type=int, default=N_SUB)
    ap.add_argument("--ensayos", type=int, default=ENSAYOS)
    args = ap.parse_args()
    t0 = time.time()
    warnings.filterwarnings("ignore")
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier

    sis = args.sistema
    tag = sis.lower().replace(" ", "_")
    d = cargar(sis)
    d = d[d.clase.notna()].reset_index(drop=True)
    print(f"{sis}: {len(d)} eventos; {d.clase.value_counts().to_dict()}", flush=True)
    X = escalar(d[F16])
    pca = PCA(n_components=0.9, svd_solver="full").fit(X)
    reps = {"PCA16": pca.transform(X), "FI_DET_SR": X[:, [F16.index(f) for f in FIDETSR]]}
    print(f"PCA: {pca.n_components_} componentes (90 % de la varianza)", flush=True)

    rng = np.random.default_rng(SEMILLA)
    isub = np.sort(rng.choice(len(d), size=min(args.n_sub, len(d)), replace=False))
    sub = d.iloc[isub].reset_index(drop=True)
    y_sub = sub.clase.to_numpy()

    resumen, trials_all, pur_all, cont_all, et_principal = [], [], [], [], None
    for rep, Zall in reps.items():
        Z = Zall[isub]
        D = squareform(pdist(Z))
        for metodo in ["HDBSCAN", "DBSCAN"]:
            t1 = time.time()
            et, bp, tr = ajustar(Z, D, metodo, args.ensayos, SEMILLA)
            tr["representacion"] = rep
            trials_all.append(tr)
            ng = int(len(set(et)) - (-1 in et))
            fila = dict(sistema=sis, representacion=rep, dim=Z.shape[1], metodo=metodo, n_sub=len(Z),
                        mejores_parametros=str(bp), n_grupos=ng, frac_ruido=float(np.mean(et == -1)),
                        dbcv=dbcv(D, et, Z.shape[1]), **concordancia(et, y_sub))
            # referencia: DBCV de las propias etiquetas del experto en la misma representacion
            fila["dbcv_etiquetas_experto"] = dbcv(D, pd.factorize(y_sub)[0], Z.shape[1])
            resumen.append(fila)
            et_nom = f"{metodo}_{rep}"
            pur_all.append(pureza(et, y_sub, et_nom))
            c = pd.crosstab(pd.Series(y_sub, name="clase"), pd.Series(et, name="grupo"))
            cont_all.append(c.stack().rename("n").reset_index().assign(agrupamiento=et_nom))
            sub[f"grupo_{et_nom}"] = et
            if metodo == "HDBSCAN" and rep == "PCA16":
                et_principal, cont_principal = et, c
            print(f"  {et_nom}: {bp} -> {ng} grupos, ruido {fila['frac_ruido']:.2f}, "
                  f"DBCV {fila['dbcv']:.3f} (etiquetas {fila['dbcv_etiquetas_experto']:.3f}), "
                  f"ARI {fila['ARI_todos']:.3f}  [{time.time() - t1:.0f} s]", flush=True)

    resumen = pd.DataFrame(resumen)
    resumen.to_csv(OUT / f"sesgo_experto_{tag}_agrupamiento.csv", index=False)
    pd.concat(trials_all).to_csv(OUT / f"sesgo_experto_{tag}_optuna.csv", index=False)
    pd.concat(pur_all).to_csv(OUT / f"sesgo_experto_{tag}_pureza.csv", index=False)
    pd.concat(cont_all)[["agrupamiento", "clase", "grupo", "n"]].to_csv(
        OUT / f"sesgo_experto_{tag}_contingencia.csv", index=False)
    sub[["item_id", "clase"] + [c for c in sub.columns if c.startswith("grupo_")]].to_csv(
        OUT / f"sesgo_experto_{tag}_grupos_submuestra.csv", index=False)

    # ------------------------------------------------------------------ cleanlab
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    from sklearn.metrics import balanced_accuracy_score
    clases = np.array(sorted(d.clase.unique()))
    y = np.searchsorted(clases, d.clase.to_numpy())
    t1 = time.time()
    probs = probabilidades(X, y, SEMILLA)
    print(f"probabilidades fuera de muestra [{time.time() - t1:.0f} s]", flush=True)

    # grupo HDBSCAN (principal) para todos los eventos: los de la submuestra lo
    # tienen; el resto por mayoria de K_VECINOS vecinos en la submuestra
    knn = KNeighborsClassifier(K_VECINOS).fit(reps["PCA16"][isub], et_principal)
    grupo = knn.predict(reps["PCA16"])
    grupo[isub] = et_principal
    dom = {g: cont_principal[g].idxmax() for g in cont_principal.columns if g != -1}
    d["grupo_hdbscan"] = grupo
    d["en_submuestra"] = False
    d.loc[isub, "en_submuestra"] = True
    d["clase_dominante_grupo"] = [dom.get(g, "ruido") for g in grupo]

    tabla, por_clase, cruce = [], [], []
    for k, p in probs.items():
        mask = find_label_issues(y, p, filter_by="prune_by_noise_rate")
        q = get_label_quality_scores(y, p, method="self_confidence")
        d[f"sospechoso_{k}"] = mask
        d[f"calidad_{k}"] = q
        d[f"sugerida_{k}"] = clases[p.argmax(1)]
        print(f"  {k}: exactitud balanceada fuera de muestra {balanced_accuracy_score(y, p.argmax(1)):.3f}; "
              f"{mask.sum()} sospechosos ({mask.mean():.1%})", flush=True)
        for c in clases:
            mc = d.clase == c
            s, r = mc & mask, mc & ~mask
            fila = dict(modelo=k, clase=c, n=int(mc.sum()), n_sospechosos=int(s.sum()),
                        frac_sospechosos=float(s.sum() / mc.sum()),
                        exactitud_balanceada_modelo=balanced_accuracy_score(y, p.argmax(1)),
                        snr_mediana_sospechosos=float(d.snr_db[s].median()),
                        snr_mediana_resto=float(d.snr_db[r].median()),
                        dur_mediana_sospechosos=float(d.dur[s].median()),
                        dur_mediana_resto=float(d.dur[r].median()))
            for cs in clases:
                fila[f"sugiere_{cs}"] = int((s & (d[f"sugerida_{k}"] == cs)).sum())
            # cruce con HDBSCAN: grupo cuya clase dominante no es la del experto
            for nombre, m in [("sospechosos", s), ("resto", r)]:
                gm = m & (d.clase_dominante_grupo != "ruido")
                fila[f"frac_ruido_hdbscan_{nombre}"] = float((m & (d.clase_dominante_grupo == "ruido")).sum()
                                                             / max(m.sum(), 1))
                fila[f"frac_grupo_otra_clase_{nombre}"] = float((gm & (d.clase_dominante_grupo != c)).sum()
                                                                / max(gm.sum(), 1))
            gm = s & (d.clase_dominante_grupo != "ruido")
            fila["frac_grupo_de_clase_sugerida_sospechosos"] = float(
                (gm & (d.clase_dominante_grupo == d[f"sugerida_{k}"])).sum() / max(gm.sum(), 1))
            por_clase.append(fila)
    d["sospechoso_ambos"] = d.sospechoso_xgb & d.sospechoso_logreg

    # cruce exacto en la submuestra, para cada agrupamiento: reparto de
    # sospechosos y no sospechosos (XGBoost) de cada clase entre los grupos
    s2 = sub.merge(d[["item_id", "sospechoso_xgb", "sugerida_xgb"]], on="item_id")
    cruce = []
    for col in [c for c in sub.columns if c.startswith("grupo_")]:
        t = (s2.groupby(["clase", "sospechoso_xgb", col]).size().rename("n").reset_index()
             .rename(columns={col: "grupo"}))
        t["frac_de_la_fila"] = t.n / t.groupby(["clase", "sospechoso_xgb"]).n.transform("sum")
        dom = pd.crosstab(s2.clase, s2[col]).idxmax()
        t["clase_dominante_grupo"] = [("ruido" if g == -1 else dom[g]) for g in t.grupo]
        cruce.append(t.assign(agrupamiento=col.removeprefix("grupo_")))
    pd.concat(cruce).to_csv(OUT / f"sesgo_experto_{tag}_cruce_submuestra.csv", index=False)
    pd.DataFrame(por_clase).to_csv(OUT / f"sesgo_experto_{tag}_cleanlab_por_clase.csv", index=False)

    cols = [c for c in ["item_id", "origin_time", "station", "clase", "sugerida_xgb", "calidad_xgb",
                        "sugerida_logreg", "calidad_logreg", "sospechoso_xgb", "sospechoso_logreg",
                        "sospechoso_ambos", "snr_db", "dur", "grupo_hdbscan", "clase_dominante_grupo",
                        "en_submuestra", "FI", "DET", "SR", "fpeak"] if c in d.columns]
    rev = d[d.sospechoso_ambos].sort_values("calidad_xgb")[cols].head(TOP_REVISION)
    rev = rev.rename(columns={"item_id": "event_id"})
    rev.to_csv(OUT / f"sesgo_experto_{tag}_revision_manual.csv", index=False)
    d[cols].rename(columns={"item_id": "event_id"}).to_csv(
        OUT / f"sesgo_experto_{tag}_cleanlab_eventos.csv.gz", index=False)

    susp = d[d.sospechoso_xgb].assign(modelo="xgb", clase_sugerida=d.sugerida_xgb)
    susp.attrs["todos"] = d
    tr = pd.concat(trials_all)
    figura(sis, reps["PCA16"][isub], sub, et_principal, tr, cont_principal, susp,
           OUT / f"fig_sesgo_experto_{tag}.png")
    print(f"listo en {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
