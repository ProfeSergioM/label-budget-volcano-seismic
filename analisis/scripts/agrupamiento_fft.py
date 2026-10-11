"""
Agrupamiento no supervisado sobre el espectro de amplitud de cada ventana
(FFT), frente al mismo agrupamiento sobre los 16 descriptores, en los mismos
eventos (data/processed/ventanas, de extract_ventanas.py).

Representacion FFT: amplitud de la FFT de la ventana de 10 s z-scoreada (con
ventana de Hann), bins de 0.1 Hz entre F0 y F1 (96 valores), en log10 y
centrada por evento (quita la escala); luego PCA al 90 % de la varianza,
como en sesgo_experto.py. Representacion F16: los 16 descriptores escalados
con mediana e IQR del dataset y PCA al 90 %.

Para cada representacion: HDBSCAN (principal) y DBSCAN (referencia) con
parametros elegidos por Optuna maximizando DBCV, sin etiquetas (mismas
funciones que sesgo_experto.py); despues, ARI y NMI con las etiquetas, y el
DBCV de las propias etiquetas del experto en ese espacio. Ademas, como
control supervisado, la exactitud balanceada de una regresion logistica
(5 pliegues, hasta 300 por clase) con todas las clases y con LP-VT.

Cada dataset por separado. Salidas: experiments/reports/agrupamiento_fft_*.csv
y fig_agrupamiento_fft.png.

Uso:
    nice -n 10 python scripts/agrupamiento_fft.py [--sistemas ...] [--ensayos 80]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ranking_transferibilidad import F16  # noqa: E402
from sesgo_experto import SEMILLA, ajustar, cargar, concordancia, dbcv, escalar  # noqa: E402
from extract_ventanas import EXTERNOS, OUT as VENT, tag  # noqa: E402

REP = ROOT / "experiments" / "reports"
F0, F1 = 0.5, 10.0
SPS = 100.0


def espectro(X):
    n = X.shape[1]
    A = np.abs(np.fft.rfft(X * np.hanning(n), axis=1))
    f = np.fft.rfftfreq(n, 1 / SPS)
    m = (f >= F0) & (f <= F1)
    S = np.log10(A[:, m] + 1e-6)
    return S - S.mean(axis=1, keepdims=True), f[m]


def descriptores(sistema, item_ids):
    """Los 16 descriptores de esos eventos, en el mismo orden."""
    if sistema in EXTERNOS:
        f = pd.read_parquet(ROOT / "data/processed/features_manual/features16_fija.parquet")
        f = f.drop_duplicates("item_id").set_index("item_id")
    else:
        f = cargar(sistema).drop_duplicates("item_id").set_index("item_id")
    return f.loc[item_ids, F16]


def supervisado(Z, y, seed):
    if len(np.unique(y)) < 2:
        return np.nan
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    i = np.concatenate([rng.choice(np.flatnonzero(y == c), min(300, int((y == c).sum())), replace=False)
                        for c in np.unique(y)])
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(m, Z[i], y[i], cv=cv, scoring="balanced_accuracy").mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sistemas", nargs="*")
    ap.add_argument("--ensayos", type=int, default=80)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    from sklearn.decomposition import PCA
    rutas = sorted(VENT.glob("ventanas_*.npz"))
    if a.sistemas:
        rutas = [VENT / f"ventanas_{tag(s)}.npz" for s in a.sistemas]
    hecho = REP / "agrupamiento_fft_resumen.csv"
    resumen = pd.read_csv(hecho).to_dict("records") if hecho.exists() and not a.sistemas else []
    trials = ([pd.read_csv(REP / "agrupamiento_fft_optuna.csv")]
              if (REP / "agrupamiento_fft_optuna.csv").exists() and not a.sistemas else [])
    listos = {r["sistema"] for r in resumen}
    for ruta in rutas:
        z = np.load(ruta, allow_pickle=True)
        sis = str(z["sistema"])
        if sis in listos:
            continue
        X, y, ids = z["X"].astype(float), z["clase"].astype(str), z["item_id"].astype(str)
        vc = pd.Series(y).value_counts()
        y_ok = vc[vc >= 20].index
        if len(y_ok) < 2:
            print(f"{sis}: una sola clase con >= 20 eventos, se omite", flush=True)
            continue
        S, _ = espectro(X)
        reps = {"FFT": PCA(n_components=0.9, svd_solver="full").fit_transform(S - S.mean(0)),
                "F16": PCA(n_components=0.9, svd_solver="full").fit_transform(
                    escalar(descriptores(sis, ids)))}
        crudo = {"FFT": S, "F16": np.nan_to_num(descriptores(sis, ids).to_numpy(float))}
        print(f"{sis}: {len(y)} eventos {dict(pd.Series(y).value_counts())}; "
              f"PCA FFT {reps['FFT'].shape[1]} comp., F16 {reps['F16'].shape[1]} comp.", flush=True)
        for rep, Z in reps.items():
            D = squareform(pdist(Z))
            lpvt = np.isin(y, ["LP", "VT"])
            sup = dict(ba_todas=supervisado(crudo[rep], y, SEMILLA),
                       ba_lp_vt=(supervisado(crudo[rep][lpvt], y[lpvt], SEMILLA)
                                 if len(np.unique(y[lpvt])) == 2 and
                                 pd.Series(y[lpvt]).value_counts().min() >= 20 else np.nan))
            for metodo in ("HDBSCAN", "DBSCAN"):
                t1 = time.time()
                et, bp, tr = ajustar(Z, D, metodo, a.ensayos, SEMILLA)
                trials.append(tr.assign(sistema=sis, representacion=rep))
                ng = int(len(set(et)) - (-1 in et))
                fila = dict(sistema=sis, representacion=rep, dim=Z.shape[1], n=len(Z),
                            metodo=metodo, mejores_parametros=str(bp), n_grupos=ng,
                            frac_ruido=float(np.mean(et == -1)), dbcv=dbcv(D, et, Z.shape[1]),
                            dbcv_etiquetas_experto=dbcv(D, pd.factorize(y)[0], Z.shape[1]),
                            **concordancia(et, y), **sup)
                resumen.append(fila)
                print(f"  {rep} {metodo}: {ng} grupos, ruido {fila['frac_ruido']:.2f}, DBCV "
                      f"{fila['dbcv']:.3f} (etiquetas {fila['dbcv_etiquetas_experto']:.3f}), "
                      f"ARI {fila['ARI_todos']:.3f}; BA todas {sup['ba_todas']:.2f}, LP-VT "
                      f"{sup['ba_lp_vt']:.2f} [{time.time() - t1:.0f} s]", flush=True)
        pd.DataFrame(resumen).to_csv(REP / "agrupamiento_fft_resumen.csv", index=False)
        pd.concat(trials).to_csv(REP / "agrupamiento_fft_optuna.csv", index=False)
    print("->", REP / "agrupamiento_fft_resumen.csv")


if __name__ == "__main__":
    main()
