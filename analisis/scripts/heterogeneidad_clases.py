"""
Paso 1 de la revision de clases mezcladas: cuantos grupos hay dentro de
cada etiqueta de cada sistema.

Para cada combinacion sistema-clase con al menos N_FIJO eventos, un modelo
de mezcla gaussiana (covarianza completa) sobre los 16 descriptores,
escalados con la mediana y el IQR de la propia clase y reducidos por PCA al
90 % de la varianza; el numero de componentes (1 a K_MAX) se elige por BIC.
Como el BIC premia mas componentes cuanto mas eventos hay, cada clase se
evalua sobre submuestras del mismo tamano (N_FIJO, REPS veces), y LP y VT de
cada sistema sirven de referencia de una clase "limpia".

Despues, para las clases pedidas (--detalle, por defecto EX y OT), el modelo
se ajusta con todos sus eventos y se describe cada componente (tamano,
medianas de FI, DET, fpeak, SR, duracion y SNR); la asignacion queda en un
csv para revisar a mano (paso 3).

Cada sistema va por separado; nunca se juntan volcanes.

Uso:
    python scripts/heterogeneidad_clases.py [--ventana fija] [--detalle EX OT]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ranking_transferibilidad import F16, OVDAS  # noqa: E402

OUT = ROOT / "experiments" / "reports"
N_FIJO, REPS, K_MAX = 100, 20, 6
SEMILLA = 41
TIPO_TR = {"Harmonic Tremor": "TRh", "Monochromatic Tremor": "TRm",
           "Broadband Tremor": "TRb"}


def datos(ventana: str) -> pd.DataFrame:
    partes = []
    for key, (nombre, pool) in OVDAS.items():
        ruta = (f"data/processed/features_dur/features_{key}{pool}.parquet" if ventana == "coda"
                else f"data/processed/features/features_{key}.parquet")
        d = pd.read_parquet(ROOT / ruta).drop_duplicates("event_id")
        partes.append(pd.DataFrame(dict(origen="OVDAS", sistema=nombre, item_id=d.event_id.astype(str),
                                        clase=d.label.replace({"LV": "VLP"}),
                                        dur=d.duration_s_original, snr_db=d.snr_db))
                      .join(d[F16].reset_index(drop=True).set_index(d.index)))
    base = ROOT / "data/processed/features_manual"
    v = pd.read_parquet(base / f"features16_{ventana}.parquet")
    v = v[(v.motivo == "") & ~v.duplicado.fillna(False).astype(bool)].copy()
    dec = v.sistema == "Decepcion"
    v.loc[dec, "sistema"] = "Decepcion" + v.loc[dec, "item_id"].str.extract(r"dec\.(\d\d)")[0]
    partes.append(pd.DataFrame(dict(origen=v.dataset, sistema=v.sistema, item_id=v.item_id,
                                    clase=v.clase, dur=v.fin.astype(float) - v.onset.astype(float),
                                    snr_db=v.snr_db)).join(v[F16]))
    if ventana == "fija":
        c = pd.read_parquet(base / "features16_continuas.parquet")
        c = c[c.motivo == ""]
        clase = np.where(c.dataset == "voissnet", c.clase_corpus.map(TIPO_TR).fillna(c.clase), c.clase)
        partes.append(pd.DataFrame(dict(origen=c.dataset, sistema=c.sistema, item_id=c.item_id,
                                        clase=clase, dur=np.nan, snr_db=np.nan)).join(c[F16]))
    return pd.concat(partes, ignore_index=True)


def matriz(g: pd.DataFrame) -> np.ndarray:
    x = g[F16].astype(float)
    x = x.fillna(x.median())
    iqr = (x.quantile(0.75) - x.quantile(0.25)).replace(0, 1.0)
    z = ((x - x.median()) / iqr).clip(-10, 10).fillna(0.0).to_numpy()
    return PCA(n_components=0.9, svd_solver="full").fit_transform(z)


def k_bic(z: np.ndarray, rng) -> tuple[int, float]:
    bics = []
    for k in range(1, K_MAX + 1):
        gm = GaussianMixture(k, covariance_type="full", n_init=3, reg_covar=1e-3,
                             random_state=int(rng.integers(1e9))).fit(z)
        bics.append(gm.bic(z))
    k = int(np.argmin(bics)) + 1
    return k, bics[0] - min(bics)


def resumen(d: pd.DataFrame, rng) -> pd.DataFrame:
    filas = []
    for (org, sis, cl), g in d.groupby(["origen", "sistema", "clase"]):
        if len(g) < N_FIJO:
            continue
        ks, ds = [], []
        for _ in range(REPS):
            s = g.sample(N_FIJO, random_state=int(rng.integers(1e9)))
            k, db = k_bic(matriz(s), rng)
            ks.append(k)
            ds.append(db)
        filas.append(dict(origen=org, sistema=sis, clase=cl, n=len(g),
                          k_mediana=float(np.median(ks)), frac_k_mayor_1=float(np.mean(np.array(ks) > 1)),
                          dbic_mediana=float(np.median(ds))))
    return pd.DataFrame(filas).sort_values(["origen", "sistema", "clase"])


def detalle(d: pd.DataFrame, clases: list[str], rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp, asig = [], []
    for (org, sis, cl), g in d[d.clase.isin(clases)].groupby(["origen", "sistema", "clase"]):
        if len(g) < 50:
            continue
        z = matriz(g)
        k, _ = k_bic(z, rng)
        gm = GaussianMixture(k, covariance_type="full", n_init=5, reg_covar=1e-3,
                             random_state=SEMILLA).fit(z)
        lab = gm.predict(z)
        p = gm.predict_proba(z).max(axis=1)
        g = g.assign(componente=lab, prob=p)
        asig.append(g[["origen", "sistema", "clase", "item_id", "componente", "prob"]])
        for c, h in g.groupby("componente"):
            comp.append(dict(origen=org, sistema=sis, clase=cl, k=k, componente=c, n=len(h),
                             frac=len(h) / len(g), FI=h.FI.median(), DET=h.DET.median(),
                             fpeak=h.fpeak.median(), SR=h.SR.median(),
                             dur_s=h.dur.median(), snr_db=h.snr_db.median()))
    return pd.DataFrame(comp), pd.concat(asig) if asig else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ventana", default="fija", choices=["fija", "coda"])
    ap.add_argument("--detalle", nargs="*", default=["EX", "OT"])
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(SEMILLA)
    d = datos(a.ventana)
    r = resumen(d, rng)
    pd.set_option("display.width", 200)
    print(f"=== ventana {a.ventana}: grupos por clase (submuestras de {N_FIJO}, {REPS} veces) ===")
    print(r.round(2).to_string(index=False))
    comp, asig = detalle(d, a.detalle, rng)
    print(f"\n=== componentes de {', '.join(a.detalle)} (todos los eventos) ===")
    print(comp.round(2).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / f"heterogeneidad_{a.ventana}.csv", index=False)
    comp.to_csv(OUT / f"heterogeneidad_componentes_{a.ventana}.csv", index=False)
    asig.to_csv(OUT / f"heterogeneidad_asignacion_{a.ventana}.csv", index=False)


if __name__ == "__main__":
    main()
