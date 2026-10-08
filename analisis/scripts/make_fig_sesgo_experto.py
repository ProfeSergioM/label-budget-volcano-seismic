"""
Figura resumen del sesgo del experto (sesgo_experto.py y
sesgo_entre_datasets.py).

(a) Confident learning con el criterio propio de cada sistema (XGBoost,
    probabilidades fuera de muestra): fraccion de sospechosos por sistema y
    clase; en blanco, clases ausentes o con menos de 50 eventos.
(b) Etiquetas del destino juzgadas con el criterio de otra fuente frente al
    criterio propio: fraccion de sospechosos por destino y clase (mediana
    sobre fuentes del promedio sobre repeticiones). Circulos: pares LP-VT
    (rotulados los que se alejan mas de 0.2 de la diagonal); triangulos:
    pares con HB (HB-LP o HB-LP-VT). Sobre la diagonal, la fuente
    marca mas de lo que marca el propio destino.

Uso:
    python scripts/make_fig_sesgo_experto.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "experiments" / "reports"
EN = {"Hawái": "Hawai`i", "Norte de California": "Northern California",
      "Nevados de Chillan": "Nevados de Chillán", "Decepcion09": "Deception Island 2009",
      "hudson": "Hudson", "copahue": "Copahue", "villarrica": "Villarrica",
      "laguna_del_maule": "Laguna del Maule", "nevados_de_chillan": "Nevados de Chillán",
      "ruiz-olleta": "Ruiz-Olleta", "ruiz-bis": "Ruiz-Bis", "alaska": "Alaska",
      "decepcion09": "Deception Island 2009"}
CLASES = ["LP", "VT", "HB", "AV", "VLP", "TO", "TR", "NTR", "RE", "OT"]
COLOR = {"LP": "#3b75af", "VT": "#c0392b", "HB": "#e69f00"}


def propio():
    filas = []
    for p in REP.glob("sesgo_experto_*_cleanlab_por_clase.csv"):
        t = pd.read_csv(p)
        t = t[t.modelo == "xgb"]
        sis = p.name.removeprefix("sesgo_experto_").removesuffix("_cleanlab_por_clase.csv")
        filas.append(t.assign(sistema=EN.get(sis, sis)))
    t = pd.concat(filas)
    m = t.pivot_table(index="sistema", columns="clase", values="frac_sospechosos")
    m = m[[c for c in CLASES if c in m.columns]]
    orden = m.mean(axis=1).sort_values().index
    return m.loc[orden]


def main():
    m = propio()
    f = pd.read_csv(REP / "sesgo_entre_datasets_pares.csv")
    r = (f.groupby(["destino", "clases", "clase", "criterio", "fuente"]).frac_sospechosos.mean()
         .groupby(["destino", "clases", "clase", "criterio"]).median().unstack("criterio")
         .reset_index())

    plt.rcParams.update({"font.size": 11})
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    im = a.imshow(m.to_numpy(), cmap="Reds", vmin=0, vmax=0.75, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            if np.isfinite(v):
                a.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10,
                       color="white" if v > 0.45 else "black")
    a.set_xticks(range(m.shape[1]), m.columns)
    a.set_yticks(range(m.shape[0]), m.index)
    a.set_xlabel("analyst class")
    a.set_title("(a) Suspect labels, own criterion", fontsize=12)
    fig.colorbar(im, ax=a, fraction=0.04, pad=0.02, label="fraction of the class")

    b.plot([0, 1], [0, 1], color="0.5", lw=1, ls="--")
    for _, x in r.iterrows():
        mk = "o" if x.clases == "LP-VT" else "^"
        b.scatter(x.propio, x.fuente, marker=mk, s=60, color=COLOR[x.clase],
                  edgecolor="k", lw=0.5, alpha=0.85)
    for _, x in r[(r.clases == "LP-VT") & ((r.fuente - r.propio) > 0.2)].iterrows():
        b.annotate(f"{EN.get(x.destino, x.destino)} {x.clase}", (x.propio, x.fuente),
                   textcoords="offset points", xytext=(5, -3), fontsize=8)
    for c, col in COLOR.items():
        b.scatter([], [], color=col, edgecolor="k", lw=0.5, label=c)
    b.scatter([], [], marker="o", color="w", edgecolor="k", label="LP-VT pairs")
    b.scatter([], [], marker="^", color="w", edgecolor="k", label="pairs with HB")
    b.legend(fontsize=9, loc="lower right")
    b.set_xlim(0, 0.9)
    b.set_ylim(0, 0.9)
    b.set_aspect("equal")
    b.grid(alpha=.3)
    b.set_xlabel("suspect fraction, own criterion")
    b.set_ylabel("suspect fraction, other source's criterion")
    b.set_title("(b) Target labels judged by another dataset", fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(REP / f"fig_sesgo_experto.{ext}", dpi=200)
    print("->", REP / "fig_sesgo_experto.pdf")


if __name__ == "__main__":
    main()
