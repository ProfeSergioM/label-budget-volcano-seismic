"""
Figura de la curva de presupuesto de etiquetas (presupuesto_etiquetas.py):
un panel por destino, Gm frente al numero de etiquetas del destino.

Lineas: mediana sobre fuentes del Gm promedio sobre repeticiones.
  - logistica, XGBoost y TabPFN con fuente + n al azar (continuas);
  - logistica con fuente + n por aprendizaje activo (punteada);
  - TabPFN solo con n del destino, al azar (discontinua) y estratificados
    (punto-raya);
  - banda gris: rango de los tres modelos locales (hasta 100 por clase).
Paneles ordenados por el Gm de la logistica sin etiquetas del destino. El eje
x es categorico (0, 10, 20, 50, 100, 200); cuando el pool del destino tiene
menos de 200 eventos (Cascade, Ruiz-Olleta) el ultimo punto usa el pool entero.

Uso:
    python scripts/make_fig_presupuesto.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "experiments" / "reports"
EN = {"Hawái": "Hawai`i", "Norte de California": "Northern California",
      "Nevados de Chillan": "Nevados de Chillán"}
CURVAS = [  # (modelo, seleccion, etiqueta, color, estilo)
    ("log", "azar", "LR, source + random", "#3b75af", "-"),
    ("log", "activo", "LR, source + active", "#3b75af", ":"),
    ("xgb", "azar", "XGBoost, source + random", "#e08a1e", "-"),
    ("tabpfn", "azar", "TabPFN, source + random", "#2e8b57", "-"),
    ("tabpfn", "solo_destino", "TabPFN, target only", "#2e8b57", "--"),
    ("tabpfn", "solo_destino_estrat", "TabPFN, target only, stratified", "#2e8b57", "-."),
]
TICKS = [0, 10, 20, 50, 100, 200]


def main():
    d = pd.read_parquet(REP / "presupuesto_etiquetas.parquet")
    m = (d.groupby(["destino", "fuente", "modelo", "seleccion", "n"]).gm.mean()
         .groupby(["destino", "modelo", "seleccion", "n"]).median().reset_index())
    orden = (m[(m.modelo == "log") & (m.seleccion == "azar") & (m.n == 0)]
             .sort_values("gm", ascending=False).destino.tolist())
    plt.rcParams.update({"font.size": 11})
    fig, ejes = plt.subplots(2, 5, figsize=(14, 6.4), sharey=True, sharex=True)
    for ax, dst in zip(ejes.flat, orden):
        h = m[m.destino == dst]
        loc = h[h.seleccion == "local"].gm
        ax.axhspan(loc.min(), loc.max(), color="0.6", alpha=.35, lw=0,
                   label="local models (≤100 per class)")
        for mod, sel, lab, col, ls in CURVAS:
            q = h[(h.modelo == mod) & (h.seleccion == sel)].sort_values("n")
            x = [TICKS.index(min((t for t in TICKS if t >= v), default=200)) for v in q.n]
            ax.plot(x, q.gm, ls, color=col, marker="o", ms=3, lw=1.6, label=lab)
        ax.set_xlim(-0.3, len(TICKS) - 0.7)
        ax.set_xticks(range(len(TICKS)))
        ax.set_xticklabels(TICKS)
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=.3)
        ax.set_title(EN.get(dst, dst), fontsize=12)
    for ax in ejes[-1]:
        ax.set_xlabel("target labels (n)")
    for ax in ejes[:, 0]:
        ax.set_ylabel("Gm (LP, VT)")
    h, l = ejes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(REP / f"fig_presupuesto.{ext}", dpi=200)
    print("->", REP / "fig_presupuesto.pdf")


if __name__ == "__main__":
    main()
