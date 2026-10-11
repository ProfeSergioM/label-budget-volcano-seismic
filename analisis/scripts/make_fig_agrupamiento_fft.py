"""
Figura de agrupamiento_fft.py: espectro (FFT) frente a los 16 descriptores
(F16), en los mismos eventos, por dataset.

(a) ARI de HDBSCAN con las etiquetas del experto.
(b) Fraccion de eventos que HDBSCAN deja como ruido.
(c) Exactitud balanceada supervisada (regresion logistica, 5 pliegues) con
    todas las clases del dataset (control).

Uso:
    python scripts/make_fig_agrupamiento_fft.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "experiments" / "reports"
EN = {"Hawái": "Hawai`i", "Norte de California": "N. California",
      "Nevados de Chillan": "Nevados de Chillán", "Decepcion09": "Deception 2009",
      "Decepcion95": "Deception 1995"}
COL = {"F16": "#3b75af", "FFT": "#e08a1e"}


def main():
    r = pd.read_csv(REP / "agrupamiento_fft_resumen.csv")
    h = r[r.metodo == "HDBSCAN"].pivot_table(index="sistema", columns="representacion",
                                             values=["ARI_todos", "frac_ruido", "ba_todas"])
    h = h.loc[h["ARI_todos"].max(axis=1).sort_values().index]
    y = np.arange(len(h))
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    for ax, (col, titulo, xl) in zip(axes, [("ARI_todos", "(a) HDBSCAN vs. labels", "adjusted Rand index"),
                                            ("frac_ruido", "(b) Events left as noise", "fraction of events"),
                                            ("ba_todas", "(c) Supervised control", "balanced accuracy, all classes")]):
        for k, rep in enumerate(("F16", "FFT")):
            ax.barh(y + (k - 0.5) * 0.38, h[col][rep], height=0.36, color=COL[rep], label=rep)
        ax.set_title(titulo, fontsize=12, loc="left")
        ax.set_xlabel(xl, fontsize=11)
        ax.grid(axis="x", alpha=.3)
        ax.axvline(0, color="k", lw=0.8)
    axes[0].set_yticks(y, [EN.get(s, s) for s in h.index], fontsize=10)
    axes[0].legend(fontsize=10, loc="lower right")
    axes[0].set_xlim(-0.1, 1.0)
    axes[1].set_xlim(0, 1.0)
    axes[2].set_xlim(0, 1.0)
    fig.tight_layout()
    for ext, dpi in (("pdf", 300), ("png", 200)):
        fig.savefig(REP / f"fig_agrupamiento_fft.{ext}", dpi=dpi)
    print("->", REP / "fig_agrupamiento_fft.png")


if __name__ == "__main__":
    main()
