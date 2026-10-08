"""
Figura de las dos ventanas sobre un mismo evento (Sec. "Coda-length
window"): la ventana fija de 10 s y la ventana hasta el fin de la coda,
cada una con su filtro, su ventana de ruido y sus margenes de borde.

El evento es un LP de Hudson cuya duracion del analista excede con holgura
los 10 s, para que la diferencia entre ambas ventanas se vea. Reutiliza
rqa_core y la descarga de make_fig_window.py, de modo que lo dibujado es lo
que hace el pipeline.

Uso:
    python scripts/make_fig_coda_window.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from obspy import UTCDateTime  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import rqa_core as C  # noqa: E402
from class_style import ROW_SHADE_COLOR  # noqa: E402
from download_and_extract import fetch_block, make_clients  # noqa: E402
from make_fig_window import (CONFIG_PATH, COL_EDGE, COL_NOISE, COL_SIGNAL,  # noqa: E402
                             COL_TRACE, OUT_DIR, station_sensitivity)

# solo para elegir el evento; la eleccion no depende de la banda
DUR = ROOT / "data" / "processed" / "features_dur_banda01" / "features_hudson.parquet"


def pick_event(dur_min=22.0, dur_max=32.0, snr_min=16.0, snr_max=26.0):
    d = pd.read_parquet(DUR)
    d = d[(d.label == "LP") & d.duration_s_original.between(dur_min, dur_max)
          & d.snr_db.between(snr_min, snr_max)].sort_values("snr_db", ascending=False)
    return d.iloc[len(d) // 2]


def segment(vcfg, onset, pre_s, post_s, sr):
    clients = make_clients(vcfg)
    st = fetch_block(clients, vcfg, onset - pre_s - 5, onset + post_s + 5)
    if st is None:
        raise RuntimeError("sin datos")
    n = int(round((pre_s + post_s) * sr))
    for tr in st:
        if tr.stats.sampling_rate == sr and tr.stats.starttime <= onset - pre_s \
                and tr.stats.endtime >= onset + post_s:
            x = tr.slice(onset - pre_s, onset + post_s).data
            if len(x) >= n:
                return np.asarray(x[:n], dtype=np.float64)
    raise RuntimeError("ningun trace cubre el segmento")


FS_AXIS, FS_TICK, FS_ANNOT = 12, 11, 10          # estilo de fig_spectral
VEL = r"$v$ ($\mu$m s$^{-1}$)"


def panel(ax, t, y, x0, x1, signal, noise, label_signal, label_noise):
    ax.plot(t, y, color=COL_TRACE, linewidth=0.5, zorder=3)
    ax.axvspan(x0, x0 + C.EDGE_PAD_S, facecolor=COL_EDGE, alpha=0.10, zorder=0)
    ax.axvspan(signal[1], signal[1] + C.EDGE_PAD_S, facecolor=COL_EDGE, alpha=0.10, zorder=0)
    ax.axvspan(*signal, facecolor=COL_SIGNAL, alpha=0.14, zorder=0)
    ax.axvspan(*noise, facecolor=COL_NOISE, alpha=0.30, zorder=0)
    ax.axvline(0, color="black", linewidth=0.9, linestyle=":", zorder=5)
    ax.set_ylabel(VEL, fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    ax.set_title(label_signal, fontsize=FS_ANNOT, loc="left", pad=4)
    ax.annotate(label_noise, xy=(sum(noise) / 2, 0.97), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=FS_ANNOT - 1, color="#4a4a4a")


def build():
    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
    vcfg = cfg["hudson"]
    sr = float(vcfg["sps"])
    ev = pick_event()
    onset = UTCDateTime(pd.Timestamp(ev["origin_time"]).isoformat())
    dur = float(ev["duration_s_original"])
    print(f"evento: {ev['event_id']} LP hudson onset={onset} duracion={dur:.1f} s")
    plan = C.duration_window_plan(dur, sr)
    sens = station_sensitivity(vcfg["ref_station_channel"])
    pre, post = plan["pre_s"], plan["post_s"]
    raw = segment(vcfg, onset, pre, post, sr) * sens
    t = np.arange(len(raw)) / sr - pre
    # Las dos ventanas comparten la banda (0.5-10 Hz): una sola traza
    # acondicionada, y cada panel marca los intervalos de su ventana.
    y = C.preprocess_segment(raw, sr, band=C.DURATION_FILTER_BAND)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.35})
    panel(axes[0], t, y, -C.PADDED_PRE_S, C.PADDED_POST_S, (0, C.WINDOW_POST_S),
          (-C.NOISE_PRE_S, -C.NOISE_MARGIN_S),
          "Fixed-length window: 10 s from the onset", "noise\n2 s")
    panel(axes[1], t, y, -pre, post, (0, plan["n_window"] / sr),
          (-(C.DUR_NOISE_WIN_S + C.DUR_NOISE_MARGIN_S), -C.DUR_NOISE_MARGIN_S),
          f"Coda-length window: {plan['n_window'] / sr:.0f} s, onset to end of coda",
          "noise\n5 s")
    axes[1].axvline(dur, color="black", linewidth=0.9, linestyle="--", zorder=5)
    axes[1].set_xlabel("Time relative to the onset (s)", fontsize=FS_AXIS)
    axes[1].set_xlim(-pre, post)
    fig.subplots_adjust(top=0.93, left=0.12, right=0.985, bottom=0.12)
    fig.align_ylabels(axes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_coda_window.pdf", dpi=300)
    fig.savefig(OUT_DIR / "fig_coda_window.png", dpi=200)
    print(f"-> {OUT_DIR / 'fig_coda_window.pdf'}")
    return ev


if __name__ == "__main__":
    build()
