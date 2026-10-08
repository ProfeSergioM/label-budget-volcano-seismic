"""
Figura de la ventana de analisis (Sec. "Signal conditioning and analysis
window"): segmento acolchado, margenes de borde del filtro, ventana de ruido
y ventana de senal, sobre una traza real descargada del servidor Winston.

Reutiliza rqa_core (constantes y filtro) y el mecanismo de descarga de
download_and_extract.py, para que lo dibujado sea exactamente lo que hace el
pipeline y no una reconstruccion aparte.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from obspy import UTCDateTime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import rqa_core as C  # noqa: E402
from download_and_extract import make_clients, fetch_block, find_covering_trace  # noqa: E402
from class_style import ROW_SHADE_COLOR  # noqa: E402

CONFIG_PATH = ROOT / "config" / "volcanoes.json"
SCNL_PATH = ROOT / "config" / "estaciones-scnl.json"
FEATURES_DIR = ROOT / "data" / "processed" / "features"
OUT_DIR = ROOT / "docs" / "figures"

FS_AXIS, FS_TICK, FS_ANNOT, FS_CREDIT = 13, 12, 11.5, 10.5

COL_TRACE = "#1a4d7a"
COL_SIGNAL = "#1a4d7a"
COL_NOISE = "#7d7d7d"
COL_EDGE = "#c0392b"


def station_sensitivity(station: str) -> float:
    """Counts -> um/s. La sensibilidad vive en estaciones-scnl.json, no en
    volcanoes.json, con la unidad declarada en el mismo registro."""
    for e in json.load(open(SCNL_PATH, encoding="utf-8")):
        if e["station"] == station:
            assert "m/s" in e["unidad"], f"unidad inesperada: {e['unidad']!r}"
            return float(e["sensibilidad"])
    raise KeyError(station)


def pick_event(volcano: str, klass: str, snr_min: float, snr_max: float):
    """Un evento de SNR alto pero no extremo: la ventana de ruido debe verse
    tranquila y la de senal clara, sin que la escala aplaste todo lo demas."""
    df = pd.read_parquet(FEATURES_DIR / f"features_{volcano}.parquet")
    sub = df[(df["label"] == klass) & df["snr_db"].between(snr_min, snr_max)]
    sub = sub.sort_values("snr_db", ascending=False)
    return sub.iloc[len(sub) // 2]


def download_padded(vcfg: dict, onset: UTCDateTime):
    clients = make_clients(vcfg)
    st = fetch_block(clients, vcfg,
                     onset - C.PADDED_PRE_S - 5, onset + C.PADDED_POST_S + 5)
    if st is None:
        raise RuntimeError("sin datos en el bloque")
    raw = find_covering_trace(st, onset, vcfg["sps"])
    if raw is None:
        raise RuntimeError("ningun trace cubre el segmento acolchado completo")
    return raw


def build(volcano="hudson", klass="VT", snr_min=8.0, snr_max=14.0):
    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
    vcfg = cfg[volcano]
    ev = pick_event(volcano, klass, snr_min, snr_max)
    onset = UTCDateTime(str(ev["origin_time"]))
    print(f"evento: {ev['event_id']} {klass} {volcano} "
          f"onset={onset} SNR={ev['snr_db']:.1f} dB")

    sens = station_sensitivity(vcfg["ref_station_channel"])
    raw = download_padded(vcfg, onset) * sens          # um/s
    # el filtro es lineal, asi que escalar antes o despues es equivalente
    filt = C.preprocess_segment(raw, vcfg["sps"])
    t = np.arange(len(raw)) / vcfg["sps"] - C.PADDED_PRE_S

    # Desde que FILTER_BAND paso a 0.5-10 Hz, la banda del SNR coincide con la
    # de analisis: un solo tramo acondicionado, dos paneles en vez de tres.
    _i = C.padded_window_indices(vcfg["sps"])
    snr_now = C.compute_snr_db(filt[_i["noise_start"]:_i["noise_end"]],
                               filt[_i["signal_start"]:_i["signal_end"]])
    print(f"SNR ({C.SNR_BAND[0]}-{C.SNR_BAND[1]:g} Hz, ruido "
          f"{C.NOISE_PRE_S - C.NOISE_MARGIN_S:g} s): {snr_now:.1f} dB")

    # Estilo de fig_spectral: marco completo, letra 12/11/10, unidades en
    # mathtext y titulo de panel arriba a la izquierda.
    fs_axis, fs_tick, fs_annot = 12, 11, 10
    vel = r"$v$ ($\mu$m s$^{-1}$)"
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True,
                             gridspec_kw={"hspace": 0.35})
    ax_r, ax_f = axes
    ax_r.plot(t, raw, color="#8a8a8a", linewidth=0.55)
    ax_f.plot(t, filt, color=COL_TRACE, linewidth=0.55)
    ax_r.set_title("Raw padded segment", fontsize=fs_annot, loc="left", pad=4)
    ax_f.set_title(f"Conditioned, {C.FILTER_BAND[0]}–{C.FILTER_BAND[1]:g} Hz",
                   fontsize=fs_annot, loc="left", pad=4)
    for ax in axes:
        ax.set_ylabel(vel, fontsize=fs_axis)
        ax.set_xlim(-C.PADDED_PRE_S, C.PADDED_POST_S)
        ax.axvline(0, color="black", linewidth=0.9, linestyle=":", zorder=5)
        ax.tick_params(labelsize=fs_tick)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.8)
        for x0, x1 in [(-C.PADDED_PRE_S, -C.PADDED_PRE_S + C.EDGE_PAD_S),
                       (C.WINDOW_POST_S, C.PADDED_POST_S)]:
            ax.axvspan(x0, x1, facecolor=COL_EDGE, alpha=0.10, zorder=0)
        ax.axvspan(-C.WINDOW_PRE_S, C.WINDOW_POST_S, facecolor=COL_SIGNAL,
                   alpha=0.14, zorder=0)
    ax_f.axvspan(-C.NOISE_PRE_S, -C.NOISE_MARGIN_S, facecolor=COL_NOISE,
                 alpha=0.30, zorder=0)
    ax_f.set_xlabel("Time relative to the onset (s)", fontsize=fs_axis)
    for xc, label, col in [
        (-C.PADDED_PRE_S + C.EDGE_PAD_S / 2, "edge\nmargin", COL_EDGE),
        (C.WINDOW_POST_S / 2, "signal window, 10 s", COL_SIGNAL),
        ((C.WINDOW_POST_S + C.PADDED_POST_S) / 2, "edge\nmargin", COL_EDGE),
    ]:
        ax_r.annotate(label, xy=(xc, 0.97), xycoords=("data", "axes fraction"),
                      ha="center", va="top", fontsize=fs_annot - 1, color=col)
    ax_f.annotate("noise\n2 s", xy=(-(C.NOISE_PRE_S + C.NOISE_MARGIN_S) / 2, 0.97),
                  xycoords=("data", "axes fraction"), ha="center", va="top",
                  fontsize=fs_annot - 1, color="#4a4a4a")
    fig.subplots_adjust(top=0.93, left=0.12, right=0.985, bottom=0.12)
    fig.align_ylabels(axes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_window.pdf", dpi=300)
    fig.savefig(OUT_DIR / "fig_window.png", dpi=200)
    print(f"-> {OUT_DIR/'fig_window.pdf'}")


if __name__ == "__main__":
    build()
