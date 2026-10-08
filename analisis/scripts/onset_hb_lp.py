"""
Criterio HB/LP en Hudson: el experto rotula HB un LP cuyo onset trae un
paquete de alta frecuencia, rasgo que solo se ve en eventos energeticos
(observacion de S. Morales, 2026-10-08; en Hudson la fraccion de HB entre
HB+LP sube de 1 % a 14 % del quintil mas bajo al mas alto de SNR).

Para todos los HB de Hudson y una muestra de LP repartida por SNR, se baja
la traza de HUDZ (Winston), se filtra entre 0.5 y 10 Hz y se mide el
contenido de alta frecuencia del onset frente al del resto del evento:
  FI_onset   log10 A(4.5-8 Hz)/A(1-2.5 Hz) en los primeros ONSET_S segundos
  FI_resto   lo mismo entre ONSET_S y 10 s
  dFI        FI_onset - FI_resto (paquete de alta frecuencia al inicio)
  HF_onset   fraccion de energia sobre 4.5 Hz en los primeros ONSET_S s
Luego compara HB y LP a igual SNR (por cuartil de SNR) y cruza con la
revision manual de los sospechosos (experiments/picks/revision).

Uso:
    python scripts/onset_hb_lp.py --sistema hudson|copahue|ruiz [--n-lp 2000] [--n-hb 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "experiments" / "reports"
ONSET_S, PRE_S, POST_S, SR = 2.0, 10.0, 12.0, 100.0
SEMILLA = 41


def amplitud(x, sr, f0, f1):
    a = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return a[(f >= f0) & (f < f1)].mean()


def medir(x):
    from scipy.signal import butter, detrend, sosfiltfilt
    y = sosfiltfilt(butter(4, (0.5, 10.0), "bandpass", fs=SR, output="sos"), detrend(x))
    i0 = int(PRE_S * SR)
    on = y[i0:i0 + int(ONSET_S * SR)]
    resto = y[i0 + int(ONSET_S * SR):i0 + int(10 * SR)]
    fi = lambda s: np.log10(amplitud(s, SR, 4.5, 8.0) / amplitud(s, SR, 1.0, 2.5))  # noqa: E731
    p = np.abs(np.fft.rfft(on * np.hanning(len(on)))) ** 2
    f = np.fft.rfftfreq(len(on), 1 / SR)
    return dict(FI_onset=fi(on), FI_resto=fi(resto), dFI=fi(on) - fi(resto),
                HF_onset=p[f >= 4.5].sum() / p[(f >= 1.0)].sum())


def eventos_ovdas(clave, n_lp, n_hb):
    """HB y LP de un sistema OVDAS, cada clase repartida por deciles de SNR."""
    from obspy import UTCDateTime
    from make_fig_coda_window import segment
    cfg = json.load(open(ROOT / "config" / "volcanoes.json", encoding="utf-8"))[clave]
    d = pd.read_parquet(ROOT / f"data/processed/features/features_{clave}.parquet",
                        columns=["event_id", "label", "origin_time", "snr_db", "duration_s_original"])
    d = d.dropna(subset=["snr_db"])
    partes = []
    for c, n in (("HB", n_hb), ("LP", n_lp)):
        g = d[d.label == c].copy()
        if len(g) > n:
            g["q"] = pd.qcut(g.snr_db, 10, labels=False, duplicates="drop")
            g = g.groupby("q", group_keys=False).apply(
                lambda h: h.sample(min(len(h), n // 10), random_state=SEMILLA))
        partes.append(g.drop(columns="q", errors="ignore"))
    ev = pd.concat(partes)
    for r in ev.itertuples():
        lector = (lambda r=r: segment(cfg, UTCDateTime(pd.Timestamp(r.origin_time).isoformat()),
                                      PRE_S, POST_S, SR))
        yield dict(event_id=str(r.event_id), label=r.label, sistema=cfg["nombre"],
                   snr_db=r.snr_db, dur=r.duration_s_original), lector


def eventos_ruiz():
    """HB y LP del SGC en Olleta y Bis, con el onset del corte manual."""
    import cortar_trazas as CT
    from extract_picks import a_100hz
    f = pd.read_parquet(ROOT / "data/processed/features_manual/features16_fija.parquet")
    f = f[(f.motivo == "") & ~f.duplicado.fillna(False).astype(bool)
          & f.sistema.isin(["Ruiz-Olleta", "Ruiz-Bis"]) & f.clase.isin(["HB", "LP"])]
    cola = pd.read_csv(CT.COLA, dtype=str, keep_default_na=False).set_index("item_id")
    picks = (pd.read_csv(CT.PICKS, dtype=str, keep_default_na=False)
             .drop_duplicates("item_id", keep="last").set_index("item_id"))
    for r in f.itertuples():
        def lector(r=r):
            item = cola.loc[r.item_id].to_dict() | {"item_id": r.item_id,
                                                     "estacion": picks.loc[r.item_id, "estacion"]}
            t = CT.cargar(item)
            x = a_100hz(np.nan_to_num(t["x"]), t["sr"])
            i0 = int(round((float(picks.loc[r.item_id, "onset"]) - PRE_S) * SR))
            n = int((PRE_S + POST_S) * SR)
            if i0 < 0 or i0 + n > len(x):
                raise ValueError("traza corta")
            return x[i0:i0 + n]
        yield dict(event_id=r.item_id, label=r.clase, sistema=r.sistema, snr_db=r.snr_db,
                   dur=float(r.fin) - float(r.onset)), lector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sistema", default="hudson", choices=["hudson", "copahue", "ruiz"])
    ap.add_argument("--n-lp", type=int, default=2000)
    ap.add_argument("--n-hb", type=int, default=2000)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    gen = eventos_ruiz() if a.sistema == "ruiz" else eventos_ovdas(a.sistema, a.n_lp, a.n_hb)
    destino = OUT / f"onset_hb_lp_{a.sistema}.csv"
    hechos = pd.read_csv(destino, dtype={"event_id": str}) if destino.exists() \
        else pd.DataFrame(columns=["event_id"])
    ya = set(hechos.event_id.astype(str))
    filas, k = [], 0
    for meta, lector in gen:
        if meta["event_id"] in ya:
            continue
        k += 1
        try:
            filas.append(dict(meta, **medir(lector())))
        except Exception as e:                          # noqa: BLE001
            filas.append(dict(meta, error=type(e).__name__))
        if k % 200 == 0:
            pd.concat([hechos, pd.DataFrame(filas)]).to_csv(destino, index=False)
            print(f"  {k}", flush=True)
    pd.concat([hechos, pd.DataFrame(filas)]).to_csv(destino, index=False)
    t = pd.read_csv(destino)
    if "error" in t:
        t = t[t.error.isna()]
    if "sistema" not in t:
        t["sistema"] = "Hudson"
    t["sistema"] = t.sistema.fillna("Hudson")
    filas = []
    for sis, g in t.groupby("sistema"):
        g = g.assign(q=pd.qcut(g.snr_db, 4, labels=False, duplicates="drop"))
        r = g.groupby(["q", "label"]).agg(snr=("snr_db", "median"), n=("label", "size"),
                                          dFI=("dFI", "median"), FI_onset=("FI_onset", "median"),
                                          HF_onset=("HF_onset", "median")).round(3)
        filas.append(r.assign(sistema=sis))
        print(f"\n{sis}\n", r.to_string())
    pd.concat(filas).to_csv(OUT / f"onset_hb_lp_{a.sistema}_resumen.csv")
    if a.sistema != "hudson":
        return
    # cruce con la revision manual de los sospechosos de Hudson
    rv = ROOT / "experiments/picks/revision/revision_hudson_sospechosos.csv"
    if rv.exists():
        p = pd.read_csv(rv, dtype=str, keep_default_na=False).drop_duplicates("item_id", keep="last")
        p["event_id"] = p.item_id.str.split(":").str[-1]
        p = p[p.clase == "HB"]
        p["final"] = p.clase_revisada.where(p.clase_revisada != "", "HB")
        m = t.assign(event_id=t.event_id.astype(str)).merge(p[["event_id", "final"]], on="event_id")
        c = m.groupby("final").agg(n=("final", "size"), snr=("snr_db", "median"),
                                   dFI=("dFI", "median"), HF_onset=("HF_onset", "median")).round(3)
        print("\nHB revisados a mano, segun la clase final:\n", c.to_string())
        c.to_csv(OUT / "onset_hb_lp_hudson_revision.csv")


if __name__ == "__main__":
    main()
