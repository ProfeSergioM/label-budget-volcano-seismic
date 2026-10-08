"""
Nucleo del pipeline: para cada volcan, descarga la traza de la estacion de
referencia por bloques continuos (no evento a evento -- con ~1.49M eventos
en total, una peticion Winston por evento haria la corrida impracticable) y
calcula, para cada evento, los 29 descriptores + metadatos del Entregable 1.

Checkpointing por mes: cada (año, mes) se procesa y se guarda por separado en
data/interim/features_checkpoints/<volcan>/<year>-<month>.parquet. Si el
proceso se interrumpe, re-ejecutar el mismo comando salta los meses ya
hechos. Con --finalize se concatenan todos los meses en el Parquet final del
Entregable 1.

Nunca se rellena con ceros ni se fabrica dato: un evento se excluye si su
ventana no cabe completa en un unico segmento continuo de la traza.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.earthworm import Client

import rqa_core as C

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "volcanoes.json"
INTERIM_DIR = ROOT / "data" / "interim"
CKPT_DIR = INTERIM_DIR / "features_checkpoints"
FEATURES_DIR = ROOT / "data" / "processed" / "features"
LOG_DIR = ROOT / "logs"

BLOCK_SECONDS = 2 * 3600  # 2 horas: agrupa eventos cercanos en una sola peticion Winston
BLOCK_MARGIN_S = 15.0     # > PADDED_PRE_S (13s) y > PADDED_POST_S (12s): cubre cualquier evento del bloque

# Decision del 2026-08-20 (confirmada con el usuario): la grilla RADII=[0.1,0.2,0.4]
# heredada del trabajo previo esta calibrada para su propia construccion del dominio
# de frecuencia (vector de 512 puntos, banda 0.5-10Hz). Con la ventana fija de 1000
# muestras de este estudio, el vector de frecuencia enmascarado tiene ~100 puntos y
# a esa escala radius=0.1/0.2 dan un RR degenerado (~0.0105, casi identico en todos
# los eventos, DET/L/DIV/TT indefinidos) que el rango de aceptancia ancho [0.01,0.10]
# deja pasar como si fuera valido. Cambiar la grilla es "reoptimizar" (prohibido por
# el usuario), asi que el dominio de frecuencia queda pendiente: se corre tiempo +
# espectrales + metadatos ahora, y las 13 columnas _f + eps_freq/rr_freq se dejan en
# NaN hasta que el usuario decida como proceder con la grilla de epsilon en frecuencia.
# Nota: como las trazas crudas no se archivan (ver plan), completar freq_* mas
# adelante requerira volver a descargar y reprocesar todos los eventos.
SKIP_FREQ_DOMAIN = True

FREQ_NAME_MAP = {
    "DET": "DET_f", "L": "Lavg_f", "Lmax": "Lmax_f", "DIV": "DIV_f",
    "Lentr": "Lentr_f", "LAM": "LAM_f", "TT": "TT_f", "Vmax": "Vmax_f",
    "Ventr": "Ventr_f", "W": "W_f", "Wmax": "Wmax_f", "Wdiv": "Wdiv_f",
    "Wentr": "Wentr_f",
}


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def make_clients(vcfg: dict):
    clients = []
    if vcfg.get("winston_historico"):
        clients.append(Client(vcfg["winston_historico"], vcfg["winston_port"], timeout=30))
    if vcfg.get("winston_server"):
        clients.append(Client(vcfg["winston_server"], vcfg["winston_port"], timeout=30))
    return clients


def fetch_block(clients, vcfg: dict, t1: UTCDateTime, t2: UTCDateTime):
    for client in clients:
        try:
            st = client.get_waveforms(
                vcfg["network"], vcfg["ref_station_channel"], vcfg["location"], vcfg["channel"], t1, t2
            )
            if len(st) > 0 and any(tr.stats.npts > 0 for tr in st):
                return st
        except Exception:
            continue
    return None


def find_covering_trace(stream, onset: UTCDateTime, sr_expected: float):
    """Devuelve el array numpy del tramo [onset-PADDED_PRE_S, onset+PADDED_POST_S)
    si algun Trace del stream lo cubre completo y sin gaps; si no, None.
    """
    t1 = onset - C.PADDED_PRE_S
    t2 = onset + C.PADDED_POST_S
    n_expected = int(round(C.PADDED_TOTAL_S * sr_expected))

    for tr in stream:
        if tr.stats.sampling_rate != sr_expected:
            continue
        if tr.stats.starttime <= t1 and tr.stats.endtime >= t2:
            seg = tr.slice(starttime=t1, endtime=t2)
            data = seg.data
            if len(data) >= n_expected:
                return np.asarray(data[:n_expected], dtype=np.float64)
    return None


def extract_event_row(event: pd.Series, raw_segment: np.ndarray, sr: float, volcano_label: str, vcfg: dict) -> dict:
    filtered = C.preprocess_segment(raw_segment, sr)
    idx = C.padded_window_indices(sr)

    window = filtered[idx["signal_start"]:idx["signal_end"]]

    # El SNR se mide sobre el mismo segmento filtrado en SNR_BAND, no en
    # FILTER_BAND: en 0.1-10 Hz el denominador queda dominado por microsismo
    # oceanico y el cociente deja de describir la calidad de registro en la
    # banda donde emiten los eventos (ver nota de SNR_BAND en rqa_core).
    filtered_snr = C.preprocess_segment(raw_segment, sr, band=C.SNR_BAND)
    noise = filtered_snr[idx["noise_start"]:idx["noise_end"]]
    signal_snr = filtered_snr[idx["signal_start"]:idx["signal_end"]]

    if len(window) != C.WINDOW_SAMPLES:
        raise C.ExclusionError(f"ventana incompleta ({len(window)}/{C.WINDOW_SAMPLES} muestras)")

    xz = C.zscore(window)
    rqa_t, eps_t, rr_t = C.compute_rqa_domain(xz, C.M_TIME, C.TAU_TIME)

    spectral, amp_log = C.compute_spectral_features(window, sr)
    if SKIP_FREQ_DOMAIN:
        # fpeak/SR/FI no dependen de la grilla de epsilon y si se calculan;
        # solo se pausan los 13 descriptores RQA de frecuencia (ver SKIP_FREQ_DOMAIN).
        rqa_f = {k: float("nan") for k in C.RQA_KEYS}
        eps_f, rr_f = float("nan"), float("nan")
    elif amp_log is not None:
        try:
            amp_z = C.zscore(amp_log)
            rqa_f, eps_f, rr_f = C.compute_rqa_domain(amp_z, C.M_FREQ, C.TAU_FREQ)
        except C.ExclusionError:
            rqa_f = {k: float("nan") for k in C.RQA_KEYS}
            eps_f, rr_f = float("nan"), float("nan")
    else:
        rqa_f = {k: float("nan") for k in C.RQA_KEYS}
        eps_f, rr_f = float("nan"), float("nan")

    snr_db = (C.compute_snr_db(noise, signal_snr)
              if len(noise) == (idx["noise_end"] - idx["noise_start"]) else float("nan"))

    lat = event.get("latitud")
    lon = event.get("longitud")
    depth = event.get("profundidad")
    located = pd.notna(lat)

    # 'idevento' del catalogo OVDAS no es unico por si solo (se reutiliza entre
    # eventos distintos, confirmado 2026-08-24). (idevento, fecha) si es estable
    # y unico (tras deduplicar filas completas identicas en filter_reference_events.py),
    # asi que event_id combina ambos, en epoca-milisegundos para que sea determinista.
    epoch_ms = int(round(event["fecha"].timestamp() * 1000))
    event_id = f"{int(event['idevento'])}-{epoch_ms}"

    row = {
        "event_id": event_id,
        "volcano": volcano_label,
        "label": event["tipoevento"],
        "origin_time": event["fecha"].isoformat(timespec="milliseconds"),
        "station": vcfg["ref_station"],
        "duration_s_original": float(event["duracion"]),
        "snr_db": snr_db,
        "eps_time": eps_t,
        "rr_time": rr_t,
        "eps_freq": eps_f,
        "rr_freq": rr_f,
        "lat": float(lat) if pd.notna(lat) else float("nan"),
        "lon": float(lon) if pd.notna(lon) else float("nan"),
        "depth_km": float(depth) if pd.notna(depth) else float("nan"),
        "located": bool(located),
        "loc_quality": event.get("calidad") if pd.notna(event.get("calidad")) else None,
    }
    row.update(rqa_t)
    for k, v in rqa_f.items():
        row[FREQ_NAME_MAP[k]] = v
    row.update(spectral)
    return row


def process_month(events: pd.DataFrame, vcfg: dict, volcano_label: str, clients) -> tuple[pd.DataFrame, list[dict]]:
    sr = vcfg["sps"]
    rows = []
    exclusions = []

    current_block_idx = None
    current_stream = None

    for _, event in events.iterrows():
        onset = UTCDateTime(event["fecha"].to_pydatetime())
        block_idx = int(onset.timestamp // BLOCK_SECONDS)

        if block_idx != current_block_idx:
            block_start = UTCDateTime(block_idx * BLOCK_SECONDS)
            block_end = block_start + BLOCK_SECONDS
            current_stream = fetch_block(
                clients, vcfg, block_start - BLOCK_MARGIN_S, block_end + BLOCK_MARGIN_S
            )
            current_block_idx = block_idx

        if current_stream is None:
            exclusions.append({"event_id": int(event["idevento"]), "reason": "sin datos de estacion en el bloque"})
            continue

        raw_segment = find_covering_trace(current_stream, onset, sr)
        if raw_segment is None:
            exclusions.append({"event_id": int(event["idevento"]), "reason": "gap o cobertura incompleta en la traza"})
            continue

        try:
            row = extract_event_row(event, raw_segment, sr, volcano_label, vcfg)
            rows.append(row)
        except C.ExclusionError as e:
            exclusions.append({"event_id": int(event["idevento"]), "reason": f"excluido: {e}"})
        except Exception as e:
            exclusions.append({"event_id": int(event["idevento"]), "reason": f"error: {e}"})
            traceback.print_exc()

    return pd.DataFrame(rows), exclusions


def _process_one_month(key: str, ym: str, month_events: pd.DataFrame, vcfg: dict, volcano_label: str,
                        ckpt_path: Path, excl_log_path: Path) -> tuple[str, int, int, int, float]:
    clients = make_clients(vcfg)
    if not clients:
        raise RuntimeError(f"[{key}] sin servidores Winston configurados")

    t0 = time.time()
    feats, exclusions = process_month(month_events, vcfg, volcano_label, clients)
    dt = time.time() - t0

    feats.to_parquet(ckpt_path, index=False)

    if exclusions:
        excl_df = pd.DataFrame(exclusions)
        excl_df["volcano"] = key
        excl_df["year_month"] = ym
        excl_df.to_csv(excl_log_path, mode="a", header=not excl_log_path.exists(), index=False)

    return ym, len(month_events), len(feats), len(exclusions), dt


def process_volcano(key: str, cfg: dict, force: bool = False, limit_months: int | None = None, workers: int = 4,
                     suffix: str = ""):
    vcfg = cfg[key]
    volcano_label = vcfg["nombre"]

    catalog_path = INTERIM_DIR / f"catalog_{key}{suffix}.parquet"
    df = pd.read_parquet(catalog_path)
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True)
    df["_ym"] = df["fecha"].dt.strftime("%Y-%m")

    ckpt_dir = CKPT_DIR / f"{key}{suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    excl_log_path = LOG_DIR / f"exclusions_{key}{suffix}.csv"

    months = sorted(df["_ym"].unique())
    if limit_months:
        months = months[:limit_months]

    pending = []
    for ym in months:
        ckpt_path = ckpt_dir / f"{ym}.parquet"
        if ckpt_path.exists() and not force:
            print(f"[{key}] {ym} ya procesado, se salta")
            continue
        month_events = df[df["_ym"] == ym].sort_values("fecha")
        pending.append((key, ym, month_events, vcfg, volcano_label, ckpt_path, excl_log_path))

    if not pending:
        print(f"[{key}] nada pendiente ({len(months)} meses ya procesados)")
        return

    print(f"[{key}] {len(pending)} meses pendientes de {len(months)}, workers={workers}", flush=True)

    done = 0
    if workers <= 1:
        for args in pending:
            result = _process_one_month(*args)
            done += 1
            _log_month_result(key, result, done, len(pending))
    else:
        import multiprocessing as mp
        with mp.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(_process_one_month_star, pending):
                done += 1
                _log_month_result(key, result, done, len(pending))


def _process_one_month_star(args):
    return _process_one_month(*args)


def _log_month_result(key: str, result: tuple, done: int, total: int):
    ym, n_total, n_ok, n_excl, dt = result
    rate = n_total / dt if dt > 0 else float("nan")
    print(f"[{key}] {ym} ({done}/{total}): {n_total} eventos | ok={n_ok} excluidos={n_excl} | "
          f"{dt:.1f}s ({rate:.2f} ev/s)", flush=True)


def finalize_volcano(key: str, cfg: dict):
    # Fusiona TODOS los directorios de checkpoints de este volcan: el base
    # (LP/VT/HB) y cualquier extension por sufijo (ej. "<key>_extra" con
    # LV/AV/TO, agregado el 2026-08-24) -- ver filter_reference_events.py.
    ckpt_dirs = sorted(p for p in CKPT_DIR.glob(f"{key}*") if p.is_dir())
    parts = [p for d in ckpt_dirs for p in sorted(d.glob("*.parquet"))]
    if not parts:
        print(f"[{key}] no hay checkpoints para consolidar")
        return
    print(f"[{key}] consolidando {len(parts)} checkpoints de {[d.name for d in ckpt_dirs]}")
    dfs = [pd.read_parquet(p) for p in parts]
    full = pd.concat(dfs, ignore_index=True)

    # Deduplicacion por event_id. El catalogo de origen trae filas repetidas
    # (misma idevento y mismo unixtime): 928 claves en nevchillan, 48 en
    # villarrica, 24 en copahue; lagmaule y hudson no tienen. La extraccion
    # procesa cada fila fielmente, asi que el evento sale dos veces con
    # descriptores identicos -- verificado en los 981 casos. Sin esto, la
    # poblacion analizada queda inflada y, peor, un mismo evento puede caer a
    # ambos lados de una particion train/test, que es exactamente la fuga que
    # este estudio mide.
    n_before = len(full)
    full = full.drop_duplicates(subset="event_id", keep="first")
    if len(full) < n_before:
        print(f"[{key}] {n_before - len(full)} filas duplicadas del catalogo descartadas")

    full = full.sort_values("origin_time").reset_index(drop=True)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / f"features_{key}.parquet"
    full.to_parquet(out_path, index=False)
    print(f"[{key}] consolidado: {len(full)} eventos -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volcano", choices=["villarrica", "nevchillan", "copahue", "lagmaule", "hudson"], required=True)
    ap.add_argument("--force", action="store_true", help="reprocesa meses ya hechos")
    ap.add_argument("--limit-months", type=int, default=None, help="solo procesa los primeros N meses (pruebas)")
    ap.add_argument("--finalize", action="store_true", help="solo consolida checkpoints existentes, no descarga nada")
    ap.add_argument("--workers", type=int, default=4, help="procesos en paralelo (1 = secuencial)")
    ap.add_argument("--extra", action="store_true", help="procesa catalog_<key>_extra.parquet (LV/AV/TO) en vez del base (LP/VT/HB)")
    args = ap.parse_args()

    cfg = load_config()
    suffix = "_extra" if args.extra else ""
    if args.finalize:
        finalize_volcano(args.volcano, cfg)
    else:
        process_volcano(args.volcano, cfg, force=args.force, limit_months=args.limit_months, workers=args.workers,
                         suffix=suffix)


if __name__ == "__main__":
    main()
