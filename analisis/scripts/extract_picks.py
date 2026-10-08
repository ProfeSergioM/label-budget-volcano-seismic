"""
Descriptores de los corpus externos desde el corte manual.

Toma la ultima marca de cada traza en experiments/picks/picks_manuales.csv
(estado bueno o dudoso), lee la traza igual que la pagina de corte
(cortar_trazas.cargar), la lleva a 100 Hz y pasa onset y fin marcados a
features_from_signal del paquete. Todo lo demas es la tuberia de siempre:
mismas bandas, mismo m y tau, misma ventana de ruido del paquete.

La ventana de ruido marcada a mano no reemplaza la del paquete; entra como
columna aparte, `snr_manual_db`: RMS del evento sobre RMS del ruido marcado,
en 0,5-10 Hz. `clase` es la revisada al cortar si la hay (la del corpus
queda en `clase_corpus`), salvo los tornillos: se analizan como LP y quedan
marcados en `subtipo` = TO. `fuera_de_ventana` marca los eventos cuyo onset cae fuera de la ventana
etiquetada y que el analista acepto igual. `duplicado` marca las copias de un mismo evento (ver marcar_duplicados);
los analisis usan solo duplicado = False. `ruido_pisa_evento` marca los eventos de Decepcion cuyo ruido
se cruza con otro evento (ver experiments/picks/README.md).

Es incremental: una traza se recalcula solo si su marca cambio desde la
corrida anterior, de modo que se puede correr mientras el corte avanza.

Con --onset afinado, el onset es el del AIC (scripts/afinar_onset.py) en VT
y en Decepcion, donde la calibracion lo respalda, y la marca manual en el
resto (LP emergentes). La salida va a otro archivo para comparar ambas; la
columna `fuente_onset` dice cual se uso.

Uso:
    python scripts/extract_picks.py                    # lo nuevo o cambiado
    python scripts/extract_picks.py --todo             # recalcula todo
    python scripts/afinar_onset.py && python scripts/extract_picks.py --onset afinado
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dist" / "descriptores_volcanicos"))
import cortar_trazas as CT  # noqa: E402

OUT = ROOT / "data" / "processed" / "features_manual"
TODO = OUT / "features_manual.parquet"
AFINADO = OUT / "features_manual_afinado.parquet"
ONSET_AIC = ROOT / "data" / "processed" / "onset_afinado.parquet"
SPS = 100.0
SUBTIPOS = {"TO": "LP"}        # subtipo reclasificado -> clase con que se analiza
BANDA_SNR = (0.5, 10.0)


def a_100hz(x: np.ndarray, sr: float) -> np.ndarray:
    from scipy.signal import resample_poly
    nominal = round(sr)          # 99,9993 o 100,009 Hz se toman como 100
    if nominal == SPS:
        return x
    f = Fraction(int(SPS), nominal)
    return resample_poly(x - x.mean(), f.numerator, f.denominator)


def snr_manual(x: np.ndarray, p: dict) -> float:
    from scipy.signal import butter, sosfiltfilt
    y = sosfiltfilt(butter(4, BANDA_SNR, btype="bandpass", fs=SPS, output="sos"),
                    x - x.mean())
    rms = lambda a, b: np.sqrt(np.mean(y[int(a * SPS):int(b * SPS)] ** 2))  # noqa: E731
    try:
        return float(20 * np.log10(rms(p["onset"], p["fin"])
                                   / rms(p["ruido_ini"], p["ruido_fin"])))
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return float("nan")


def una(args) -> dict:
    fila, item = args
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    from descriptores import WindowError, features_from_signal
    base = dict(item_id=fila["item_id"], guardado_utc=fila["guardado_utc"])
    try:
        t = CT.cargar(item)
    except Exception as e:                              # noqa: BLE001
        return dict(base, motivo=f"lectura: {type(e).__name__}")
    x = a_100hz(np.nan_to_num(t["x"]), t["sr"])
    m = {k: float(fila[k]) for k in ("ruido_ini", "ruido_fin", "onset", "fin")}
    base.update(sps_original=t["sr"], duracion_s=m["fin"] - m["onset"],
                snr_manual_db=snr_manual(x, m))
    try:
        f = features_from_signal(x, onset_s=m["onset"], coda_end_s=m["fin"], sps=SPS)
    except WindowError as e:
        return dict(base, motivo=f"ventana: {e}")
    return dict(base, motivo="", **f.as_dict())


def marcar_duplicados(d: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    """Un mismo evento puede entrar mas de una vez: en VCSEIS un sismo viene
    en varias estaciones, y en VOISS-Net y el SGC ventanas o archivos de
    minutos contiguos contienen el mismo evento. `evento_id` agrupa las
    copias y `duplicado` marca todas menos una: la de onset dentro de su
    ventana y, entre esas, la de mayor SNR manual. Decepcion no trae hora
    absoluta; ahi las copias dentro de un archivo ya se revisaron al cortar."""
    d = d.copy()
    t0 = picks.drop_duplicates("item_id", keep="last").set_index("item_id").t0_utc
    d["t_onset_utc"] = (pd.to_datetime(d.item_id.map(t0), utc=True, errors="coerce",
                                       format="mixed")
                        + pd.to_timedelta(pd.to_numeric(d.onset, errors="coerce"), unit="s"))
    d["evento_id"] = d.item_id
    # VCSEIS: el identificador del sismo en el corpus
    v = d.dataset.eq("vcseis")
    if v.any():
        chunk = d.loc[v, "item_id"].str.split(":").str[1]
        traza = d.loc[v, "item_id"].str.split(":", n=2).str[2]
        src = {}
        for ch in chunk.unique():
            m = pd.read_csv(ROOT / "data" / "external" / "vcseis" / f"metadata_{ch}.csv",
                            usecols=["trace_name", "source_id"], low_memory=False)
            src.update(dict(zip(ch + ":" + m.trace_name, m.source_id.astype(str))))
        d.loc[v, "evento_id"] = "vcseis:" + (chunk + ":" + traza).map(src).fillna(d.loc[v, "item_id"])
    # VOISS-Net y SGC: onsets a menos de 3 s en el mismo sistema
    for (ds, sis), g in d[d.dataset.isin(["voissnet", "sgc"]) & d.t_onset_utc.notna()
                          ].groupby(["dataset", "sistema"]):
        g = g.sort_values("t_onset_utc")
        nuevo = g.t_onset_utc.diff().dt.total_seconds().fillna(1e9) >= 3.0
        grupo = nuevo.cumsum()
        d.loc[g.index, "evento_id"] = (ds + ":" + sis + ":" + grupo.astype(str)).values
    orden = d.assign(_f=d.get("fuera_de_ventana", False).astype(bool),
                     _s=-d.snr_manual_db.fillna(-1e9))
    orden = orden.sort_values(["evento_id", "_f", "_s"])
    d["duplicado"] = orden.duplicated("evento_id").reindex(d.index)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todo", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--onset", choices=["manual", "afinado"], default="manual")
    a = ap.parse_args()
    destino = TODO if a.onset == "manual" else AFINADO

    picks = pd.read_csv(CT.PICKS, dtype=str, keep_default_na=False)
    picks = picks.drop_duplicates("item_id", keep="last")
    picks = picks[picks.estado.isin(["bueno", "dudoso"])].copy()
    picks["fuente_onset"] = "manual"
    if "clase_revisada" not in picks:
        picks["clase_revisada"] = ""
    if a.onset == "afinado":
        af = pd.read_parquet(ONSET_AIC).set_index("item_id")
        vig = picks.item_id.map(af.guardado_utc) == picks.guardado_utc
        usa = vig & (picks.clase.eq("VT") | picks.dataset.eq("decepcion"))
        on = picks.item_id.map(af.onset_aic)
        usa &= on.notna()
        picks.loc[usa, "onset"] = on[usa].map(lambda v: f"{v:.3f}")
        picks.loc[usa, "fuente_onset"] = "aic"
        # la cache distingue por onset usado, no solo por la hora de guardado
        picks["guardado_utc"] = picks.guardado_utc + "|" + picks.onset
        print(f"onset AIC en {int(usa.sum())} de {len(picks)}; "
              f"{int((~vig).sum())} marcas sin afinar todavia (correr afinar_onset.py)")
    cola = pd.read_csv(CT.COLA, dtype=str, keep_default_na=False)
    items = cola.set_index("item_id")

    previo = pd.DataFrame()
    if destino.exists() and not a.todo:
        previo = pd.read_parquet(destino)
        # se conserva solo lo que sigue marcado igual
        vigente = picks.set_index("item_id").guardado_utc
        previo = previo[previo.item_id.map(vigente) == previo.guardado_utc]
    pend = picks[~picks.item_id.isin(previo.item_id if len(previo) else [])]
    print(f"{len(picks)} marcas buenas o dudosas; {len(previo)} ya calculadas, "
          f"{len(pend)} por calcular", flush=True)

    # la estacion es la de la marca (en el Ruiz puede ser Bis en vez de Olleta)
    trabajos = [(f, items.loc[f["item_id"]].to_dict() | {"item_id": f["item_id"]}
                 | ({"estacion": f["estacion"]} if f.get("estacion") else {}))
                for f in pend.to_dict("records")]
    nuevas = []
    if trabajos:
        with ProcessPoolExecutor(a.workers) as ex:
            for k, r in enumerate(ex.map(una, trabajos, chunksize=8), 1):
                nuevas.append(r)
                if k % 250 == 0:
                    print(f"  {k}/{len(trabajos)}", flush=True)

    d = pd.concat([previo, pd.DataFrame(nuevas)], ignore_index=True)
    meta = picks[["item_id", "dataset", "sistema", "clase", "estacion", "estado",
                  "origen", "ruido_ini", "ruido_fin", "onset", "fin", "fuente_onset",
                  "clase_revisada"]]
    d = meta.merge(d.drop(columns=[c for c in meta.columns if c != "item_id" and c in d]),
                   on="item_id", how="left")
    # clase del corpus, salvo que el analista la haya reclasificado al cortar.
    # Los tornillos (TO) son muy pocos para ser una clase aparte: se analizan
    # como LP y quedan identificados en `subtipo`.
    d["clase_corpus"] = d.clase
    rev = d.clase_revisada.fillna("")
    d["subtipo"] = rev.where(rev.isin(SUBTIPOS), "")
    d["clase"] = rev.map(lambda c: SUBTIPOS.get(c, c)).where(rev != "", d.clase)
    # El Ruiz se corto en Olleta y, para lo debil en Olleta, en Bis (regla del
    # SGC). La estacion pesa en los descriptores y no se reparte igual entre
    # clases, asi que cada estacion es un sistema aparte, como un volcan.
    ruiz = d.dataset.eq("sgc")
    d.loc[ruiz, "sistema"] = np.where(d.loc[ruiz, "estacion"].str.startswith("BIS"),
                                      "Ruiz-Bis", "Ruiz-Olleta")
    # onset fuera de la ventana que etiqueto el corpus, aceptado a mano: se
    # conserva pero queda marcado para analisis de sensibilidad
    ultimas = (pd.read_csv(CT.PICKS, dtype=str, keep_default_na=False)
               .drop_duplicates("item_id", keep="last"))
    d["fuera_de_ventana"] = d.item_id.isin(CT.fuera_de_ventana(ultimas, cola))
    d = marcar_duplicados(d, ultimas)
    d["ruido_pisa_evento"] = d.item_id.isin(CT.ruido_con_evento(
        pd.read_csv(CT.PICKS, dtype=str, keep_default_na=False)
        .drop_duplicates("item_id", keep="last"), cola))
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(destino, index=False)

    ok = d.motivo == ""
    print(f"duplicados marcados: {int((ok & d.duplicado).sum())} de {int(ok.sum())} con descriptores")
    t = d.assign(ok=ok).groupby(["dataset", "sistema", "clase"]).ok.agg(["sum", "size"])
    t.columns = ["con descriptores", "marcas"]
    print(t.to_string())
    malos = d.loc[~ok, "motivo"].str.replace(r"[\d.]+", "N", regex=True).value_counts()
    if len(malos):
        print("\nsin descriptores:\n" + malos.to_string())
    print(f"\n-> {destino}")


if __name__ == "__main__":
    main()
