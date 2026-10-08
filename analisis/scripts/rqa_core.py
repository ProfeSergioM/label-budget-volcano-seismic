"""
Nucleo compartido de preprocesado, RQA y features espectrales.

Metodologia reutilizada tal cual del trabajo previo
(/home/smorales/Documentos/Paper_Clustering/scripts/mseed_rqa_features.py,
compute_snr_spectrum.py): algoritmo de seleccion de epsilon (grilla fija de
radios, sin busqueda continua) y formula de SNR. La ventana fija de 1000
muestras (en vez de duracion variable + remuestreo a 512 puntos) y la banda
de filtro 0.1-10 Hz son requisitos nuevos de este estudio, no del trabajo
previo. Ver docs/README.md para el detalle completo de decisiones.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation

# --- Ventana y filtro (fijados a priori, no derivados del catalogo) ---
SPS = 100.0

# Ventana de senal: [onset, onset+10s]. Antes [onset-1s, onset+9s]; el segundo
# previo al onset se elimino el 2026-08-27. Medido sobre 20 eventos de 4
# combinaciones volcan-clase: ese segundo ocupa el 1.0% del area de la matriz
# de recurrencia pero aporta el 7.3% de TODAS las recurrencias (mediana, hasta
# 22.4%), y el 22.2% de las recurrencias involucra al menos una muestra
# pre-onset (hasta 43.5%). Es decir, el 10% de las muestras pesaba el doble de
# lo que le corresponde, y por construccion no contiene evento: al ser casi
# plano, tras el z-score todos sus puntos caen dentro de epsilon entre si y
# saturan un bloque de la matriz. Efecto en los descriptores al quitarlo
# (mediana): RR -4.0%, TT -5.0%, LAM -0.9%, DET -0.3%; el epsilon elegido no
# cambio en ninguno de los 20 eventos.
# Se conservan las 1000 muestras extendiendo la ventana 1 s hacia la coda, en
# vez de acortarla a 900: Lmax, W, Wmax y las entropias dependen del largo de
# la serie, y mantener N fijo las deja comparables entre eventos y con el
# barrido de duracion, que ya mostro que ventanas mas largas resuelven mejor
# la estructura de familias.
WINDOW_PRE_S = 0.0
WINDOW_POST_S = 10.0
WINDOW_SAMPLES = int(round((WINDOW_PRE_S + WINDOW_POST_S) * SPS))  # 1000

FILTER_ORDER = 4

# Banda de analisis. Subida de 0.1 a 0.5 Hz el 2026-08-27, por la misma razon
# que ya habia movido la banda del SNR: bajo 0.5 Hz el registro esta dominado
# por el microsismo oceanico secundario (pico en 0.125 Hz), y con el corte en
# 0.1 Hz ese ruido entraba a la ventana sobre la que se calculan LOS DIECISEIS
# descriptores, no solo el SNR. Medido sobre 48 eventos, la fraccion de la
# energia de la ventana que cae bajo 0.5 Hz es: Hudson VT 98.2%, Chillan VLP
# 85.6%, Copahue VLP 68.1%, Copahue LP 47.9%, Copahue HB 48.5%, Chillan TO
# 42.9%. En Hudson eso significa que la RQA de un VT estaba describiendo el
# oleaje y no el sismo (solo el 6.4% del RMS sobrevive al filtrar 0.5-10).
# COSTO A DECLARAR: VLP pierde la mayor parte de su energia (conserva 26-40%
# del RMS). Parte de lo que pierde es ruido comun a todas las clases de esa
# estacion -- Copahue VLP tiene 68% bajo 0.5 Hz contra 48% de LP y HB en la
# misma estacion, asi que el exceso de ~20 puntos es señal VLP genuina, pero
# el resto no.
FILTER_BAND = (0.5, 10.0)

# Ventana de ruido: [onset-4s, onset-2s], 2 s. Acortada de 8 s a 2 s el
# 2026-08-27; ver SNR_BAND abajo para el otro cambio de la misma revision.
NOISE_PRE_S = 4.0      # inicio de la ventana de ruido, relativo al onset
NOISE_MARGIN_S = 2.0   # fin de la ventana de ruido, relativo al onset (deja 2s de margen)

# Banda en la que se mide el SNR, distinta de FILTER_BAND (2026-08-27).
# Medido sobre 30 eventos, el 91.8% de la POTENCIA de la ventana de ruido en
# 0.1-10 Hz cae bajo 0.5 Hz: es microsismo oceanico secundario (pico en 0.125
# Hz, periodo 8 s), que el pasabanda de analisis deja pasar por diseno porque
# los VLP viven ahi. Con el denominador dominado por microsismo, el SNR medido
# comparaba nivel de oleaje y no calidad de registro: los LP de Hudson daban
# SNR mediano NEGATIVO (-1.6 dB) pese a ser eventos catalogados, y la
# diferencia Hudson-Villarrica reflejaba sobre todo la distancia al Pacifico
# (92-98% de ruido bajo 0.5 Hz en Hudson contra 23-39% en Villarrica). Medido
# en 0.5-10 Hz las medianas quedan en 8.9-12.2 dB. Salvedad a declarar: los
# VLP si emiten bajo 0.5 Hz y quedan subestimados por esta banda.
# Con FILTER_BAND en 0.5-10 Hz, la banda del SNR coincide con la de analisis
# y el SNR se mide sobre el mismo segmento acondicionado. Se mantiene como
# constante propia para que quede explicito sobre que banda se define.
SNR_BAND = (0.5, 10.0)

EDGE_PAD_S = 3.0  # margen adicional a cada lado para evitar artefactos de borde del filtfilt

# Segmento total a descargar/filtrar por evento: [onset - PADDED_PRE_S, onset + PADDED_POST_S)
# Acortado de 13 s a 7 s el 2026-08-27, al pasar la ventana de ruido a 2 s.
# Medido: contra una referencia de 38 s de lead, la ventana de senal filtrada
# difiere 2.4e-3 (relativo al pico) con lead de 7 s y 1.3e-3 con lead de 13 s
# -- ambos en la misma meseta, la convergencia real recien llega a ~26-32 s.
# Los 13 s no compraban precision sobre 7 s; se elige 7 s = NOISE_PRE_S + EDGE_PAD_S,
# que es coherente con la ventana de ruido y descarga 6 s menos por evento.
PADDED_PRE_S = NOISE_PRE_S + EDGE_PAD_S    # 7 s
PADDED_POST_S = WINDOW_POST_S + EDGE_PAD_S  # 12 s
PADDED_TOTAL_S = PADDED_PRE_S + PADDED_POST_S  # 25 s


def padded_window_indices(sr: float) -> dict:
    """Indices (en muestras, dentro del segmento filtrado ya alineado a
    onset - PADDED_PRE_S) para la ventana de ruido y la ventana de señal.
    """
    t0 = -PADDED_PRE_S  # tiempo del primer sample del segmento, relativo al onset

    def idx(t):
        return int(round((t - t0) * sr))

    return {
        "noise_start": idx(-NOISE_PRE_S),
        "noise_end": idx(-NOISE_MARGIN_S),
        "signal_start": idx(-WINDOW_PRE_S),
        "signal_end": idx(WINDOW_POST_S),
        "total_samples": int(round(PADDED_TOTAL_S * sr)),
    }


# --- Descriptores espectrales de banda ancha (2026-09-06) ----------------
#
# La banda de 0.5-10 Hz deja fuera la frecuencia dominante del 95.5% de los VT
# de Laguna del Maule y del 95.5% de los VLP de Copahue, medido contra la
# frecuencia que el analista estima de forma independiente. Con esa banda,
# fpeak no guarda relacion con la frecuencia dominante real donde la clase
# cae afuera: la correlacion de rangos con el analista es 0.66 en Nevados de
# Chillan, cuyas clases caen dentro, y 0.001 en Laguna del Maule.
#
# Los cortes de abajo salen de la distribucion por clase, no de una eleccion
# a ojo. Cuartiles de la frecuencia del analista, ponderados por sistema:
#
#   VLP   p05 0.20   mediana 0.38   p95 0.64
#   LP    p05 1.05   mediana 1.67   p95 3.87
#   TO    p05 1.56   mediana 3.42   p95 6.73
#   AV    p05 1.66   mediana 3.69   p95 13.40
#   HB    p05 1.65   mediana 6.99   p95 10.72
#   VT    p05 8.18   mediana 11.35  p95 17.03
#
# El filtro se abre mas que la banda de evaluacion para que esta caiga en la
# zona plana: con 0.25-25 Hz la respuesta zero-phase es 98% en 0.4 Hz, 100%
# entre 0.9 y 5 Hz, 99% en 17 Hz y 93% en 20 Hz.
#
# El borde inferior no baja de 0.25 Hz: el microsismo oceanico secundario
# domina bajo esa frecuencia. Cuesta el 5% mas bajo de los VLP, cuya mediana
# de 0.38 Hz queda comoda por encima.
WIDE_FILTER_BAND = (0.25, 25.0)
WIDE_SPECTRAL_BAND = (0.35, 20.0)      # banda de evaluacion, dentro de lo plano
WIDE_BAND_VLP = (0.35, 0.9)            # VLP termina en 0.64, LP empieza en 1.05
WIDE_BAND_LP = (0.9, 5.0)              # LP y TO; AV y HB reparten
WIDE_BAND_VT = (5.0, 20.0)             # VT empieza en 8.18


def compute_spectral_wide(sig: np.ndarray, sr: float) -> dict:
    """Cuatro descriptores espectrales sobre la banda ancha.

    fpeak y SR conservan su definicion, medidos sobre 0.35-20 Hz en vez de
    1-8 Hz. Los dos indices de frecuencia son razones entre bandas contiguas:

      FI_hi = log10(A[5-20] / A[0.9-5])    contrasta VT contra LP
      FI_lo = log10(A[0.9-5] / A[0.35-0.9]) contrasta LP contra VLP

    El segundo es el que falta en el conjunto actual. Con una sola banda baja
    que empieza en 1 Hz, VLP y LP quedan del mismo lado del unico corte, y esa
    es la confusion que el diagnostico encontro en Copahue, donde LP y VLP son
    el par mas frecuente entre familias de forma de onda con etiqueta mezclada.

    Se reportan tambien las tres fracciones de energia por banda, que dicen lo
    mismo de otra manera y permiten leer a HB, cuya mediana de 6.99 Hz cae
    entre la banda de LP y la de VT.
    """
    amp = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / sr)

    def band_mean(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(amp[m].mean()) if m.any() else float("nan")

    def band_energy(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float((amp[m] ** 2).sum()) if m.any() else 0.0

    m = (freqs >= WIDE_SPECTRAL_BAND[0]) & (freqs <= WIDE_SPECTRAL_BAND[1])
    if not m.any() or not np.any(amp[m] > 0):
        keys = ["fpeak_w", "SR_w", "FI_hi", "FI_lo", "E_vlp", "E_lp", "E_vt"]
        return {k: float("nan") for k in keys}

    af, ff = amp[m], freqs[m]
    a_vlp, a_lp, a_vt = (band_mean(*WIDE_BAND_VLP), band_mean(*WIDE_BAND_LP),
                         band_mean(*WIDE_BAND_VT))
    e = np.array([band_energy(*WIDE_BAND_VLP), band_energy(*WIDE_BAND_LP),
                  band_energy(*WIDE_BAND_VT)])
    tot = e.sum()
    return {
        "fpeak_w": float(ff[np.argmax(af)]),
        "SR_w": float(af.mean() / af.max()),
        "FI_hi": float(np.log10(a_vt / a_lp)) if a_lp > 0 else float("nan"),
        "FI_lo": float(np.log10(a_lp / a_vlp)) if a_vlp > 0 else float("nan"),
        "E_vlp": float(e[0] / tot) if tot > 0 else float("nan"),
        "E_lp": float(e[1] / tot) if tot > 0 else float("nan"),
        "E_vt": float(e[2] / tot) if tot > 0 else float("nan"),
    }


# --- Modo de ventana por duracion (2026-09-06) ---------------------------
#
# El acondicionamiento v2 recorta una ventana fija de 10 s desde el onset. El
# trabajo previo (Morales et al., EPSL) usa en cambio la DURACION COMPLETA del
# evento, y ahi Wmax resulto el descriptor mas importante, con mediana ~2400
# muestras para LP. En una ventana de 1000 muestras ese valor no existe: Wmax
# se satura contra el borde, que es exactamente la censura que el diagnostico
# de descriptores encontro. Este modo replica aquel acondicionamiento.
#
# Diferencias respecto de v2, todas del trabajo previo:
#   - ventana = [onset, onset + duracion], truncada a MAX_WINDOW_SAMPLES
#   - eventos de duracion menor que MIN_DURATION_S excluidos
#   - banda del filtro 0.1-10 Hz en vez de 0.5-10 Hz
#   - ventana de ruido de 5 s en vez de 2 s
#
# Una diferencia deliberada: el trabajo previo mide el ruido en los 5 s
# inmediatamente anteriores al onset; aca se conserva el margen de 2 s de v2,
# porque un arribo emergente entra en la ventana de ruido y sesga el SNR.
# 2026-10-06: la esquina inferior sube de 0.1 a 0.5 Hz, la misma de la ventana
# fija. Con 0.1 Hz el microsismo oceanico entra a la ventana hasta la coda y,
# en estaciones costeras o de isla, domina los descriptores de recurrencia:
# en Hudson la correlacion de DET entre ambas bandas, evento a evento, es 0.04
# en VT (scripts/test_banda_coda.py). Los espectrales, evaluados en 1-8 Hz, no
# cambian, y la separacion LP/VT queda igual o mejora. Con la misma banda en
# las dos ventanas, la comparacion entre ellas mide solo el largo.
# Resultados con 0.1 Hz respaldados en data/processed/features_dur_banda01.
DURATION_FILTER_BAND = (0.5, 10.0)
MIN_DURATION_S = 5.0
MAX_WINDOW_SAMPLES = 10_000          # 100 s a 100 Hz
DUR_NOISE_WIN_S = 5.0
DUR_NOISE_MARGIN_S = 2.0
DUR_PADDED_PRE_S = DUR_NOISE_WIN_S + DUR_NOISE_MARGIN_S + EDGE_PAD_S   # 10 s


def duration_window_plan(duration_s: float, sr: float) -> dict:
    """Ventana y segmento con padding para un evento de duracion dada.

    Devuelve los segundos a pedir antes y despues del onset, y los indices de
    las ventanas de senal y ruido dentro del segmento filtrado.
    """
    if not np.isfinite(duration_s) or duration_s < MIN_DURATION_S:
        raise ExclusionError(
            f"duracion {duration_s} menor que el minimo de {MIN_DURATION_S} s")
    n_win = min(int(round(duration_s * sr)), MAX_WINDOW_SAMPLES)
    post_s = n_win / sr
    t0 = -DUR_PADDED_PRE_S

    def idx(t):
        return int(round((t - t0) * sr))

    return {
        "pre_s": DUR_PADDED_PRE_S,
        "post_s": post_s + EDGE_PAD_S,
        "n_window": n_win,
        "noise_start": idx(-(DUR_NOISE_WIN_S + DUR_NOISE_MARGIN_S)),
        "noise_end": idx(-DUR_NOISE_MARGIN_S),
        "signal_start": idx(0.0),
        "signal_end": idx(0.0) + n_win,
        "total_samples": int(round((DUR_PADDED_PRE_S + post_s + EDGE_PAD_S) * sr)),
    }


# --- Embedding (parametros ya publicados, no se reoptimizan) ---
M_TIME, TAU_TIME = 2, 3
M_FREQ, TAU_FREQ = 6, 1

# --- Seleccion de epsilon (grilla fija, identica al trabajo previo) ---
RADII = [0.1, 0.2, 0.4]
RR_TARGET = (0.01, 0.05)
RR_ACCEPT_UPPER = RR_TARGET[1] * 2  # 0.10, igual que mseed_rqa_features.py

# Bandas espectrales para fpeak/SR/FI (corregido 2026-08-24, ver paper):
# el filtro 0.1-10Hz de FILTER_BAND ya no es plano cerca del corte de 10Hz
# (retiene 99.9% a 4.5Hz, 97.6% a 6.5Hz, 87.3% a 8Hz, 50% a 10Hz, con
# sosfiltfilt zero-phase -- verificado con scipy.signal.sosfreqz). fpeak/SR
# se evaluan solo en la parte plana de la respuesta, y FI usa dos bandas
# separadas por un hueco (no un unico corte) para ensanchar la separacion
# entre clases, siguiendo Cui et al. 2021 / Buurman & West 2010.
# Banda de evaluacion de fpeak y SR. El borde inferior sube de 0.5 a 1.0 Hz
# junto con FILTER_BAND, por el mismo criterio con que el superior esta en 8 y
# no en 10: evaluar sobre la parte plana de la respuesta. El pasabanda
# 0.5-10 Hz de orden 4 zero-phase retiene 50.0% en 0.5 Hz, 83.8% en 0.6,
# 95.7% en 0.7, 99.7% en 0.9 y 99.9% desde 1.0 Hz.
SPECTRAL_BAND = (1.0, 8.0)
FI_LOW_BAND = (1.0, 2.5)            # banda baja de FI, mismo borde plano
FI_HIGH_BAND = (4.5, 8.0)           # banda alta de FI (hueco 2.5-4.5 Hz excluido)

RQA_KEYS = [
    "DET", "L", "Lmax", "DIV", "Lentr", "LAM",
    "TT", "Vmax", "Ventr", "W", "Wmax", "Wdiv", "Wentr",
]


class ExclusionError(ValueError):
    """Evento excluido por una razon de calidad de datos (nunca se rellena)."""


def preprocess_segment(raw: np.ndarray, sr: float, band: tuple = None) -> np.ndarray:
    """Detrend lineal -> pasabanda Butterworth orden 4, zero-phase, 0.1-10 Hz.

    Se aplica sobre un tramo con margen (EDGE_PAD_S a cada lado de la ventana
    final) para que los artefactos de borde del filtro no caigan dentro de la
    ventana de 1000 muestras que efectivamente se usa.
    """
    d = detrend(raw.astype(np.float64), type="linear")
    sos = butter(FILTER_ORDER, band or FILTER_BAND, btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, d)


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std == 0 or not np.isfinite(std):
        raise ExclusionError("señal constante (std=0), z-score no definido")
    return (x - x.mean()) / std


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def compute_snr_db(noise_filtered: np.ndarray, signal_filtered: np.ndarray) -> float:
    """20*log10(RMS(señal)/RMS(ruido)).

    Ambos tramos deben venir del segmento filtrado en SNR_BAND, no en
    FILTER_BAND -- ver la nota de SNR_BAND arriba.
    """
    r_noise = rms(noise_filtered)
    r_signal = rms(signal_filtered)
    if r_noise == 0:
        return float("nan")
    return 20.0 * np.log10(r_signal / r_noise)


def _rqa_result_to_dict(result) -> dict:
    d = {
        "DET": float(result.determinism),
        "L": float(result.average_diagonal_line),
        "Lmax": float(result.longest_diagonal_line),
        "DIV": float(result.divergence),
        "Lentr": float(result.entropy_diagonal_lines),
        "LAM": float(result.laminarity),
        "TT": float(result.trapping_time),
        "Vmax": float(result.longest_vertical_line),
        "Ventr": float(result.entropy_vertical_lines),
        "W": float(result.average_white_vertical_line),
        "Wmax": float(result.longest_white_vertical_line),
        "Wdiv": float(result.longest_white_vertical_line_inverse),
        "Wentr": float(result.entropy_white_vertical_lines),
    }
    # DIV / Wdiv son 1/longest_line: si la linea mas larga es 0, dan inf.
    # Se homologa a NaN (mismo criterio que el trabajo previo).
    for k in ("DIV", "Wdiv"):
        if not np.isfinite(d[k]):
            d[k] = float("nan")
    return d


def compute_rqa_domain(x: np.ndarray, m: int, tau: int) -> tuple[dict, float, float]:
    """Busca epsilon en la grilla fija RADII (por evento, por dominio).

    Devuelve (descriptores_13, eps_elegido, rr_conseguido). Si todos los
    intentos fallan (excepcion de pyrqa), devuelve descriptores NaN, eps=NaN,
    rr=NaN en vez de fabricar un valor.
    """
    best = None
    best_radius = None
    best_dist = float("inf")

    for radius in RADII:
        try:
            ts = TimeSeries(x, embedding_dimension=m, time_delay=tau)
            settings = Settings(
                ts,
                analysis_type=Classic,
                neighbourhood=FixedRadius(radius),
                similarity_measure=EuclideanMetric,
                theiler_corrector=1,
            )
            settings.min_diagonal_line_length = 2
            settings.min_vertical_line_length = 2
            settings.min_white_vertical_line_length = 2

            result = RQAComputation.create(settings, verbose=False).run()
            rr = result.recurrence_rate

            if RR_TARGET[0] <= rr <= RR_ACCEPT_UPPER:
                return _rqa_result_to_dict(result), float(radius), float(rr)

            dist = abs(rr - np.mean(RR_TARGET))
            if dist < best_dist:
                best_dist = dist
                best = result
                best_radius = radius
        except Exception:
            continue

    if best is None:
        return {k: float("nan") for k in RQA_KEYS}, float("nan"), float("nan")
    return _rqa_result_to_dict(best), float(best_radius), float(best.recurrence_rate)


def compute_spectral_features(sig: np.ndarray, sr: float,
                               freq_min: float = SPECTRAL_BAND[0],
                               freq_max: float = SPECTRAL_BAND[1],
                               fi_low_band: tuple = FI_LOW_BAND,
                               fi_high_band: tuple = FI_HIGH_BAND):
    """fpeak, SR, FI + el vector log1p(amplitud) enmascarado, listo para RQA
    en frecuencia (m=6, tau=1).

    Corregido 2026-08-24, dos cambios:
    1. SR: estaba definido como razon de amplitud sumada baja/alta banda,
       redundante con FI (misma idea, mismas dos bandas). Se reemplaza por la
       definicion de Cui et al. 2021: razon entre amplitud espectral media y
       maxima -- mide concentracion/ancho de banda del espectro (cerca de 0 =
       fuente armonica de banda angosta, cerca de 1 = espectro plano de banda
       ancha), una dimension distinta de fpeak (ubicacion del pico) y FI
       (balance alto/bajo).
    2. Banda de evaluacion: fpeak/SR se calculan sobre 0.5-8 Hz (SPECTRAL_BAND),
       no sobre el rango completo del filtro (0.1-10 Hz) -- el filtro deja de
       ser plano cerca del corte de 10 Hz (retiene 87.3% a 8Hz, 50% a 10Hz con
       sosfiltfilt). FI usa dos bandas separadas por un hueco (FI_LOW_BAND,
       FI_HIGH_BAND) en vez de un unico corte, para ensanchar la separacion
       entre clases (Cui et al. 2021; Buurman & West 2010).
    """
    spectrum = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / sr)
    amp = np.abs(spectrum)

    mask = (freqs >= freq_min) & (freqs <= freq_max)
    freqs_f, amp_f = freqs[mask], amp[mask]

    if len(amp_f) == 0 or not np.any(amp_f > 0):
        return {"fpeak": float("nan"), "SR": float("nan"), "FI": float("nan")}, None

    peak_freq = float(freqs_f[np.argmax(amp_f)])
    spectral_ratio = float(amp_f.mean() / amp_f.max())

    low_mask = (freqs >= fi_low_band[0]) & (freqs <= fi_low_band[1])
    high_mask = (freqs >= fi_high_band[0]) & (freqs <= fi_high_band[1])
    a_lower = amp[low_mask].mean() if low_mask.any() else float("nan")
    a_upper = amp[high_mask].mean() if high_mask.any() else float("nan")
    fi = float(np.log10(a_upper / a_lower)) if (a_lower and a_lower > 0) else float("nan")

    spectral = {"fpeak": peak_freq, "SR": spectral_ratio, "FI": fi}
    amp_log_masked = np.log1p(amp_f.clip(min=0))
    return spectral, amp_log_masked
