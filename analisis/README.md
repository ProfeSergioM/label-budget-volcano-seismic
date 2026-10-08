# Análisis del paper: código, datos derivados y notebooks

Carpeta para los coautores (Andrés Ávila, Mailyn Moreno, Oscar Valderrama). Reúne lo que sostiene el paper actual —cuántas etiquetas expertas necesita un dataset nuevo: transferencia LP-VT entre datasets, curva de presupuesto de etiquetas y sesgo del experto— sin formas de onda ni archivos originales.

```
analisis/
├── README.md
├── requirements.txt
├── notebooks/     4 notebooks autónomos (leen solo ../datos y ../resultados)
├── datos/         descriptores y cortes manuales de los catálogos de validación
├── resultados/    tablas agregadas que respaldan las figuras y cifras del paper
└── scripts/       código del proyecto de análisis (se ejecuta desde el proyecto completo)
```

## Vía rápida: los notebooks

```bash
python -m venv venv && source venv/bin/activate
pip install -r analisis/requirements.txt      # para los notebooks basta: numpy pandas pyarrow matplotlib scikit-learn jupyter
cd analisis/notebooks
jupyter lab                                   # o: jupyter nbconvert --to notebook --execute --inplace 01_*.ipynb
```

| Notebook | Pregunta | Lee |
|---|---|---|
| `01_datos_y_descriptores.ipynb` | Inventario por catálogo, sistema y clase; distribución de FI, DET y SR por sistema y clase; relación entre la ventana fija y la ventana hasta la coda | `datos/features16_*.parquet` |
| `02_transferencia.ipynb` | AUC LP-VT por sistema y descriptor; matrices fuente→destino; resumen por escalón; demostración ejecutable (logística entrenada en Cascade, evaluada en Hawái, Norte de California y Alaska) | `resultados/transferibilidad_*.csv`, `resultados/transferencia_resumen_*.csv`, `datos/features16_fija.parquet` |
| `03_presupuesto_de_etiquetas.ipynb` | Gm frente a n etiquetas del destino, por destino: fuente + azar, fuente + activo, solo destino y modelo local; logística, XGBoost y TabPFN | `resultados/presupuesto_resumen.csv` |
| `04_sesgo_del_experto.ipynb` | Confident learning en Hudson (sospechosos por clase, revisión manual) y criterio HB/LP en el onset por cuartil de SNR (Hudson, Copahue, Ruiz-Olleta, Ruiz-Bis) | `resultados/sesgo_experto_hudson_*.csv`, `resultados/onset_hb_lp_*.csv` |

Los notebooks se guardan ejecutados, con sus figuras. Todo se presenta por sistema; ningún resultado promedia volcanes distintos en una sola cifra sin mostrar el detalle.

**Advertencias vigentes**

- `presupuesto_resumen.csv` es **provisorio**: se generó mientras el cálculo de la curva seguía corriendo (faltan destinos y pares). Se regenera con el comando de la sección de la curva de presupuesto.
- XGBoost "solo destino" con n = 10 da Gm ≈ 0 por un artefacto de configuración (con tan pocos eventos el modelo predice una sola clase); se corregirá.

## Datos públicos de validación y cómo obtenerlos

| Catálogo | Sistemas | Origen | Script |
|---|---|---|---|
| **VCSEIS** (Zhong y Tan, 2024, GRL 51, e2024GL108438; CC-BY 4.0) | Alaska, Cascade, Hawái, Norte de California | Formato SeisBench (`seisbench.data.VCSEIS`), servido por HTTPS en `https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets/vcseis/` | `fetch_vcseis_muestra.py` baja por lectura de rango HTTP solo las trazas de la muestra (sin descargar los ~43 GB); `extract_vcseis.py` calcula descriptores |
| **Isla Decepción** (Cortés et al., 2018; Zenodo, doi:10.5281/zenodo.1307437; CC-BY 4.0) | Decepcion95, Decepcion09 | Bases `dec.95M.16c` y `dec.09U.4c` del depósito de Zenodo, en `data/external/deception/` | `extract_deception.py` |
| **VOISS-Net** (etiquetas de Fee et al., 2025) | Kilauea, Pavlof, Redoubt, Semisopochnoi, Tanaga, Veniaminof (y Popocatépetl, solo tremor) | Lista de ventanas etiquetadas y traductor a pedidos FDSN (`voissnet_dataset.py`) enviados por D. Fee; formas de onda desde EarthScope (FDSN) | `extract_voissnet.py --plan`, `--descargar`, `--extraer` |
| **Volcán Nevado del Ruiz** (SGC) | Ruiz-Olleta, Ruiz-Bis | Archivos cedidos por el Servicio Geológico Colombiano para esta colaboración | `extract_sgc_ruiz.py` (no ejecutable sin los archivos) |

Los descriptores de los cuatro catálogos se calculan sobre cortes manuales: `cortar_trazas.py` arma la cola de trazas (`cola.csv`) y sirve la página de corte, donde se marca onset, fin y ruido (`picks_manuales.csv`); luego `extract_picks.py` / `extract_picks_16.py` calculan los descriptores sobre esos cortes y `extract_continuas.py` sobre las señales continuas (tremor, ruido). `extract_picks_auto.py` repite el cálculo con el corte automático, para compararlo con el manual (`comparar_corte.py`).

### Qué no está y por qué

- **Archivos originales del SGC** (Ruiz): cedidos para la colaboración, no se redistribuyen. Sí están sus descriptores derivados y sus cortes manuales, con autorización.
- **Formas de onda** de todos los catálogos: pesan decenas de GB; se obtienen de las fuentes públicas de la tabla anterior.
- **Descriptores de OVDAS** (Copahue, Hudson, Laguna del Maule, Nevados de Chillán, Villarrica): son datos institucionales y se comparten aparte, a pedido. En `resultados/` aparecen solo como valores agregados (AUC, Gm, recall, cuartiles), con una excepción acotada: `sesgo_experto_hudson_revision_manual.csv` lista los ~200 eventos de Hudson de la cola de revisión, con FI, DET, SR y fpeak, y `sesgo_experto_hudson_grupos_submuestra.csv` lista identificadores de evento con su grupo de agrupamiento (sin descriptores).
- La tabla por evento de confident learning de Hudson (`sesgo_experto_hudson_cleanlab_eventos.csv.gz`, ~2,5 MB) y los sorteos individuales del corte automático quedan en el proyecto completo.

## Archivos de datos (`datos/`)

**`features16_fija.parquet`**, **`features16_coda.parquet`** — un evento por fila (10 807), mismos eventos en ambos archivos; `fija` usa una ventana de 10 s (1000 muestras a 100 Hz) desde el onset y `coda` la ventana desde el onset hasta el fin de la coda.

| Columnas | Contenido |
|---|---|
| `item_id` | identificador `<dataset>:<ruta de la traza>`; une con `picks_manuales.csv` y `cola.csv` |
| `dataset`, `sistema` | catálogo (`voissnet`, `vcseis`, `decepcion`, `sgc`) y sistema; Decepción trae `sistema = "Decepcion"` y la campaña en `item_id` (`dec.95` / `dec.09`) |
| `clase`, `clase_corpus`, `subtipo` | clase homologada (LP, VT, HB, EX, OT, RE, TR, AV), clase original del catálogo, subtipo (p. ej. TO = tornillo, analizado como LP) |
| `estacion`, `onset`, `fin` | estación y corte manual (s desde el inicio de la traza) |
| `duplicado`, `fuera_de_ventana`, `ruido_pisa_evento` | marcas de calidad; los análisis usan `motivo == ""` y `duplicado == False` |
| `motivo` | vacío si hay descriptores; si no, la razón (traza corta, señal constante, < 5 s) |
| `snr_db` | SNR del evento frente a la ventana de ruido marcada |
| `DET, L, Lmax, DIV, Lentr, LAM, TT, Vmax, Ventr, W, Wmax, Wdiv, Wentr` | 13 descriptores de RQA (m = 2, τ = 3, a 100 Hz); `eps_time`, `rr_time`: umbral ε y tasa de recurrencia |
| `fpeak`, `SR`, `FI` | frecuencia del pico espectral, razón espectral e índice de frecuencia |
| `n_window`, `duration_s_used` | muestras de la ventana; duración usada (solo `coda`) |

**`features16_continuas.parquet`** — 5 231 ventanas de señal continua (tremor TR, ruido RUIDO, NTR de Decepción) de VOISS-Net (Pavlof, Semisopochnoi, Veniaminof y Popocatépetl), Decepción (NTR) y el Ruiz (SGC, tremor). Mismos descriptores; `centro` y `largo_etiqueta` describen la ventana dentro del segmento etiquetado.

**`picks_manuales.csv`** — 13 850 cortes manuales (vale el último por `item_id`). `t0_utc`, `sps`; `ruido_ini`/`ruido_fin`, `onset`/`fin` en segundos desde `t0_utc`; `estado` (`bueno`, `descartado`, `dudoso`) y `motivo` del descarte; `auto_onset`/`auto_fin` (propuesta automática) y `origen` (`aceptado`, `editado`, `lote_auto`…); `analista`, `guardado_utc`; `clase_revisada` si el analista reclasificó.

**`cola.csv`** — 20 087 trazas en orden de revisión: `orden`, `item_id`, `dataset`, `sistema`, `clase`, `estacion`, `fuente` (trozo del catálogo de origen) y `extra` (JSON con la etiqueta original y el inicio de la ventana).

## Resultados (`resultados/`)

| Archivo | Contenido | Script |
|---|---|---|
| `transferibilidad_auc.csv` | AUC LP-VT por sistema, descriptor y ventana (`auc` = P(LP > VT)) | `ranking_transferibilidad.py` |
| `transferibilidad_ranking.csv` | ranking de descriptores: AUC mediana, IQR, mínimo, sistemas invertidos, puntaje | `ranking_transferibilidad.py` |
| `transferencia_resumen_pares.csv` | por par fuente→destino, matriz (M1 LP-VT-HB, M2 LP-VT), ventana, escalado y modelo: recall por clase, Gm, importancias por permutación (`pi_*`), escalón y pérdida respecto de la diagonal | `transferencia.py` → `resumen_transferencia.py` |
| `transferencia_resumen_escalones.csv`, `_m1.csv`, `_una_clase.csv` | Gm por escalón (diagonal, misma agencia, entre agencias); detalle de M1; recall en destinos de una sola clase | `resumen_transferencia.py` |
| `clasificacion_por_sistema_fija.csv`, `_clases_fija.csv`, `_confusion_fija.csv` | clasificación multiclase dentro de cada sistema: exactitud balanceada y Gm, recall/precisión por clase, matrices de confusión | `clasificacion_por_sistema.py` |
| `verif_prior_destino.csv`, `verif_temporal_destino.csv` | verificaciones de la transferencia: corrección por prior del destino (CC, BBSE, EM) y partición temporal frente a azar | `verificaciones_transferencia.py` |
| `corte_auto_*_fija.csv` | corte automático frente a manual: desplazamientos de onset y fin, cambio de descriptores, Gm dentro del sistema y en transferencia | `comparar_corte.py` |
| `presupuesto_resumen.csv` | curva de presupuesto: media sobre repeticiones de `gm`, `rec_LP`, `rec_VT` (y `gm_sd`, `n_reps`) por `destino`, `fuente`, `modelo` (log, xgb, tabpfn), `seleccion` (azar, activo, solo_destino, local) y `n` | `presupuesto_etiquetas.py` → `resumir_presupuesto.py` |
| `sesgo_experto_hudson_*.csv` | Hudson: agrupamiento HDBSCAN/DBSCAN ajustado por DBCV (Optuna), contingencia y pureza clase × grupo, confident learning por clase, cruce con la submuestra, cola de revisión manual | `sesgo_experto.py` |
| `onset_hb_lp_{hudson,copahue,ruiz}_resumen.csv` | por cuartil de SNR (`q`) y clase (`label`, HB o LP): SNR, n, `FI_onset`, `dFI` y `HF_onset` (fracción de energía sobre 4,5 Hz en el onset) | `onset_hb_lp.py` |
| `onset_hb_lp_hudson_revision.csv` | clase final de la revisión manual en Hudson con SNR, ΔFI y `HF_onset` medianos | `onset_hb_lp.py` |
| `candidatas_analisis_resumen.csv` | clases candidatas a globales (HB, tremor) frente a la referencia LP-VT (`VT_ref`): exactitud balanceada, recall, confusiones con LP y VT y AUC, por tipo de par (diagonal, mismo catálogo, entre catálogos) | `candidatas_analisis.py` (usa `candidatos_globales.py`) |

## Scripts (`scripts/`)

Copia de los scripts del proyecto de análisis que usa el paper y de los módulos locales que importan. Usan rutas relativas a la raíz del proyecto original (`ROOT/data/...`, `ROOT/experiments/...`, `ROOT/dist/descriptores_volcanicos`) y **se ejecutan desde el proyecto completo**, no desde esta carpeta; aquí están para leer el método. La vía autónoma son los notebooks.

| Grupo | Scripts |
|---|---|
| Núcleo de cálculo | `rqa_core.py` (filtrado, ventanas, RQA y descriptores espectrales), `coda_duration.py` (fin de coda por envolvente), `class_style.py` (colores por clase) |
| Datos públicos y corte | `fetch_vcseis_muestra.py`, `extract_vcseis.py`, `extract_deception.py`, `extract_voissnet.py`, `extract_sgc_ruiz.py`, `cortar_trazas.py` (cola y página de corte), `extract_picks.py`, `extract_picks_16.py`, `extract_continuas.py`, `extract_picks_auto.py` |
| Dependencias de figuras y acceso a OVDAS | `download_and_extract.py` (lectura de Winston de OVDAS; requiere configuración institucional), `make_fig_window.py`, `make_fig_coda_window.py` |
| Transferencia | `ranking_transferibilidad.py`, `heterogeneidad_clases.py` (cargador `datos()`), `transferencia.py`, `resumen_transferencia.py`, `verificaciones_transferencia.py`, `comparar_corte.py`, `clasificacion_por_sistema.py`, `candidatos_globales.py`, `candidatas_analisis.py` |
| Presupuesto de etiquetas | `presupuesto_etiquetas.py`, `resumir_presupuesto.py` |
| Sesgo del experto | `sesgo_experto.py`, `onset_hb_lp.py` |

Orden de ejecución en el proyecto completo (desde su raíz, `python scripts/<script>.py`; el uso detallado está en el docstring de cada script):

```bash
# 1. descriptores de los catálogos de validación (requiere formas de onda)
python scripts/extract_deception.py
python scripts/fetch_vcseis_muestra.py --chunks ak_lp ak_rg hw12t21_lp hw12t21_rg
python scripts/extract_voissnet.py --plan && python scripts/extract_voissnet.py --descargar
python scripts/cortar_trazas.py cola && python scripts/cortar_trazas.py servir --dataset voissnet
python scripts/extract_picks_16.py && python scripts/extract_continuas.py
# 2. transferencia
python scripts/ranking_transferibilidad.py
python scripts/transferencia.py --workers 3 && python scripts/resumen_transferencia.py
python scripts/clasificacion_por_sistema.py --ventana fija
python scripts/verificaciones_transferencia.py
python scripts/extract_picks_auto.py --ventana fija && python scripts/comparar_corte.py --ventana fija
python scripts/candidatos_globales.py && python scripts/candidatas_analisis.py
# 3. presupuesto de etiquetas
python scripts/presupuesto_etiquetas.py --workers 3 --reps 5
python scripts/resumir_presupuesto.py --entrada experiments/reports/presupuesto \
    --salida <repo del paper>/analisis/resultados/presupuesto_resumen.csv
# 4. sesgo del experto (requiere descriptores y trazas de OVDAS)
python scripts/sesgo_experto.py --sistema Hudson
python scripts/onset_hb_lp.py --sistema hudson   # también copahue, ruiz
```

Los pasos 2 a 4 también leen los descriptores de OVDAS (`data/processed/features*`), que no están aquí. El paquete de descriptores para observatorios (`descriptores`, en `dist/descriptores_volcanicos` del proyecto; repositorio `ProfeSergioM/descriptores-volcanicos`) lo usan `extract_voissnet.py` y `extract_sgc_ruiz.py`.

## Entorno

`requirements.txt` fija las versiones del entorno donde se generaron los resultados (Python 3.13). TabPFN y PyTorch solo hacen falta para `presupuesto_etiquetas.py`; ObsPy, h5py, PyRQA y pyopencl solo para la extracción de descriptores.
