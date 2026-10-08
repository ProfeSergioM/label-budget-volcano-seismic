# Paper analysis: code, derived data and notebooks

Folder for the co-authors (Andrés Ávila, Mailyn Moreno, Oscar Valderrama). It gathers what supports the current paper (how many expert labels a new dataset needs: LP-VT transfer between datasets, the label-budget curve, and expert bias) without waveforms or original files.

```
analisis/
├── README.md
├── requirements.txt
├── notebooks/     4 self-contained notebooks (they read only ../datos and ../resultados)
├── datos/         descriptors and manual segmentation of the validation catalogs
├── resultados/    aggregated tables behind the figures and numbers of the paper
└── scripts/       code of the analysis project (run from the full project)
```

Folder, file and column names are kept in Spanish, as they appear in the data (`datos` = data, `resultados` = results). Below, every column is quoted verbatim and explained in English.

## Quick route: the notebooks

```bash
python -m venv venv && source venv/bin/activate
pip install -r analisis/requirements.txt      # the notebooks only need: numpy pandas pyarrow matplotlib scikit-learn jupyter
cd analisis/notebooks
jupyter lab                                   # or: jupyter nbconvert --to notebook --execute --inplace 01_*.ipynb
```

| Notebook | Question | Reads |
|---|---|---|
| `01_datos_y_descriptores.ipynb` | Inventory by catalog, system and class; distribution of FI, DET and SR by system and class; relation between the fixed window and the window up to the end of the coda | `datos/features16_*.parquet` |
| `02_transferencia.ipynb` | LP-VT AUC by system and descriptor; source→target matrices; summary by step; executable demonstration (logistic regression trained on Cascade, evaluated on Hawai`i, Northern California and Alaska) | `resultados/transferibilidad_*.csv`, `resultados/transferencia_resumen_*.csv`, `datos/features16_fija.parquet` |
| `03_presupuesto_de_etiquetas.ipynb` | Gm versus the number n of target labels, by target: source + random, source + active, target only, and local model; logistic regression, XGBoost and TabPFN | `resultados/presupuesto_resumen.csv` |
| `04_sesgo_del_experto.ipynb` | Confident learning on Hudson (suspects by class, manual review) and the HB/LP criterion at the onset by SNR quartile (Hudson, Copahue, Ruiz-Olleta, Ruiz-Bis) | `resultados/sesgo_experto_hudson_*.csv`, `resultados/onset_hb_lp_*.csv` |

The notebooks are saved executed, with their figures. Everything is presented by system; no result averages different volcanoes into a single number without showing the detail.

**Current caveats**

- `presupuesto_resumen.csv` is **provisional**: it was generated while the computation of the curve was still running (some targets and pairs are missing). It is regenerated with the command given in the label-budget section below.
- XGBoost "target only" with n = 10 gives Gm ≈ 0 because of a configuration artifact (with so few events the model predicts a single class); this will be fixed.

## Public validation data and how to obtain them

| Catalog | Systems | Source | Script |
|---|---|---|---|
| **VCSEIS** (Zhong and Tan, 2024, GRL 51, e2024GL108438; CC-BY 4.0) | Alaska, Cascade, Hawai`i, Northern California | SeisBench format (`seisbench.data.VCSEIS`), served over HTTPS at `https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets/vcseis/` | `fetch_vcseis_muestra.py` downloads, through HTTP range reads, only the traces in the sample (without downloading the ~43 GB); `extract_vcseis.py` computes the descriptors |
| **Deception Island** (Cortés et al., 2018; Zenodo, doi:10.5281/zenodo.1307437; CC-BY 4.0) | Decepcion95, Decepcion09 | Databases `dec.95M.16c` and `dec.09U.4c` from the Zenodo deposit, in `data/external/deception/` | `extract_deception.py` |
| **VOISS-Net** (labels from Fee et al., 2025) | Kilauea, Pavlof, Redoubt, Semisopochnoi, Tanaga, Veniaminof (and Popocatépetl, tremor only) | List of labeled windows and translator to FDSN requests (`voissnet_dataset.py`) provided by D. Fee; waveforms from EarthScope (FDSN) | `extract_voissnet.py --plan`, `--descargar`, `--extraer` |
| **Nevado del Ruiz volcano** (SGC) | Ruiz-Olleta, Ruiz-Bis | Files provided by the Colombian Geological Survey (Servicio Geológico Colombiano) for this collaboration | `extract_sgc_ruiz.py` (cannot be run without the files) |

The descriptors of the four catalogs are computed on manual segmentation: `cortar_trazas.py` builds the trace queue (`cola.csv`) and serves the segmentation page, where onset, end and noise are marked (`picks_manuales.csv`); then `extract_picks.py` / `extract_picks_16.py` compute the descriptors on those segments and `extract_continuas.py` on the continuous signals (tremor, noise). `extract_picks_auto.py` repeats the computation with the automatic segmentation, to compare it with the manual one (`comparar_corte.py`).

### What is not included and why

- **Original SGC files** (Ruiz): provided for the collaboration and not redistributed. Their derived descriptors and manual segmentation are included, with permission.
- **Waveforms** of all catalogs: they amount to tens of GB; they are obtained from the public sources in the table above.
- **OVDAS descriptors** (Copahue, Hudson, Laguna del Maule, Nevados de Chillán, Villarrica): these are institutional data and are shared separately, on request. In `resultados/` they appear only as aggregated values (AUC, Gm, recall, quartiles), with one limited exception: `sesgo_experto_hudson_revision_manual.csv` lists the ~200 Hudson events of the review queue, with FI, DET, SR and fpeak, and `sesgo_experto_hudson_grupos_submuestra.csv` lists event identifiers with their clustering group (without descriptors).
- The per-event confident-learning table for Hudson (`sesgo_experto_hudson_cleanlab_eventos.csv.gz`, ~2.5 MB) and the individual draws of the automatic segmentation remain in the full project.

## Data files (`datos/`)

**`features16_fija.parquet`**, **`features16_coda.parquet`**: one event per row (10,807), the same events in both files; `fija` (fixed) uses a 10 s window (1000 samples at 100 Hz) from the onset, and `coda` uses the window from the onset to the end of the coda.

| Columns | Content |
|---|---|
| `item_id` | identifier `<dataset>:<trace path>`; joins with `picks_manuales.csv` and `cola.csv` |
| `dataset`, `sistema` | catalog (`voissnet`, `vcseis`, `decepcion`, `sgc`) and system; Deception has `sistema = "Decepcion"` and the campaign in `item_id` (`dec.95` / `dec.09`) |
| `clase`, `clase_corpus`, `subtipo` | harmonized class (LP, VT, HB, EX, OT, RE, TR, AV), original class of the catalog, subtype (e.g. TO = tornillo, analyzed as LP) |
| `estacion`, `onset`, `fin` | station and manual segmentation (onset and end, in s from the start of the trace) |
| `duplicado`, `fuera_de_ventana`, `ruido_pisa_evento` | quality flags (duplicate, outside the window, noise window overlapping the event); the analyses use `motivo == ""` and `duplicado == False` |
| `motivo` | empty if descriptors exist; otherwise the reason (short trace, constant signal, < 5 s) |
| `snr_db` | SNR of the event relative to the marked noise window |
| `DET, L, Lmax, DIV, Lentr, LAM, TT, Vmax, Ventr, W, Wmax, Wdiv, Wentr` | 13 RQA descriptors (m = 2, τ = 3, at 100 Hz); `eps_time`, `rr_time`: threshold ε and recurrence rate |
| `fpeak`, `SR`, `FI` | spectral peak frequency, spectral ratio and frequency index |
| `n_window`, `duration_s_used` | samples in the window; duration used (`coda` only) |

**`features16_continuas.parquet`**: 5,231 windows of continuous signal (tremor TR, noise RUIDO, NTR from Deception) from VOISS-Net (Pavlof, Semisopochnoi, Veniaminof and Popocatépetl), Deception (NTR) and Ruiz (SGC, tremor). Same descriptors; `centro` (center) and `largo_etiqueta` (label length) describe the window within the labeled segment.

**`picks_manuales.csv`**: 13,850 manual segmentations (the last one per `item_id` is the valid one). `t0_utc`, `sps` (samples per second); `ruido_ini`/`ruido_fin` (noise start/end), `onset`/`fin` (onset/end) in seconds from `t0_utc`; `estado` (status: `bueno` = good, `descartado` = discarded, `dudoso` = doubtful) and `motivo` (reason for discarding); `auto_onset`/`auto_fin` (automatic proposal) and `origen` (origin: `aceptado` = accepted, `editado` = edited, `lote_auto` = automatic batch, …); `analista` (analyst), `guardado_utc` (save time); `clase_revisada` (revised class) if the analyst reclassified the event.

**`cola.csv`**: 20,087 traces in review order: `orden` (order), `item_id`, `dataset`, `sistema` (system), `clase` (class), `estacion` (station), `fuente` (chunk of the source catalog) and `extra` (JSON with the original label and the start of the window).

## Results (`resultados/`)

| File | Content | Script |
|---|---|---|
| `transferibilidad_auc.csv` | LP-VT AUC by system, descriptor and window (`auc` = P(LP > VT)) | `ranking_transferibilidad.py` |
| `transferibilidad_ranking.csv` | descriptor ranking: median AUC, IQR, minimum, inverted systems, score | `ranking_transferibilidad.py` |
| `transferencia_resumen_pares.csv` | for each source→target pair, matrix (M1 LP-VT-HB, M2 LP-VT), window, scaling and model: recall by class, Gm, permutation importances (`pi_*`), step, and loss relative to the diagonal | `transferencia.py` → `resumen_transferencia.py` |
| `transferencia_resumen_escalones.csv`, `_m1.csv`, `_una_clase.csv` | Gm by step (diagonal, same agency, between agencies); detail of M1; recall on single-class targets | `resumen_transferencia.py` |
| `clasificacion_por_sistema_fija.csv`, `_clases_fija.csv`, `_confusion_fija.csv` | multiclass classification within each system: balanced accuracy and Gm, recall/precision by class, confusion matrices | `clasificacion_por_sistema.py` |
| `verif_prior_destino.csv`, `verif_temporal_destino.csv` | checks of the transfer: correction for the target prior (CC, BBSE, EM) and temporal versus random split | `verificaciones_transferencia.py` |
| `corte_auto_*_fija.csv` | automatic versus manual segmentation: onset and end shifts, change in descriptors, Gm within the system and in transfer | `comparar_corte.py` |
| `presupuesto_resumen.csv` | label-budget curve: mean over repetitions of `gm`, `rec_LP`, `rec_VT` (and `gm_sd`, `n_reps`) by `destino` (target), `fuente` (source), `modelo` (model: log, xgb, tabpfn), `seleccion` (selection: `azar` = random, `activo` = active, `solo_destino` = target only, `local`) and `n` | `presupuesto_etiquetas.py` → `resumir_presupuesto.py` |
| `sesgo_experto_hudson_*.csv` | Hudson: HDBSCAN/DBSCAN clustering tuned by DBCV (Optuna), class × cluster contingency and purity, confident learning by class, cross-check with the subsample, manual review queue | `sesgo_experto.py` |
| `onset_hb_lp_{hudson,copahue,ruiz}_resumen.csv` | by SNR quartile (`q`) and class (`label`, HB or LP): SNR, n, `FI_onset`, `dFI` and `HF_onset` (fraction of energy above 4.5 Hz at the onset) | `onset_hb_lp.py` |
| `onset_hb_lp_hudson_revision.csv` | final class from the manual review on Hudson, with median SNR, ΔFI and `HF_onset` | `onset_hb_lp.py` |
| `candidatas_analisis_resumen.csv` | candidate global classes (HB, tremor) against the LP-VT reference (`VT_ref`): balanced accuracy, recall, confusions with LP and VT, and AUC, by pair type (diagonal, same catalog, between catalogs) | `candidatas_analisis.py` (uses `candidatos_globales.py`) |

## Scripts (`scripts/`)

Copy of the scripts of the analysis project used by the paper and of the local modules they import. They are verbatim copies of the working analysis code, so their docstrings and comments are in Spanish. They use paths relative to the root of the original project (`ROOT/data/...`, `ROOT/experiments/...`, `ROOT/dist/descriptores_volcanicos`) and **are run from the full project**, not from this folder; they are here so the method can be read. The self-contained route is the notebooks.

| Group | Scripts |
|---|---|
| Computational core | `rqa_core.py` (filtering, windows, RQA and spectral descriptors), `coda_duration.py` (end of coda from the envelope), `class_style.py` (colors by class) |
| Public data and segmentation | `fetch_vcseis_muestra.py`, `extract_vcseis.py`, `extract_deception.py`, `extract_voissnet.py`, `extract_sgc_ruiz.py`, `cortar_trazas.py` (queue and segmentation page), `extract_picks.py`, `extract_picks_16.py`, `extract_continuas.py`, `extract_picks_auto.py` |
| Figure dependencies and OVDAS access | `download_and_extract.py` (reads OVDAS Winston; requires institutional configuration), `make_fig_window.py`, `make_fig_coda_window.py` |
| Transfer | `ranking_transferibilidad.py`, `heterogeneidad_clases.py` (`datos()` loader), `transferencia.py`, `resumen_transferencia.py`, `verificaciones_transferencia.py`, `comparar_corte.py`, `clasificacion_por_sistema.py`, `candidatos_globales.py`, `candidatas_analisis.py` |
| Label budget | `presupuesto_etiquetas.py`, `resumir_presupuesto.py` |
| Expert bias | `sesgo_experto.py`, `onset_hb_lp.py` |

Execution order in the full project (from its root, `python scripts/<script>.py`; detailed usage is in the docstring of each script):

```bash
# 1. descriptors of the validation catalogs (requires waveforms)
python scripts/extract_deception.py
python scripts/fetch_vcseis_muestra.py --chunks ak_lp ak_rg hw12t21_lp hw12t21_rg
python scripts/extract_voissnet.py --plan && python scripts/extract_voissnet.py --descargar
python scripts/cortar_trazas.py cola && python scripts/cortar_trazas.py servir --dataset voissnet
python scripts/extract_picks_16.py && python scripts/extract_continuas.py
# 2. transfer
python scripts/ranking_transferibilidad.py
python scripts/transferencia.py --workers 3 && python scripts/resumen_transferencia.py
python scripts/clasificacion_por_sistema.py --ventana fija
python scripts/verificaciones_transferencia.py
python scripts/extract_picks_auto.py --ventana fija && python scripts/comparar_corte.py --ventana fija
python scripts/candidatos_globales.py && python scripts/candidatas_analisis.py
# 3. label budget
python scripts/presupuesto_etiquetas.py --workers 3 --reps 5
python scripts/resumir_presupuesto.py --entrada experiments/reports/presupuesto \
    --salida <paper repository>/analisis/resultados/presupuesto_resumen.csv
# 4. expert bias (requires OVDAS descriptors and traces)
python scripts/sesgo_experto.py --sistema Hudson
python scripts/onset_hb_lp.py --sistema hudson   # also copahue, ruiz
```

Steps 2 to 4 also read the OVDAS descriptors (`data/processed/features*`), which are not included here. The descriptor package for observatories (`descriptores`, in `dist/descriptores_volcanicos` of the project; repository `ProfeSergioM/descriptores-volcanicos`) is used by `extract_voissnet.py` and `extract_sgc_ruiz.py`.

## Environment

`requirements.txt` pins the versions of the environment in which the results were generated (Python 3.13). TabPFN and PyTorch are needed only for `presupuesto_etiquetas.py`; ObsPy, h5py, PyRQA and pyopencl only for the descriptor extraction.
