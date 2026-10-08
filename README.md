# Paper 2 — Transferring volcano-seismic classifiers across datasets

Private repository for the second article of Sergio Morales Méndez's doctoral thesis (Doctoral Program in Engineering, Universidad de La Frontera; OVDAS–SERNAGEOMIN), in collaboration with Andrés Ávila, Mailyn Moreno Espino and Oscar Valderrama Cayumán. It addresses **specific objective 3** of the thesis: to evaluate the generalization and transfer of classifiers across volcanic contexts and installation conditions.

> **Status:** draft proposal for the co-authors. Data, Methodology and Results are written; Introduction, Discussion, Conclusion and Abstract will be written once the analyses are closed. The focus is being reoriented (see below), and the structure of the text may still change.

## Central question (focus since 2026-10-08)

**How many expert labels does a new dataset need, and do recent methods reduce that burden?**

The work is framed from the data side: volcano seismology is the case study, with about 20 datasets from five agencies (OVDAS, SGC, VCSEIS, VOISS-Net and Deception Island). The three axes are:

1. **Transfer without target labels:** what carries over from one dataset to another (LP and VT transfer; local classes such as HB do not).
2. **Label budget curve:** how many labeled events of the new dataset are needed to reach a local model, comparing logistic regression, XGBoost and TabPFN, and active learning against random selection.
3. **Expert bias:** confident learning and unsupervised clustering to separate transfer error from differences in labeling criteria between centers (for example, the HB/LP criterion based on high-frequency content at the onset).

The title in the `.tex` is provisional; a data-oriented title is under discussion, e.g. *How many expert labels does a new dataset need?* The target journal is also under discussion (e.g. IEEE TGRS); for now the text uses the Springer template.

## Structure

```
.
├── sn-article.tex        article text (Springer Nature template, sn-jnl)
├── references.bib        bibliography
├── sn-jnl.cls, bst/, *.bst   template files
├── fig_*.pdf / fig_*.png article figures
└── analisis/             code, derived data, results and notebooks (see analisis/README.md)
```

## Compiling

The repository is linked to **Overleaf**, which reads this GitHub repository and compiles `sn-article.tex`. To see the updated PDF, sync the Overleaf project (Menu → GitHub → Pull GitHub changes). No local compilation is needed.

## Analysis and data

Everything needed to review and reproduce the results is in [`analisis/`](analisis/README.md):

- **Notebooks** that run on their own, reading only `analisis/datos` and `analisis/resultados` (the quickest way to explore).
- **Derived data** from the validation catalogs (descriptors and manual segmentation). The public waveforms can be downloaded with the scripts.
- **Results**: aggregated tables behind the numbers and figures of the text.
- **Scripts** of the analysis project.

The original SGC files (shared for this collaboration) and the institutional OVDAS data are not included; they are shared separately on request.

## Related work

- Descriptor package (RQA and spectral), public: <https://github.com/ProfeSergioM/descriptores-volcanicos>
- Paper 1 (preprint): Morales, Ávila, Franco and Malik (2026), *Recurrence Quantification Analysis of Volcanic Seismicity: Physically Interpretable Features for Multi-class Event Classification*, ESS Open Archive, doi:10.22541/essoar.15004691/v2.
- Systematic review (specific objective 1): Morales, Ávila and Moreno Espino (2026), *Artificial intelligence techniques for volcanic seismic signal analysis: a systematic review*, Earth Science Informatics 19:26, doi:10.1007/s12145-026-02073-2.

## Contact

Sergio Morales Méndez — s.morales01@ufromail.cl
