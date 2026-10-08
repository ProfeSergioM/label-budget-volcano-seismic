# Paper 2 — Transferencia de clasificadores sismovolcánicos entre datasets

Repositorio privado del segundo artículo de la tesis doctoral de Sergio Morales Méndez (Programa de Doctorado en Ingeniería, Universidad de La Frontera; OVDAS–SERNAGEOMIN), en colaboración con Andrés Ávila, Mailyn Moreno Espino y Oscar Valderrama Cayumán. Responde al **objetivo específico 3** de la tesis: evaluar la generalización y la transferencia de los clasificadores entre contextos volcánicos y condiciones de instalación.

> **Estado:** borrador en propuesta para los coautores. Data, Methodology y Results están escritos; Introduction, Discussion, Conclusion y Abstract se escriben al cerrar los análisis. El enfoque se está reorientando (ver abajo) y la estructura del texto todavía puede cambiar.

## Pregunta central (enfoque desde 2026-10-08)

**¿Cuántas etiquetas expertas necesita un dataset nuevo, y los métodos recientes reducen esa carga?**

El trabajo se plantea desde los datos: la sismología volcánica es el caso de estudio, con unos 20 datasets de cinco agencias (OVDAS, SGC, VCSEIS, VOISS-Net e Isla Decepción). Los ejes son:

1. **Transferencia sin etiquetas del destino:** qué se reutiliza de un dataset a otro (LP y VT se transfieren; las clases locales, como HB, no).
2. **Curva de presupuesto de etiquetas:** cuántos eventos etiquetados del dataset nuevo hacen falta para alcanzar un modelo local, comparando regresión logística, XGBoost y TabPFN, y aprendizaje activo frente a selección al azar.
3. **Sesgo del experto:** confident learning y agrupamiento no supervisado para separar el error de transferencia de las diferencias de criterio entre centros (por ejemplo, el criterio HB/LP basado en la alta frecuencia del onset).

Título provisorio en el `.tex`; en discusión un título orientado a datos, p. ej. *How many expert labels does a new dataset need?* Revista objetivo en discusión (p. ej. IEEE TGRS); por ahora el texto usa la plantilla Springer.

## Estructura

```
.
├── sn-article.tex        texto del artículo (plantilla Springer Nature, sn-jnl)
├── references.bib        bibliografía
├── sn-jnl.cls, bst/, *.bst   archivos de la plantilla
├── fig_*.pdf / fig_*.png figuras del artículo
└── analisis/             código, datos derivados, resultados y notebooks (ver analisis/README.md)
```

## Cómo se compila

El repositorio está conectado a **Overleaf**, que lee este GitHub y compila `sn-article.tex`. Para ver el PDF actualizado, sincroniza el proyecto en Overleaf (Menu → GitHub → Pull GitHub changes). No hace falta compilar localmente.

## Análisis y datos

Todo lo necesario para revisar y reproducir los resultados está en [`analisis/`](analisis/README.md):

- **Notebooks** autónomos que leen solo `analisis/datos` y `analisis/resultados` (vía rápida para explorar).
- **Datos derivados** de los catálogos de validación (descriptores y cortes manuales). Las formas de onda públicas se bajan con los scripts.
- **Resultados** agregados que respaldan las cifras y figuras del texto.
- **Scripts** del proyecto de análisis.

No se incluyen los archivos originales del SGC (cedidos para la colaboración) ni los datos institucionales de OVDAS; se comparten aparte a pedido.

## Código relacionado

- Paquete de descriptores (RQA y espectrales), público: <https://github.com/ProfeSergioM/descriptores-volcanicos>
- Paper 1 (preprint): Morales, Ávila, Franco y Malik (2026), *Recurrence Quantification Analysis of Volcanic Seismicity: Physically Interpretable Features for Multi-class Event Classification*, ESS Open Archive, doi:10.22541/essoar.15004691/v2.
- Revisión sistemática (OE1): Morales, Ávila y Moreno Espino (2026), *Artificial intelligence techniques for volcanic seismic signal analysis: a systematic review*, Earth Science Informatics 19:26, doi:10.1007/s12145-026-02073-2.

## Contacto

Sergio Morales Méndez — s.morales01@ufromail.cl
