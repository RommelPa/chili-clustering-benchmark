# Chili Clustering Benchmark

Benchmark reproducible para evaluar algoritmos de clustering aplicados a la tipología hidrológico-operativa anual del sistema regulado Chili.

## Objetivo

Construir un agrupamiento anual reproducible a partir de variables hidrológico-operativas, comparar algoritmos de clustering y seleccionar una tipología final con criterios de desempeño, estabilidad y soporte entre configuraciones.

## Datos

El archivo crudo no se versiona en Git.

Debe ubicarse localmente en:

```text
data/raw/BDREPRESAS.xlsx
```

Las salidas generadas se crean localmente y tampoco se versionan:

```text
data/processed/
outputs/
```

## Variables anuales

La unidad de análisis es el año hidrológico-operativo, desde AH-2012 hasta AH-2025.

Variables usadas:

```text
vol_min
vol_amp
inflow_avenida
inflow_estiaje
dias_derrame
```

## Instalación

```powershell
uv sync
```

## Validación

```powershell
uv run pytest -v
uv run ruff check .
```

## Ejecución completa

```powershell
uv run python -m src.pipeline
```

Este comando ejecuta todo el flujo:

```text
benchmark
data_summary
dataset_figures
ranking
partition_equivalence
stability
final_ranking
statistical_tests
algorithm_analysis
interpretation
figures
```

## Ejecución rápida de prueba

```powershell
uv run python -m src.pipeline --no-mlflow --max-stability-runs 5
```

Este modo solo valida el flujo hasta estabilidad. No genera ranking final porque limita las corridas de estabilidad.

## Salidas principales

```text
outputs/tables/final_partition_ranking.csv
outputs/tables/statistical_bootstrap_metrics.csv
outputs/tables/statistical_stability_mannwhitney.csv
outputs/tables/statistical_silhouette_wilcoxon.csv
outputs/tables/statistical_ari_matrix.csv
outputs/tables/statistical_significance_summary.csv
outputs/tables/algorithm_performance_summary.csv
outputs/tables/selected_partition_membership.csv
outputs/tables/selected_cluster_profiles.csv
outputs/tables/data_processing_summary.csv
outputs/tables/final_grouping_ranking_ieee.csv
outputs/figures/figure_1_monthly_total_storage.png
outputs/figures/figure_2_seasonal_storage_pattern.png
outputs/figures/feature_correlation_heatmap.png
outputs/figures/valid_partitions_by_algorithm.png
outputs/figures/selected_grouping_pca_projection.png
```

## Resultado actual

El agrupamiento seleccionado está representado por `spectral_002`.

```text
Configuración representante: spectral_002
Algoritmo representante: spectral
Clústeres: 2
Configuraciones con el mismo agrupamiento: 13
Algoritmos de soporte: agglomerative, birch, gaussian_mixture, kmeans, spectral
Score final: 0.969816
Estabilidad: 0.999338
```

Este resultado no significa que Spectral Clustering sea universalmente el mejor algoritmo. Significa que `spectral_002` representa el agrupamiento final seleccionado por el ranking integrado.