# Chili Clustering Benchmark

Benchmark reproducible para evaluar algoritmos de clustering aplicados a la tipología hidrológico-operativa anual del sistema regulado Chili.

## Objetivo

Construir una partición anual reproducible a partir de variables hidrológico-operativas, comparar algoritmos de clustering y seleccionar una tipología final con criterios de desempeño, equivalencia y estabilidad.

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
ranking
partition_equivalence
stability
final_ranking
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
outputs/tables/algorithm_performance_summary.csv
outputs/tables/selected_partition_membership.csv
outputs/tables/selected_cluster_profiles.csv
outputs/tables/data_processing_summary.csv
outputs/figures/
```

## Resultado actual

La partición seleccionada es `partition_001`.

```text
Representante: spectral_002
Algoritmo representante: spectral
Clústeres: 2
Corridas equivalentes: 13
Algoritmos de soporte: agglomerative, birch, gaussian_mixture, kmeans, spectral
Score final: 0.969816
Estabilidad: 0.999338
```

Este resultado no significa que Spectral Clustering sea universalmente el mejor algoritmo. Significa que `spectral_002` representa la partición final seleccionada por el ranking integrado.