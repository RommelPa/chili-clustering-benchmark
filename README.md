# Chili Clustering Benchmark

Benchmark reproducible para evaluar algoritmos de clustering aplicados a la tipología hidrológico-operativa anual del sistema regulado Chili.

## Objetivo

Construir un agrupamiento anual reproducible a partir de variables hidrológico-operativas, comparar algoritmos de clustering y seleccionar una tipología final con métricas internas, estabilidad y restricciones estructurales. El soporte entre configuraciones se informa y se usa como desempate.

## Datos

El archivo crudo no se versiona en Git.

La ruta se declara en `rutas.datos_crudos` de `config/pipeline.yaml`. Coloque allí la fuente XLSX o XLSM exacta. El nombre y la extensión deben coincidir con la configuración.

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
uv sync --locked --python 3.11
```

## Validación

```powershell
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked mypy src
```

## Ejecución completa

```powershell
uv run --locked python -m src.pipeline
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
partition_comparison
algorithm_analysis
interpretation
figures
```

## Ejecución rápida de prueba

```powershell
uv run --locked python -m src.pipeline --no-mlflow --max-stability-runs 5
```

Este modo solo valida el flujo hasta estabilidad. No genera ranking final porque limita las corridas de estabilidad.

## Salidas principales

```text
outputs/tables/final_partition_ranking.csv
outputs/tables/partition_comparison.csv
outputs/tables/partition_ari_matrix.csv
outputs/tables/algorithm_performance_summary.csv
outputs/tables/selected_partition_membership.csv
outputs/tables/selected_cluster_profiles.csv
outputs/tables/data_processing_summary.csv
outputs/figures/figure_1_monthly_total_storage.png
outputs/figures/figure_2_seasonal_storage_pattern.png
outputs/figures/feature_correlation_heatmap.png
outputs/figures/valid_partitions_by_algorithm.png
outputs/figures/selected_grouping_pca_projection.png
```

## Resultados

El agrupamiento seleccionado se consulta en final_partition_ranking.csv. No se fija un ganador en este documento: depende de la fuente, la configuración y la versión del código.

## Datos y definiciones

- Fuente: hoja BASE del archivo configurado. Se leen los valores guardados por Excel; las macros no se ejecutan y el lector no recalcula fórmulas.
- Período analizado: 1 de diciembre de 2011 a 30 de noviembre de 2025. Diciembre de 2025 pertenece a AH-2026 y no integra un año completo.
- Capacidad útil de referencia: 393 hm³, confirmada por el autor como procedente de batimetría. DATOSTEC del archivo fuente presenta un total actualizado de 393,1781031 hm³ (D15), redondeado aquí a 393 hm³. La hoja documenta adopción de batimetrías 2025 para El Frayle el 01/10/2025 y El Pañe el 04/10/2025; el total anterior es 403,8269474 hm³ (D33). Es una referencia común, no la capacidad histórica vigente cada día. El informe técnico y la homogeneidad de las curvas históricas aún requieren contraste.
- VOL_TOTAL_REL es un cociente respecto de esa referencia, no una variable acotada a [0, 1]. Los excedentes históricos se conservan sin recorte.
- INGCHINEGASA es la señal de ingreso utilizada en Aguada Blanca. El archivo fuente contiene fórmulas de balance; su equivalencia con un aforo directo no se presupone.
- Registros de rebose en cm o mm permiten identificar ocurrencia, pero no se convierten a m³/s. REBOSE_EVENT identifica ocurrencia por embalse y dias_derrame cuenta días con ocurrencia en al menos uno.
- El procesamiento se detiene si las fechas o los datos necesarios para las variables anuales están incompletos. No se imputan los datos primarios.

### Comprobaciones sobre la fuente proporcionada

En el período analizado, 525 días superan 393 hm³ y el máximo registrado es 427,433031 hm³ (18 de marzo de 2025). Esta diferencia requiere contraste de la referencia batimétrica con las curvas de volumen utilizadas durante el registro. No se atribuye automáticamente a sobrealmacenamiento físico.
Los reboses expresados como alturas se incluyen en la ocurrencia de eventos; AH-2018 presenta 174 días con derrame en la fuente verificada.

INGCHINEGASA presenta un cambio de cálculo: desde el 22/03/2016 aparecen fórmulas que sustituyen balances negativos por cero. En el período verificado hay 1.571 fórmulas sin ese recorte, 3.539 con recorte y cuatro valores literales. Debe evaluarse la influencia de este cambio antes de interpretar diferencias entre años como exclusivamente hidrológicas.

La longitud de 14 años delimita el registro disponible; no demuestra suficiencia climática. No hay validación hidroclimática externa ni evaluación de demanda satisfecha.

## Evaluación y alcance

- La grilla declara 70 combinaciones, pero cuatro duplicados RBF de Spectral se eliminan porque n_neighbors no interviene con affinity=rbf. Se ejecutan 66 configuraciones efectivas: K-means 8, jerárquico 12, mezclas gaussianas 12, Spectral 12, BIRCH 12 y DBSCAN 10.
- Estabilidad: enumeración exhaustiva sin reemplazo del 85% de los años. Con 14 años se seleccionan 12 y se evalúan las 91 combinaciones, una vez cada una y en el mismo orden para todas las configuraciones. Las semillas de los algoritmos permanecen fijas; no se evalúa variabilidad entre inicializaciones. El z-score permanece fijado al registro original: se evalúa sensibilidad de los algoritmos a exclusión de años, condicionada a esa escala.
- El score integra tres métricas internas y estabilidad. La penalización denominada penalizacion_desbalance mide desbalance (1 - tamaño mínimo/tamaño máximo), no una penalización adicional independiente de tamaño absoluto.
- El soporte entre configuraciones se informa y se usa para desempatar; no forma parte de la suma ponderada ni constituye validación independiente.
- Particiones equivalentes significa idéntica pertenencia de años salvo numeración de etiquetas. ARI mide acuerdo, no equivalencia estadística.
- La comparación de estabilidad describe diferencias en submuestras comunes. Los cuantiles son rangos empíricos de perturbación, no intervalos de confianza poblacionales. No se emiten valores p ni veredictos de significación.
- Los resultados no evalúan pronóstico, causalidad, demanda satisfecha o confiabilidad del suministro.

Las advertencias de los algoritmos se guardan en benchmark_runs.csv y stability_iterations.csv. Las submuestras que producen menos de dos grupos efectivos o un grupo por observación se registran como fallidas. El score de estabilidad es el promedio válido multiplicado por la fracción de iteraciones válidas, para no favorecer ejecuciones con muchas fallas. conditional_stability_score conserva el promedio condicionado al éxito.

Las submuestras repetidas se reutilizan únicamente para algoritmos deterministas o con random_state explícito. Esta caché evita reajustes idénticos y no modifica el muestreo ni aumenta la muestra histórica.

## Reproducción y trazabilidad

Consulte [docs/REPRODUCIR.md](docs/REPRODUCIR.md) para instalar, verificar y ejecutar desde salidas limpias.

La verificación del 14/10/2026 en Windows con Python 3.11.9 aprobó 141 pruebas, Ruff y mypy. La ejecución completa con MLflow terminó sus 11 etapas con una lectura del Excel. Se reprodujeron, dentro de tolerancia numérica, las variables, matriz estandarizada, métricas y etiquetas del benchmark, ranking de particiones y membresía seleccionada. La revisión visual final de las figuras sigue pendiente.

`outputs/reports/run_manifest.json` registra SHA-256 de la fuente, código y configuración, versiones de Python y paquetes y commit base. Si se ejecuta con cambios locales, el commit por sí solo no identifica el código: deben conservarse también los hashes. Las tablas intermedias son dependencias del pipeline, no todas son tablas para el artículo.

Las pruebas se conservan en Git. Datos, resultados y MLflow permanecen locales. Para una entrega del estudio, conserve la fuente exacta con acceso autorizado, configuración, lockfile, versión del código, manifiesto y resultados seleccionados. Los resultados son descriptivos y retrospectivos.
