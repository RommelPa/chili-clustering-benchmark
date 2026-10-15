# Reproducir el pipeline en Windows

Ejecute desde la raíz de un clon Git del repositorio. Se requieren Git, uv y la fuente exacta del estudio. No se requiere Docker.

## Entorno y datos

```powershell
uv sync --locked --python 3.11
```

El lockfile fija dependencias. No lo regenere ni actualice paquetes para reproducir una versión publicada. Coloque el Excel en data/raw/ y configure su ruta en rutas.datos_crudos de config/pipeline.yaml. XLSX y XLSM se leen con valores guardados; no se ejecutan macros ni se recalculan fórmulas. Conserve un archivo previamente recalculado y guardado por Excel.

## Verificaciones

```powershell
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked mypy src
```

Las pruebas de integración requieren la fuente; una suite que las omite no verifica la lectura real. Las advertencias sobre un gráfico interno de Excel deben distinguirse de errores en las celdas utilizadas.

## Ejecución completa desde salidas limpias

La primera instrucción enumera archivos generados. --apply los elimina; conserva fuente, entorno e historial MLflow. No limpie mientras hay una ejecución activa. Guarde fuera de esas carpetas los resultados de una versión que necesite conservar.

```powershell
uv run --locked python scripts/clean_outputs.py
uv run --locked python scripts/clean_outputs.py --apply
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
uv run --locked python -m src.pipeline
```

El pipeline genera sus dependencias en orden y lee el Excel una sola vez. La ejecución completa produce once etapas y un manifiesto. Para la fuente y configuración verificadas: 5.114 días, 14 años, cinco variables, 66 configuraciones, 38 retenidas y 14 particiones únicas. Cambiar fuente o diseño puede cambiar los resultados.

## Prueba corta y etapas independientes

```powershell
uv run --locked python -m src.pipeline --no-mlflow --max-stability-runs 5
```

La prueba corta termina en estabilidad y no produce resultados finales completos. No interprete resultados anteriores que hayan quedado en las carpetas como resultados de esa prueba.

Los módulos se pueden ejecutar por separado con python -m src.NOMBRE dentro de uv run --locked. Deben ejecutarse en el orden indicado en el README: los módulos posteriores leen tablas producidas anteriormente y no comprueban automáticamente toda su procedencia. Ante cambios de fuente o configuración, use la ejecución completa desde salidas limpias.

## Conservar y revisar resultados

Compruebe outputs/reports/run_manifest.json. Las versiones que se reportan en el artículo son las efectivas del manifiesto, no las mínimas de pyproject.toml. Conserve las advertencias registradas en las tablas.

Para revisión: membresía seleccionada, perfiles originales, ranking de particiones, resumen de estabilidad y comparación descriptiva. La selección de figuras para el artículo requiere revisión visual. Las tablas intermedias y las iteraciones detalladas permiten rastrear los resultados; no necesitan publicarse todas como tablas del manuscrito.

MLflow guarda experimentos localmente según config/experiments.yaml. Su historial no se elimina con clean_outputs.py. Datos, resultados y el respaldo temporal de auditoría están excluidos de Git.

## Estabilidad exhaustiva

El diseño enumera las 91 combinaciones de 12 de los 14 años, con igual peso.
Se mantienen la escala de entrada y los hiperparámetros originales. No es una
prueba de significancia ni los cuantiles de las diferencias son intervalos de
confianza poblacionales. Las combinaciones fallidas se registran y reducen la
estabilidad mediante la fracción válida. La comparación pareada solo incluye
combinaciones válidas para ambos representantes, por lo que debe informarse
su cobertura. El ranking final puede cambiar al sustituir 500 sorteos por la
enumeración exhaustiva; deben regenerarse todas las etapas posteriores.

Para repetir la evaluación desde las tablas del benchmark auditado:

```powershell
uv run --locked python -m src.stability
uv run --locked python -m src.final_ranking
uv run --locked python -m src.partition_comparison
uv run --locked python -m src.algorithm_analysis
uv run --locked python -m src.interpretation
uv run --locked python -m src.figures
```

Una ejecución completa posterior es necesaria para actualizar el manifiesto.
