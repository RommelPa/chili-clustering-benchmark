# Revisar y reproducir en Windows (PowerShell)

## Aplicar el parche a su clon

Trabaje desde la carpeta del repositorio. El parche parte del commit f7799fcc9a8138e24eac24350eb46dcd71f5ca3d. Primero revise sus cambios locales: si existen, guárdelos en un commit propio antes de aplicar. No use reset --hard.

```powershell
git status --short
git rev-parse HEAD
git switch -c auditoria-articulo
git apply --stat "C:\ruta\auditoria-cambios.patch"
git apply --check "C:\ruta\auditoria-cambios.patch"
git apply "C:\ruta\auditoria-cambios.patch"
git diff --stat
git diff
```

Si --check falla, deténgase y revise la divergencia; no fuerce el parche. El ZIP es una copia del código actualizado, sin datos ni historial Git. Copiarlo encima del clon NO elimina archivos retirados: prefiera el parche. Este elimina src/statistical_tests.py y tests/test_statistical_tests.py y agrega su reemplazo descriptivo.

## Preparar entorno y fuente

Instale uv según su documentación oficial. El lockfile fija dependencias; no ejecute uv lock ni uv sync --upgrade para reproducir esta auditoría.

```powershell
uv sync --locked --python 3.11
```

Copie el XLSM original a data/raw/BDREPRESAS.xlsm. Alternativamente, copie BDREPRESAS.xlsx a data/raw/BDREPRESAS.xlsx y cambie únicamente rutas.datos_crudos en config/pipeline.yaml. Los valores BASE del período son idénticos en ambas fuentes verificadas, pero sus hashes serán distintos. Guarde el Excel recalculado antes de usarlo; openpyxl no recalcula fórmulas.

## Revisar y ejecutar

```powershell
uv run --locked pytest -v
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked python scripts/clean_outputs.py
uv run --locked python scripts/clean_outputs.py --apply
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
uv run --locked python -m src.pipeline
```

La primera limpieza solo enumera. --apply elimina resultados generados conocidos; conserva datos crudos, entorno e historial MLflow. El orden evita mezclar salidas viejas con la nueva ejecución. No limpie durante una ejecución.

Una prueba corta opcional es python -m src.pipeline --no-mlflow --max-stability-runs 5 dentro de uv run --locked. No produce todos los resultados finales ni reemplaza la ejecución completa.

Resultados esperados con la configuración auditada: 14 años, cinco variables, 66 configuraciones, 38 retenidas, 14 particiones únicas y representante spectral_001 con dos grupos. Compare outputs/reports/run_manifest.json: fuente, código, configuración, Python y versiones. IDs y puntajes pueden cambiar si se modifica el diseño. La comparación entre particiones es descriptiva; no produce p-valores.

## Herramientas y artículo

Se conservan Python, uv/uv.lock, pandas, NumPy, SciPy, scikit-learn, Pandera, openpyxl, Matplotlib y MLflow; pytest, Ruff y mypy verifican el código. No hace falta añadir otra plataforma para este alcance. Informe las versiones efectivas del manifiesto en el artículo, no las mínimas de pyproject.toml. Una reproducción exige también conservar la fuente exacta y reconocer el cambio de curvas de volumen en octubre de 2025.

Después de revisar:

```powershell
git add README.md .gitignore config src tests docs scripts
git commit -m "Audit hydrological clustering pipeline and reproducibility"
git push -u origin auditoria-articulo
```
