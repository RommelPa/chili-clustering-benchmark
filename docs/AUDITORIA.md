# Auditoría del código para el artículo

## Alcance y ejecución

Se revisaron los módulos de carga, preprocesamiento, variables anuales, matrices, algoritmos, métricas, ranking, equivalencia, estabilidad, comparación, interpretación, gráficos, configuración y pipeline, junto con sus pruebas y README. La ejecución completa se verificó con la fuente BDREPRESAS(3).xlsm y MLflow activado. Las macros y fórmulas de Excel no se ejecutan: se leen los valores guardados.

## Cambios confirmados

1. Fuente XLSM y corte 2011-12-01–2025-11-30 explícitos; el archivo crudo está excluido de Git.
2. Referencia batimétrica de 393 hm³ usada tanto en normalización como en figuras. Se conservan cocientes superiores a uno.
3. Reboses expresados como alturas cm/mm se conservan como indicadores de ocurrencia, sin convertirlos a caudal. Se separa evidencia de evento del valor numérico de flujo. El total de rebose numérico queda indefinido si falta alguna medición de flujo.
4. Datos primarios completos, fechas sin duplicados ni ausencias, finitud y no negatividad verificadas antes de agregar. El filtro de completitud del 90% no estaba implementado; se reemplazó por política explícita sin imputación.
5. Eliminados cuatro duplicados efectivos RBF de Spectral: 66 configuraciones, no 70.
6. Semilla semilla_aleatoria aplicada. Submuestras comunes para comparar ejecuciones; ajuste condicionado a escala z-score fijada sobre los 14 años originales.
7. Caché de submuestras idénticas para algoritmos deterministas o con semilla explícita. No cambia el muestreo.
8. Particiones degeneradas registradas como fallas. Estabilidad final ponderada por fracción válida; las etiquetas de ruido no cuentan como separación preservada.
9. Advertencias de algoritmos guardadas en las tablas. No se silencian las advertencias de grafos desconectados como si la ejecución fuera inequívoca.
10. Eliminada la etapa automática de significación. La comparación es descriptiva, con ARI, diferencias observadas y diferencias de estabilidad pareadas sobre submuestras comunes. Los cuantiles no son intervalos poblacionales; no se emiten veredictos de superioridad ni equivalencia estadística.
11. Equivalencia estructural exige mismos años, sin duplicados ni ejecuciones faltantes. Equivalencia significa pertenencia idéntica salvo etiquetas.
12. Penalización de desbalance nombrada según su fórmula. Soporte es descriptivo/desempate, no término ponderado ni validación externa.
13. Configuración y salidas obsoletas retiradas; README sin ganador fijo. Las figuras distinguen almacenamiento de sequía climática y mediana mensual de año observado.
14. Manifiesto de ejecución: SHA-256 de fuente, código y configuración, versiones de paquetes, Python y commit base.

## Resultados regenerados

- 5.114 días; 14 años hidrológico-operativos; cinco variables.
- 66 configuraciones exitosas; 38 retenidas; 14 particiones únicas.
- Representante: spectral_001, dos grupos, score 0,969632353 y estabilidad 0,998676471.
- Partición seleccionada: AH-2014, AH-2015, AH-2016 y AH-2023 constituyen el grupo de menor almacenamiento/aportes medios; los diez restantes constituyen el otro grupo.
- 12 configuraciones de cinco algoritmos reproducen la partición seleccionada.
- dias_derrame de AH-2018 cambia de 172 a 174.

Estos valores describen esta fuente y esta configuración, no rendimiento generalizable del algoritmo.

## Pendientes que el código no puede resolver

- DATOSTEC verificado: D15 = 393,1781031 hm³, D33 = 403,8269474 hm³ antes de 2025. Adopción de nuevas curvas: Frayle 01/10/2025 y Pañe 04/10/2025. Quedan pendientes informe técnico, niveles y homogeneidad histórica. Hay 525 días sobre la referencia y un máximo de 427,433031 hm³ el 2025-03-18. No se atribuye automáticamente a sobrealmacenamiento físico.
- INGCHINEGASA: 1.571 fórmulas de balance sin recorte, 3.539 fórmulas que sustituyen negativos por cero y cuatro registros literales en el período. El recorte aparece desde 2016-03-22. Debe documentarse su significado operativo y evaluar su influencia antes de llamarlo aforo directo o aporte natural total.
- Cambios de curvas de volumen y correcciones históricas de la fuente no pueden inferirse del Excel.
- No hay validación hidroclimática externa, prueba de suficiencia de 14 años ni evaluación de demanda satisfecha.

## Verificación

Suite completa: 145 pruebas aprobadas con la fuente real. Ruff y mypy aprobados (mypy requirió una caché nueva por corrupción de su caché local). Ejecución completa aprobada con MLflow; figuras inspeccionadas. El lector de Excel avisa que no interpreta un gráfico interno del archivo fuente; ello no implica pérdida de las celdas utilizadas y no se modifica el original.

Las pruebas antiguas de significación fueron sustituidas por pruebas de comparación descriptiva y casos de regresión para las correcciones. El número de pruebas no es directamente comparable con la suite original.

El XLSX entregado posteriormente tiene exactamente los mismos valores BASE en las 5.114 filas del período que el XLSM auditado. No se sustituyó automáticamente la fuente de la ejecución.
