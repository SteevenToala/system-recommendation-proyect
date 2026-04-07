# BI Final

Este flujo usa el archivo `BI-FINAL.xlsx`, donde la hoja principal es `FACT_ALQUILER` y las demas hojas funcionan como dimensiones.

1. Lee la tabla de hechos y las dimensiones.
2. Enriquese la informacion con nombres legibles de cliente, categoria, pelicula, tienda, ciudad, tiempo y actores.
3. Guarda el resultado en JSON e inserta los documentos en MongoDB.
4. Ejecuta K-means sobre `ingreso`.
5. Genera recomendaciones para un cliente especifico.

## Dependencias

```bash
pip install -r requirements.txt
```

## Cargar datos

```bash
python logica/carga_datos.py
```

El cargador elimina los IDs tecnicos y agrega campos legibles como `cliente_nombre_completo`, `categoria_nombre` y `pelicula_titulo`.

## Analizar clusters

```bash
python vistas/analisis_clusters.py
```

El analisis grafico usa solo segmentacion KMeans de clientes e incluye:
- clientes por segmento
- boxplot de gasto promedio por segmento
- dispersion actividad vs gasto (tamano por peliculas unicas)
- mapa de calor de centros KMeans normalizados
- comparativa de gasto/actividad promedio por cluster

## Sistema de recomendacion

```bash
python vistas/interfaz_recomendacion.py
```

La interfaz permite seleccionar cliente y generar recomendaciones sin argumentos por consola.

El modelo usa los DataFrames de soporte realmente necesarios construidos desde MongoDB:

- perfil de clientes (actividad, gasto promedio, variedad)
- resumen de peliculas (popularidad, ingreso promedio, duracion)
- resumen de categorias (popularidad e ingreso por categoria)
- matriz cliente-categoria normalizada (afinidad)
- perfiles temporales por mes (cliente y pelicula, cuando existe `tiempo_mes`)

Con esos DataFrames calcula un score hibrido que combina similitud colaborativa, afinidad de categoria, popularidad, ajuste al perfil del cliente, senal temporal y un pequeno factor de diversidad.

El algoritmo se declara explicitamente dentro de la funcion de recomendacion: hibrido (similitud coseno cliente-cliente + senales de contenido + ajuste por cluster KMeans).

## DataFrames desde Mongo (estilo Sakila)

```bash
python logica/dataframes.py
```

Este script genera y muestra DataFrames agregados que luego reutiliza el sistema de recomendacion.

## Interfaz grafica

```bash
python vistas/interfaz_recomendacion.py
```

Desde la ventana puedes seleccionar cliente, definir cantidad de recomendaciones y vecinos, ver resultados en tabla y exportar CSV.

## Arquitectura

- `logica/carga_datos.py`: carga desde Excel, enriquecimiento y subida a Mongo.
- `logica/dataframes.py`: construccion de dataframes agregados y features base.
- `logica/recomendacion.py`: motor de recomendacion y scoring (ejecutable en consola).
- `logica/clusters.py`: logica KMeans de segmentacion y exportacion para analisis de clusters (ejecutable en consola).
- `vistas/interfaz_recomendacion.py`: vista Tkinter del recomendador.
- `vistas/analisis_clusters.py`: vista y graficas del analisis de clusters.

## Salidas

- `fact_alquiler_proyecto.json`
- `clusters_fact_alquiler_proyecto.csv`
