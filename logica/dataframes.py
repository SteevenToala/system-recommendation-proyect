"""Construcción de resúmenes de datos para el sistema de recomendación.

Este módulo centraliza las consultas a MongoDB y genera tres vistas base:
resumen de películas, resumen de categorías y perfil de clientes.
"""

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from pymongo import MongoClient


MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'BI_Final'
COLLECTION_NAME = 'fact_alquiler'


def normalizar_serie(serie: pd.Series) -> pd.Series:
    """Escala una serie numérica al rango 0-1."""
    valores = pd.to_numeric(serie, errors='coerce').fillna(0)
    minimo = float(valores.min())
    maximo = float(valores.max())
    if maximo == minimo:
        return pd.Series(np.zeros(len(valores)), index=valores.index)
    return (valores - minimo) / (maximo - minimo)


def convertir_a_dataframe(registros: Iterable[dict]) -> pd.DataFrame:
    """Convierte una lista de documentos de Mongo en un DataFrame plano."""
    registros = list(registros)
    if not registros:
        return pd.DataFrame()
    return pd.json_normalize(registros, sep='_')


def _agregar_metricas_resumen_peliculas(df: pd.DataFrame) -> pd.DataFrame:
    """Añade métricas normalizadas al resumen de películas."""
    if df.empty:
        return df

    df = df.copy()
    df['popularidad_norm'] = normalizar_serie(df['total_alquileres'])
    df['ingreso_norm'] = normalizar_serie(df['ingreso_promedio'])
    if 'duracion_promedio' in df.columns:
        duracion = pd.to_numeric(df['duracion_promedio'], errors='coerce')
        df['duracion_norm'] = normalizar_serie(duracion.fillna(duracion.median()))
    else:
        df['duracion_norm'] = 0.0
    return df


def _agregar_metricas_resumen_categorias(df: pd.DataFrame) -> pd.DataFrame:
    """Añade métricas normalizadas al resumen de categorías."""
    if df.empty:
        return df

    df = df.copy()
    df['popularidad_categoria_norm'] = normalizar_serie(df['total_alquileres'])
    df['ingreso_categoria_norm'] = normalizar_serie(df['ingreso_promedio'])
    df['score_categoria_global'] = (
        df['popularidad_categoria_norm'] * 0.6 + df['ingreso_categoria_norm'] * 0.4
    )
    return df


def _agregar_metricas_perfil_clientes(df: pd.DataFrame) -> pd.DataFrame:
    """Añade métricas normalizadas al perfil de clientes."""
    if df.empty:
        return df

    df = df.copy()
    df['actividad_norm'] = normalizar_serie(df['total_alquileres'])
    df['gasto_norm'] = normalizar_serie(df['gasto_promedio'])
    return df


def construir_dataframes_desde_mongo(top_n: int = 10) -> Dict[str, pd.DataFrame]:
    """Construye los DataFrames base que usan los algoritmos de recomendación."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fact_alquiler = db[COLLECTION_NAME]

    pipeline_peliculas = [
        {
            '$group': {
                '_id': '$pelicula_titulo',
                'pelicula_titulo': {'$first': '$pelicula_titulo'},
                'categoria_nombre': {'$first': '$categoria_nombre'},
                'total_alquileres': {'$sum': 1},
                'ingreso_promedio': {'$avg': '$ingreso'},
                'ingreso_total': {'$sum': '$ingreso'},
                'clientes_unicos': {
                    '$addToSet': {'$ifNull': ['$cliente_nombre_completo', '$cliente_nombre']}
                },
                'duracion_promedio': {'$avg': '$duracion_alquiler'},
            }
        },
        {
            '$project': {
                '_id': 0,
                'pelicula_ref': '$pelicula_titulo',
                'pelicula_titulo': '$pelicula_titulo',
                'categoria_nombre': {'$ifNull': ['$categoria_nombre', 'Sin categoria']},
                'total_alquileres': 1,
                'ingreso_promedio': 1,
                'ingreso_total': 1,
                'clientes_unicos': {'$size': '$clientes_unicos'},
                'duracion_promedio': 1,
            }
        },
        {'$sort': {'ingreso_promedio': -1}},
    ]

    pipeline_categorias = [
        {
            '$group': {
                '_id': '$categoria_nombre',
                'categoria_nombre': {'$first': '$categoria_nombre'},
                'total_alquileres': {'$sum': 1},
                'clientes_unicos': {
                    '$addToSet': {'$ifNull': ['$cliente_nombre_completo', '$cliente_nombre']}
                },
                'ingreso_total': {'$sum': '$ingreso'},
                'ingreso_promedio': {'$avg': '$ingreso'},
            }
        },
        {
            '$project': {
                '_id': 0,
                'categoria_nombre': {'$ifNull': ['$categoria_nombre', 'Sin categoria']},
                'total_alquileres': 1,
                'clientes_unicos': {'$size': '$clientes_unicos'},
                'ingreso_total': 1,
                'ingreso_promedio': 1,
            }
        },
        {'$sort': {'ingreso_total': -1}},
    ]

    pipeline_clientes = [
        {
            '$group': {
                '_id': {'$ifNull': ['$cliente_nombre_completo', '$cliente_nombre']},
                'cliente_ref': {'$first': {'$ifNull': ['$cliente_nombre_completo', '$cliente_nombre']}},
                'total_alquileres': {'$sum': 1},
                'peliculas_unicas_set': {'$addToSet': '$pelicula_titulo'},
                'categorias_unicas_set': {'$addToSet': '$categoria_nombre'},
                'gasto_total': {'$sum': '$ingreso'},
                'gasto_promedio': {'$avg': '$ingreso'},
            }
        },
        {
            '$project': {
                '_id': 0,
                'cliente_ref': 1,
                'total_alquileres': 1,
                'peliculas_unicas': {'$size': '$peliculas_unicas_set'},
                'categorias_unicas': {'$size': '$categorias_unicas_set'},
                'gasto_total': 1,
                'gasto_promedio': 1,
            }
        },
        {'$sort': {'gasto_total': -1}},
    ]

    registros_peliculas = list(fact_alquiler.aggregate(pipeline_peliculas))
    registros_categorias = list(fact_alquiler.aggregate(pipeline_categorias))
    registros_clientes = list(fact_alquiler.aggregate(pipeline_clientes))

    df_resumen_peliculas = convertir_a_dataframe(registros_peliculas)
    df_resumen_categorias = convertir_a_dataframe(registros_categorias)
    df_perfil_clientes = convertir_a_dataframe(registros_clientes)

    return {
        'df_resumen_peliculas': _agregar_metricas_resumen_peliculas(df_resumen_peliculas),
        'df_resumen_categorias': _agregar_metricas_resumen_categorias(df_resumen_categorias),
        'df_perfil_clientes': _agregar_metricas_perfil_clientes(df_perfil_clientes),
    }


def main() -> None:
    """Imprime una vista previa de los DataFrames construidos desde Mongo."""
    dfs = construir_dataframes_desde_mongo(top_n=10)

    for nombre in ['df_resumen_peliculas', 'df_resumen_categorias', 'df_perfil_clientes']:
        print(f'\n{nombre}:')
        df = dfs.get(nombre, pd.DataFrame())
        if df.empty:
            print('DataFrame vacio')
        else:
            print(df.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
