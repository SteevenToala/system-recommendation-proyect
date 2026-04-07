from typing import Dict

import numpy as np
import pandas as pd
from pymongo import MongoClient


MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'BI_Final'
COLLECTION_NAME = 'fact_alquiler'


def normalizar_serie(serie: pd.Series) -> pd.Series:
    serie = pd.to_numeric(serie, errors='coerce').fillna(0)
    minimo = float(serie.min())
    maximo = float(serie.max())
    if maximo == minimo:
        return pd.Series(np.zeros(len(serie)), index=serie.index)
    return (serie - minimo) / (maximo - minimo)


def convertir_a_dataframe(data):
    data = list(data)
    if not data:
        return pd.DataFrame()
    return pd.json_normalize(data, sep='_')


def construir_dataframes_desde_mongo(top_n: int = 10) -> Dict[str, pd.DataFrame]:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fact_alquiler = db[COLLECTION_NAME]

    resultado_2 = list(fact_alquiler.aggregate([
        {'$group': {
            '_id': '$pelicula_titulo',
            'pelicula_titulo': {'$first': '$pelicula_titulo'},
            'categoria_nombre': {'$first': '$categoria_nombre'},
            'total_alquileres': {'$sum': 1},
            'ingreso_promedio': {'$avg': '$ingreso'},
            'ingreso_total': {'$sum': '$ingreso'},
            'clientes_unicos': {'$addToSet': {'$ifNull': ['$cliente_nombre_completo', '$cliente_nombre']}},
            'duracion_promedio': {'$avg': '$duracion_alquiler'},
        }},
        {'$project': {
            '_id': 0,
            'pelicula_ref': '$pelicula_titulo',
            'pelicula_titulo': '$pelicula_titulo',
            'categoria_nombre': {'$ifNull': ['$categoria_nombre', 'Sin categoria']},
            'total_alquileres': 1,
            'ingreso_promedio': 1,
            'ingreso_total': 1,
            'clientes_unicos': {'$size': '$clientes_unicos'},
            'duracion_promedio': 1,
        }},
        {'$sort': {'ingreso_promedio': -1}},
    ]))

    resultado_3 = list(fact_alquiler.aggregate([
        {'$group': {
            '_id': '$categoria_nombre',
            'categoria_nombre': {'$first': '$categoria_nombre'},
            'total_alquileres': {'$sum': 1},
            'clientes_unicos': {'$addToSet': {'$ifNull': ['$cliente_nombre_completo', '$cliente_nombre']}},
            'ingreso_total': {'$sum': '$ingreso'},
            'ingreso_promedio': {'$avg': '$ingreso'},
        }},
        {'$project': {
            '_id': 0,
            'categoria_nombre': {'$ifNull': ['$categoria_nombre', 'Sin categoria']},
            'total_alquileres': 1,
            'clientes_unicos': {'$size': '$clientes_unicos'},
            'ingreso_total': 1,
            'ingreso_promedio': 1,
        }},
        {'$sort': {'ingreso_total': -1}},
    ]))

    resultado_4 = list(fact_alquiler.aggregate([
        {'$group': {
            '_id': {'$ifNull': ['$cliente_nombre_completo', '$cliente_nombre']},
            'cliente_ref': {'$first': {'$ifNull': ['$cliente_nombre_completo', '$cliente_nombre']}},
            'total_alquileres': {'$sum': 1},
            'peliculas_unicas_set': {'$addToSet': '$pelicula_titulo'},
            'categorias_unicas_set': {'$addToSet': '$categoria_nombre'},
            'gasto_total': {'$sum': '$ingreso'},
            'gasto_promedio': {'$avg': '$ingreso'},
        }},
        {'$project': {
            '_id': 0,
            'cliente_ref': 1,
            'total_alquileres': 1,
            'peliculas_unicas': {'$size': '$peliculas_unicas_set'},
            'categorias_unicas': {'$size': '$categorias_unicas_set'},
            'gasto_total': 1,
            'gasto_promedio': 1,
        }},
        {'$sort': {'gasto_total': -1}},
    ]))


    df_resumen_peliculas = convertir_a_dataframe(resultado_2)
    df_resumen_categorias = convertir_a_dataframe(resultado_3)
    df_perfil_clientes = convertir_a_dataframe(resultado_4)

    if not df_resumen_peliculas.empty:
        df_resumen_peliculas['popularidad_norm'] = normalizar_serie(df_resumen_peliculas['total_alquileres'])
        df_resumen_peliculas['ingreso_norm'] = normalizar_serie(df_resumen_peliculas['ingreso_promedio'])
        if 'duracion_promedio' in df_resumen_peliculas.columns:
            duracion = pd.to_numeric(df_resumen_peliculas['duracion_promedio'], errors='coerce')
            df_resumen_peliculas['duracion_norm'] = normalizar_serie(duracion.fillna(duracion.median()))
        else:
            df_resumen_peliculas['duracion_norm'] = 0.0

    if not df_resumen_categorias.empty:
        df_resumen_categorias['popularidad_categoria_norm'] = normalizar_serie(df_resumen_categorias['total_alquileres'])
        df_resumen_categorias['ingreso_categoria_norm'] = normalizar_serie(df_resumen_categorias['ingreso_promedio'])
        df_resumen_categorias['score_categoria_global'] = (
            df_resumen_categorias['popularidad_categoria_norm'] * 0.6
            + df_resumen_categorias['ingreso_categoria_norm'] * 0.4
        )

    if not df_perfil_clientes.empty:
        df_perfil_clientes['actividad_norm'] = normalizar_serie(df_perfil_clientes['total_alquileres'])
        df_perfil_clientes['gasto_norm'] = normalizar_serie(df_perfil_clientes['gasto_promedio'])

    return {
        'df_resumen_peliculas': df_resumen_peliculas,
        'df_resumen_categorias': df_resumen_categorias,
        'df_perfil_clientes': df_perfil_clientes,
    }


def main() -> None:
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
