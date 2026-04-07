from typing import Dict

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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


def construir_clusters_clientes(df_perfil_clientes: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    if df_perfil_clientes.empty:
        return pd.DataFrame(columns=['cliente_ref', 'cluster_id', 'cluster_nombre', 'cluster_gasto_centro'])

    base = df_perfil_clientes.copy()
    columnas_modelo = [
        col for col in ['gasto_promedio', 'total_alquileres', 'peliculas_unicas', 'categorias_unicas']
        if col in base.columns
    ]
    if not columnas_modelo:
        return pd.DataFrame(columns=['cliente_ref', 'cluster_id', 'cluster_nombre', 'cluster_gasto_centro'])

    for col in columnas_modelo:
        base[col] = pd.to_numeric(base[col], errors='coerce')
    base = base.dropna(subset=columnas_modelo)
    if base.empty:
        return pd.DataFrame(columns=['cliente_ref', 'cluster_id', 'cluster_nombre', 'cluster_gasto_centro'])

    k = max(1, min(int(n_clusters), len(base)))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(base[columnas_modelo].fillna(0.0))

    if k == 1:
        base['cluster_id'] = 0
        base['cluster_nombre'] = 'Perfil unico'
        base['cluster_gasto_centro'] = float(base['gasto_promedio'].mean()) if 'gasto_promedio' in base.columns else 0.0
        return base[['cliente_ref', 'cluster_id', 'cluster_nombre', 'cluster_gasto_centro']]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    base['cluster_id'] = kmeans.fit_predict(X_scaled)

    centros_gasto = base.groupby('cluster_id')['gasto_promedio'].mean().sort_values(ascending=False)
    nombres_ordenados = ['Mayor gasto', 'Gasto medio', 'Menor gasto']
    mapa_nombres = {
        cluster_id: nombres_ordenados[idx] if idx < len(nombres_ordenados) else f'Cluster {idx + 1}'
        for idx, cluster_id in enumerate(centros_gasto.index.tolist())
    }

    base['cluster_nombre'] = base['cluster_id'].map(mapa_nombres)
    base['cluster_gasto_centro'] = base['cluster_id'].map(centros_gasto.to_dict()).astype(float)

    return base[['cliente_ref', 'cluster_id', 'cluster_nombre', 'cluster_gasto_centro']]


def construir_dataframes_desde_mongo(top_n: int = 10) -> Dict[str, pd.DataFrame]:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fact_alquiler = db[COLLECTION_NAME]

    resultado_1 = list(fact_alquiler.aggregate([
        {'$match': {'ingreso': {'$ne': None}}},
        {'$sort': {'ingreso': -1}},
        {'$limit': int(top_n)},
        {'$project': {
            '_id': 0,
            'cliente_ref': {'$ifNull': ['$cliente_nombre_completo', '$cliente_nombre']},
            'pelicula_ref': '$pelicula_titulo',
            'categoria_nombre': '$categoria_nombre',
            'ingreso': '$ingreso',
            'duracion_alquiler': '$duracion_alquiler',
            'tiempo_mes': '$tiempo_mes',
        }}
    ]))

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

    resultado_5 = list(fact_alquiler.aggregate([
        {'$match': {'tiempo_mes': {'$ne': None}}},
        {'$group': {
            '_id': {
                'tiempo_mes': '$tiempo_mes',
                'categoria_nombre': '$categoria_nombre',
            },
            'total_alquileres': {'$sum': 1},
            'ingreso_total': {'$sum': '$ingreso'},
        }},
        {'$project': {
            '_id': 0,
            'tiempo_mes': '$_id.tiempo_mes',
            'categoria_nombre': {'$ifNull': ['$_id.categoria_nombre', 'Sin categoria']},
            'total_alquileres': 1,
            'ingreso_total': 1,
        }},
        {'$sort': {'tiempo_mes': 1, 'ingreso_total': -1}},
    ]))

    resultado_1_df = convertir_a_dataframe(resultado_1)
    resultado_2_df = convertir_a_dataframe(resultado_2)
    resultado_3_df = convertir_a_dataframe(resultado_3)
    resultado_4_df = convertir_a_dataframe(resultado_4)
    resultado_5_df = convertir_a_dataframe(resultado_5)

    if not resultado_2_df.empty:
        resultado_2_df['popularidad_norm'] = normalizar_serie(resultado_2_df['total_alquileres'])
        resultado_2_df['ingreso_norm'] = normalizar_serie(resultado_2_df['ingreso_promedio'])
        if 'duracion_promedio' in resultado_2_df.columns:
            duracion = pd.to_numeric(resultado_2_df['duracion_promedio'], errors='coerce')
            resultado_2_df['duracion_norm'] = normalizar_serie(duracion.fillna(duracion.median()))
        else:
            resultado_2_df['duracion_norm'] = 0.0

    if not resultado_3_df.empty:
        resultado_3_df['popularidad_categoria_norm'] = normalizar_serie(resultado_3_df['total_alquileres'])
        resultado_3_df['ingreso_categoria_norm'] = normalizar_serie(resultado_3_df['ingreso_promedio'])
        resultado_3_df['score_categoria_global'] = (
            resultado_3_df['popularidad_categoria_norm'] * 0.6
            + resultado_3_df['ingreso_categoria_norm'] * 0.4
        )

    if not resultado_4_df.empty:
        resultado_4_df['actividad_norm'] = normalizar_serie(resultado_4_df['total_alquileres'])
        resultado_4_df['gasto_norm'] = normalizar_serie(resultado_4_df['gasto_promedio'])

    df_clusters_clientes = construir_clusters_clientes(resultado_4_df, n_clusters=3)

    return {
        'resultado_1_df': resultado_1_df,
        'resultado_2_df': resultado_2_df,
        'resultado_3_df': resultado_3_df,
        'resultado_4_df': resultado_4_df,
        'resultado_5_df': resultado_5_df,
        'df_resumen_peliculas': resultado_2_df,
        'df_resumen_categorias': resultado_3_df,
        'df_perfil_clientes': resultado_4_df,
        'df_clusters_clientes': df_clusters_clientes,
    }


def main() -> None:
    dfs = construir_dataframes_desde_mongo(top_n=10)

    for nombre in ['resultado_1_df', 'resultado_2_df', 'resultado_3_df', 'resultado_4_df', 'resultado_5_df']:
        print(f'\n{nombre}:')
        df = dfs.get(nombre, pd.DataFrame())
        if df.empty:
            print('DataFrame vacio')
        else:
            print(df.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
