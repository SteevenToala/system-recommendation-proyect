import numpy as np
import pandas as pd
import sys
from pymongo import MongoClient
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from logica.clusters import construir_clusters_clientes
from logica.dataframes import construir_dataframes_desde_mongo


MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'BI_Final'
COLLECTION_NAME = 'fact_alquiler'


def cargar_datos_mongo() -> pd.DataFrame:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    df = pd.DataFrame(list(collection.find({}, {'_id': 0})))
    if df.empty:
        raise ValueError('No hay datos cargados en MongoDB para recomendar.')
    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'ingreso' not in df.columns:
        raise ValueError('Falta la columna ingreso para recomendar.')
    df['ingreso'] = pd.to_numeric(df['ingreso'], errors='coerce')

    if 'id_cliente' in df.columns:
        df['id_cliente'] = pd.to_numeric(df['id_cliente'], errors='coerce')
        df = df.dropna(subset=['id_cliente'])
        df['id_cliente'] = df['id_cliente'].astype(int)
        df['cliente_ref'] = df['id_cliente'].astype(str)
    elif 'cliente_nombre_completo' in df.columns:
        df['cliente_ref'] = df['cliente_nombre_completo'].astype(str).str.strip()
        df = df[df['cliente_ref'] != '']
    else:
        raise ValueError('Falta un identificador de cliente legible en los datos.')

    if 'id_pelicula' in df.columns:
        df['id_pelicula'] = pd.to_numeric(df['id_pelicula'], errors='coerce')
        df = df.dropna(subset=['id_pelicula'])
        df['id_pelicula'] = df['id_pelicula'].astype(int)
        df['pelicula_ref'] = df['id_pelicula'].astype(str)
    elif 'pelicula_titulo' in df.columns:
        df['pelicula_ref'] = df['pelicula_titulo'].astype(str).str.strip()
        df = df[df['pelicula_ref'] != '']
    else:
        raise ValueError('Falta un identificador de pelicula legible en los datos.')

    if 'categoria_nombre' not in df.columns:
        df['categoria_nombre'] = 'Sin categoria'

    if 'duracion_alquiler' in df.columns:
        df['duracion_alquiler'] = pd.to_numeric(df['duracion_alquiler'], errors='coerce')

    df = df.dropna(subset=['cliente_ref', 'pelicula_ref', 'ingreso'])
    return df


def normalizar_serie(serie: pd.Series) -> pd.Series:
    serie = pd.to_numeric(serie, errors='coerce').fillna(0)
    minimo = float(serie.min())
    maximo = float(serie.max())
    if maximo == minimo:
        return pd.Series(np.zeros(len(serie)), index=serie.index)
    return (serie - minimo) / (maximo - minimo)


def normalizar_matriz_filas(matriz: pd.DataFrame) -> pd.DataFrame:
    suma_filas = matriz.sum(axis=1).replace(0, np.nan)
    return matriz.div(suma_filas, axis=0).fillna(0.0)


def construir_matriz_clientes_peliculas(df: pd.DataFrame) -> pd.DataFrame:
    return pd.pivot_table(
        df,
        index='cliente_ref',
        columns='pelicula_ref',
        values='ingreso',
        aggfunc='count',
        fill_value=0,
    )


def construir_df_clientes(df: pd.DataFrame) -> pd.DataFrame:
    clientes = df.groupby('cliente_ref').agg(
        total_alquileres=('pelicula_ref', 'count'),
        peliculas_unicas=('pelicula_ref', 'nunique'),
        categorias_unicas=('categoria_nombre', 'nunique'),
        gasto_total=('ingreso', 'sum'),
        gasto_promedio=('ingreso', 'mean'),
    ).reset_index()
    clientes['actividad_norm'] = normalizar_serie(clientes['total_alquileres'])
    clientes['gasto_norm'] = normalizar_serie(clientes['gasto_promedio'])
    return clientes


def resumir_peliculas(df: pd.DataFrame) -> pd.DataFrame:
    resumen = df.groupby('pelicula_ref').agg(
        pelicula_titulo=('pelicula_titulo', 'first') if 'pelicula_titulo' in df.columns else ('pelicula_ref', 'first'),
        categoria_nombre=('categoria_nombre', 'first'),
        total_alquileres=('pelicula_ref', 'count'),
        ingreso_promedio=('ingreso', 'mean'),
        ingreso_total=('ingreso', 'sum'),
        clientes_unicos=('cliente_ref', 'nunique'),
    ).reset_index()

    if 'duracion_alquiler' in df.columns:
        resumen = resumen.merge(
            df.groupby('pelicula_ref')['duracion_alquiler'].mean().rename('duracion_promedio').reset_index(),
            on='pelicula_ref',
            how='left',
        )
    else:
        resumen['duracion_promedio'] = np.nan

    resumen['popularidad_norm'] = normalizar_serie(resumen['total_alquileres'])
    resumen['ingreso_norm'] = normalizar_serie(resumen['ingreso_promedio'])
    resumen['duracion_norm'] = normalizar_serie(resumen['duracion_promedio'].fillna(resumen['duracion_promedio'].median()))
    return resumen


def construir_df_categorias(df: pd.DataFrame) -> pd.DataFrame:
    categorias = df.groupby('categoria_nombre').agg(
        total_alquileres=('pelicula_ref', 'count'),
        clientes_unicos=('cliente_ref', 'nunique'),
        ingreso_total=('ingreso', 'sum'),
        ingreso_promedio=('ingreso', 'mean'),
    ).reset_index()
    categorias['popularidad_categoria_norm'] = normalizar_serie(categorias['total_alquileres'])
    categorias['ingreso_categoria_norm'] = normalizar_serie(categorias['ingreso_promedio'])
    categorias['score_categoria_global'] = (
        categorias['popularidad_categoria_norm'] * 0.6 + categorias['ingreso_categoria_norm'] * 0.4
    )
    return categorias


def construir_df_cliente_categoria_norm(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.pivot_table(
        df,
        index='cliente_ref',
        columns='categoria_nombre',
        values='pelicula_ref',
        aggfunc='count',
        fill_value=0,
    )
    return normalizar_matriz_filas(base)


def construir_df_cliente_mes_norm(df: pd.DataFrame) -> pd.DataFrame:
    if 'tiempo_mes' not in df.columns:
        return pd.DataFrame()
    base = df.copy()
    base['tiempo_mes'] = pd.to_numeric(base['tiempo_mes'], errors='coerce')
    base = base.dropna(subset=['tiempo_mes'])
    if base.empty:
        return pd.DataFrame()
    base['tiempo_mes'] = base['tiempo_mes'].astype(int)
    pivot = pd.pivot_table(
        base,
        index='cliente_ref',
        columns='tiempo_mes',
        values='pelicula_ref',
        aggfunc='count',
        fill_value=0,
    )
    return normalizar_matriz_filas(pivot)


def construir_df_pelicula_mes_norm(df: pd.DataFrame) -> pd.DataFrame:
    if 'tiempo_mes' not in df.columns:
        return pd.DataFrame()
    base = df.copy()
    base['tiempo_mes'] = pd.to_numeric(base['tiempo_mes'], errors='coerce')
    base = base.dropna(subset=['tiempo_mes'])
    if base.empty:
        return pd.DataFrame()
    base['tiempo_mes'] = base['tiempo_mes'].astype(int)
    pivot = pd.pivot_table(
        base,
        index='pelicula_ref',
        columns='tiempo_mes',
        values='cliente_ref',
        aggfunc='count',
        fill_value=0,
    )
    return normalizar_matriz_filas(pivot)


def construir_dataframes_modelo(df: pd.DataFrame):
    dataframes_base = {
        'clientes': construir_df_clientes(df),
        'peliculas': resumir_peliculas(df),
        'categorias': construir_df_categorias(df),
        'cliente_categoria_norm': construir_df_cliente_categoria_norm(df),
        'cliente_mes_norm': construir_df_cliente_mes_norm(df),
        'pelicula_mes_norm': construir_df_pelicula_mes_norm(df),
    }

    try:
        dfs_mongo = construir_dataframes_desde_mongo(top_n=10)
    except Exception:
        dataframes_base['clusters_clientes'] = construir_clusters_clientes(dataframes_base['clientes'], n_clusters=3)
        return dataframes_base

    if isinstance(dfs_mongo, dict):
        if 'df_perfil_clientes' in dfs_mongo and not dfs_mongo['df_perfil_clientes'].empty:
            dataframes_base['clientes'] = dfs_mongo['df_perfil_clientes'].copy()
        if 'df_resumen_peliculas' in dfs_mongo and not dfs_mongo['df_resumen_peliculas'].empty:
            dataframes_base['peliculas'] = dfs_mongo['df_resumen_peliculas'].copy()
        if 'df_resumen_categorias' in dfs_mongo and not dfs_mongo['df_resumen_categorias'].empty:
            dataframes_base['categorias'] = dfs_mongo['df_resumen_categorias'].copy()

    dataframes_base['clusters_clientes'] = construir_clusters_clientes(dataframes_base['clientes'], n_clusters=3)

    return dataframes_base


def calcular_similitud_coseno(matriz: pd.DataFrame, cliente_id: str) -> pd.Series:
    if cliente_id not in matriz.index:
        raise ValueError(f'No existe historial para el cliente {cliente_id}.')

    vector_objetivo = matriz.loc[cliente_id].to_numpy(dtype=float)
    norma_objetivo = np.linalg.norm(vector_objetivo)
    if norma_objetivo == 0:
        return pd.Series(dtype=float)

    similitudes = {}
    for otro_cliente, fila in matriz.iterrows():
        if otro_cliente == cliente_id:
            continue
        vector_otro = fila.to_numpy(dtype=float)
        norma_otro = np.linalg.norm(vector_otro)
        if norma_otro == 0:
            continue
        similitud = float(np.dot(vector_objetivo, vector_otro) / (norma_objetivo * norma_otro))
        if similitud > 0:
            similitudes[str(otro_cliente)] = similitud
    return pd.Series(similitudes).sort_values(ascending=False)


def construir_preferencias_categoria(df: pd.DataFrame, cliente_id: str) -> pd.Series:
    historial = df[df['cliente_ref'] == cliente_id]
    if historial.empty:
        return pd.Series(dtype=float)
    return historial['categoria_nombre'].astype(str).value_counts(normalize=True).sort_values(ascending=False)


def construir_motivo(fila: pd.Series, categorias_preferidas: pd.Series) -> str:
    motivos = []
    if float(fila.get('score_colaborativo', 0.0)) >= 0.55:
        motivos.append('A clientes con gustos similares tambien les gusta esta pelicula')
    if float(fila.get('score_categoria', 0.0)) >= 0.40:
        motivos.append('Coincide con tus categorias preferidas')
    if float(fila.get('score_cluster_categoria', 0.0)) >= 0.45:
        motivos.append('Es frecuente en clientes de tu mismo segmento')
    if float(fila.get('score_temporal', 0.0)) >= 0.40:
        motivos.append('Encaja con tu patron de consumo por periodo')
    if float(fila.get('score_popularidad', 0.0)) >= 0.60:
        motivos.append('Tiene buena aceptacion general en el historial')

    if not motivos:
        categoria = str(fila.get('categoria_nombre', ''))
        if categoria and categoria in categorias_preferidas.index:
            return f'Recomendacion balanceada: mantiene afinidad con la categoria {categoria}'
        return 'Recomendacion balanceada por similitud, perfil y variedad'

    return motivos[0]


def construir_motivo_detallado(fila: pd.Series) -> str:
    factores = [
        ('Colaborativo (clientes similares)', 'score_colaborativo', 0.30),
        ('Afinidad por categoria', 'score_categoria', 0.16),
        ('Popularidad de la pelicula', 'score_popularidad', 0.11),
        ('Ingreso promedio de la pelicula', 'score_ingreso', 0.07),
        ('Duracion promedio', 'score_duracion', 0.05),
        ('Fuerza global de la categoria', 'score_categoria_global', 0.08),
        ('Ajuste al perfil de gasto del cliente', 'score_ajuste_cliente', 0.07),
        ('Ajuste temporal por mes', 'score_temporal', 0.04),
        ('Diversidad', 'score_diversidad', 0.02),
        ('Ajuste por centro del cluster KMeans', 'score_cluster_ajuste', 0.06),
        ('Afinidad de categoria dentro del cluster', 'score_cluster_categoria', 0.04),
    ]

    lineas = ['Calculo del score final (suma ponderada):']
    suma_aportes = 0.0
    for nombre, col, peso in factores:
        valor = float(fila.get(col, 0.0))
        aporte = valor * peso
        suma_aportes += aporte
        lineas.append(f"- {nombre}: {valor:.4f} x {peso:.2f} = {aporte:.4f}")

    score_final = float(fila.get('score_final', suma_aportes))
    lineas.append('')
    lineas.append(f'Suma de aportes: {suma_aportes:.4f}')
    lineas.append(f'Score final mostrado: {score_final:.4f}')
    return '\n'.join(lineas)


def recomendar_peliculas(
    df: pd.DataFrame,
    cliente_id: str,
    n_recomendaciones: int = 10,
    n_vecinos: int = 20,
    dataframes_modelo=None,
):
    """
    Algoritmo usado: recomendador hibrido.
    - Filtrado colaborativo basado en vecinos (similitud coseno cliente-cliente).
    - Senales basadas en contenido (categoria, popularidad, ingreso, duracion, temporalidad).
    - Ajuste de segmento con clusters KMeans de clientes.
    """
    matriz = construir_matriz_clientes_peliculas(df)
    if cliente_id not in matriz.index:
        raise ValueError(f'No existe historial para el cliente {cliente_id}.')

    if dataframes_modelo is None:
        dataframes_modelo = construir_dataframes_modelo(df)
    resumen_peliculas = dataframes_modelo['peliculas']
    df_clientes = dataframes_modelo['clientes']
    df_categorias = dataframes_modelo['categorias']
    df_cliente_categoria_norm = dataframes_modelo['cliente_categoria_norm']
    df_cliente_mes_norm = dataframes_modelo['cliente_mes_norm']
    df_pelicula_mes_norm = dataframes_modelo['pelicula_mes_norm']
    df_clusters_clientes = dataframes_modelo.get('clusters_clientes', pd.DataFrame())

    historial_cliente = df[df['cliente_ref'] == cliente_id]
    peliculas_vistas = set(historial_cliente['pelicula_ref'].astype(str).tolist())
    categorias_preferidas = construir_preferencias_categoria(df, cliente_id)

    vecinos = calcular_similitud_coseno(matriz, cliente_id).head(n_vecinos)
    candidatos = resumen_peliculas[~resumen_peliculas['pelicula_ref'].isin(peliculas_vistas)].copy()
    if candidatos.empty:
        raise ValueError(f'El cliente {cliente_id} ya vio todas las peliculas disponibles en el conjunto.')

    score_colaborativo = pd.Series(0.0, index=candidatos.index)
    if not vecinos.empty:
        matriz_vecinos = matriz.loc[vecinos.index]
        pesos = vecinos.to_numpy(dtype=float)
        denominador = float(pesos.sum())
        if denominador > 0:
            recomendacion_vecinos = matriz_vecinos.T.dot(pesos) / denominador
            score_colaborativo = candidatos['pelicula_ref'].map(recomendacion_vecinos).fillna(0.0)
    score_colaborativo = normalizar_serie(score_colaborativo)

    score_categoria = candidatos['categoria_nombre'].astype(str).map(categorias_preferidas).fillna(0.0)
    score_categoria = normalizar_serie(score_categoria)

    score_categoria_global = pd.Series(0.0, index=candidatos.index)
    if not df_categorias.empty:
        mapa_categoria_global = df_categorias.set_index('categoria_nombre')['score_categoria_global'].to_dict()
        score_categoria_global = candidatos['categoria_nombre'].astype(str).map(mapa_categoria_global).fillna(0.0)
    score_categoria_global = normalizar_serie(score_categoria_global)

    score_ajuste_cliente = pd.Series(0.0, index=candidatos.index)
    fila_cliente = df_clientes[df_clientes['cliente_ref'] == cliente_id]
    if not fila_cliente.empty:
        gasto_objetivo = float(fila_cliente.iloc[0]['gasto_promedio'])
        rango = float(resumen_peliculas['ingreso_promedio'].max() - resumen_peliculas['ingreso_promedio'].min())
        if rango > 0:
            score_ajuste_cliente = 1.0 - (candidatos['ingreso_promedio'] - gasto_objetivo).abs() / rango
            score_ajuste_cliente = score_ajuste_cliente.clip(lower=0.0, upper=1.0)
    score_ajuste_cliente = normalizar_serie(score_ajuste_cliente)

    score_temporal = pd.Series(0.0, index=candidatos.index)
    if not df_cliente_mes_norm.empty and not df_pelicula_mes_norm.empty and cliente_id in df_cliente_mes_norm.index:
        perfil_cliente = df_cliente_mes_norm.loc[cliente_id]
        columnas_comunes = [col for col in perfil_cliente.index if col in df_pelicula_mes_norm.columns]
        if columnas_comunes:
            perfil_cliente = perfil_cliente[columnas_comunes]

            def similitud_temporal(pelicula_ref: str) -> float:
                if pelicula_ref not in df_pelicula_mes_norm.index:
                    return 0.0
                perfil_pelicula = df_pelicula_mes_norm.loc[pelicula_ref, columnas_comunes]
                return float((perfil_cliente * perfil_pelicula).sum())

            score_temporal = candidatos['pelicula_ref'].astype(str).map(similitud_temporal).fillna(0.0)
    score_temporal = normalizar_serie(score_temporal)

    score_diversidad = pd.Series(0.0, index=candidatos.index)
    if not df_cliente_categoria_norm.empty and cliente_id in df_cliente_categoria_norm.index:
        perfil_categoria_cliente = df_cliente_categoria_norm.loc[cliente_id]
        score_diversidad = candidatos['categoria_nombre'].astype(str).map(
            lambda categoria: 1.0 - float(perfil_categoria_cliente.get(categoria, 0.0))
        ).fillna(0.0)
    score_diversidad = normalizar_serie(score_diversidad)

    score_cluster_ajuste = pd.Series(0.0, index=candidatos.index)
    score_cluster_categoria = pd.Series(0.0, index=candidatos.index)
    cluster_nombre_cliente = None
    if not df_clusters_clientes.empty:
        fila_cluster = df_clusters_clientes[df_clusters_clientes['cliente_ref'] == cliente_id]
        if not fila_cluster.empty:
            cluster_id_cliente = int(fila_cluster.iloc[0]['cluster_id'])
            cluster_nombre_cliente = str(fila_cluster.iloc[0].get('cluster_nombre') or '').strip() or None
            centro_cluster = float(fila_cluster.iloc[0].get('cluster_gasto_centro', 0.0))

            rango = float(resumen_peliculas['ingreso_promedio'].max() - resumen_peliculas['ingreso_promedio'].min())
            if rango > 0:
                score_cluster_ajuste = 1.0 - (candidatos['ingreso_promedio'] - centro_cluster).abs() / rango
                score_cluster_ajuste = score_cluster_ajuste.clip(lower=0.0, upper=1.0)

            mapa_cluster = df_clusters_clientes.set_index('cliente_ref')['cluster_id'].to_dict()
            base_cluster_cat = df[['cliente_ref', 'categoria_nombre']].copy()
            base_cluster_cat['cluster_id'] = base_cluster_cat['cliente_ref'].map(mapa_cluster)
            base_cluster_cat = base_cluster_cat.dropna(subset=['cluster_id'])
            base_cluster_cat['cluster_id'] = base_cluster_cat['cluster_id'].astype(int)

            preferencias_cluster = base_cluster_cat[
                base_cluster_cat['cluster_id'] == cluster_id_cliente
            ]['categoria_nombre'].astype(str).value_counts(normalize=True)
            score_cluster_categoria = candidatos['categoria_nombre'].astype(str).map(preferencias_cluster).fillna(0.0)

    score_cluster_ajuste = normalizar_serie(score_cluster_ajuste)
    score_cluster_categoria = normalizar_serie(score_cluster_categoria)

    candidatos['score_colaborativo'] = score_colaborativo.values
    candidatos['score_categoria'] = score_categoria.values
    candidatos['score_popularidad'] = candidatos['popularidad_norm'].fillna(0.0)
    candidatos['score_ingreso'] = candidatos['ingreso_norm'].fillna(0.0)
    candidatos['score_duracion'] = candidatos['duracion_norm'].fillna(0.0)
    candidatos['score_categoria_global'] = score_categoria_global.values
    candidatos['score_ajuste_cliente'] = score_ajuste_cliente.values
    candidatos['score_temporal'] = score_temporal.values
    candidatos['score_diversidad'] = score_diversidad.values
    candidatos['score_cluster_ajuste'] = score_cluster_ajuste.values
    candidatos['score_cluster_categoria'] = score_cluster_categoria.values

    candidatos['score_final'] = (
        candidatos['score_colaborativo'] * 0.30
        + candidatos['score_categoria'] * 0.16
        + candidatos['score_popularidad'] * 0.11
        + candidatos['score_ingreso'] * 0.07
        + candidatos['score_duracion'] * 0.05
        + candidatos['score_categoria_global'] * 0.08
        + candidatos['score_ajuste_cliente'] * 0.07
        + candidatos['score_temporal'] * 0.04
        + candidatos['score_diversidad'] * 0.02
        + candidatos['score_cluster_ajuste'] * 0.06
        + candidatos['score_cluster_categoria'] * 0.04
    )

    candidatos['motivo'] = candidatos.apply(lambda fila: construir_motivo(fila, categorias_preferidas), axis=1)
    candidatos['motivo_detalle'] = candidatos.apply(construir_motivo_detallado, axis=1)

    columnas_salida = [
        'pelicula_titulo', 'categoria_nombre', 'total_alquileres',
        'ingreso_promedio', 'duracion_promedio', 'score_colaborativo', 'score_categoria',
        'score_popularidad', 'score_ingreso', 'score_duracion', 'score_categoria_global',
        'score_ajuste_cliente', 'score_temporal', 'score_diversidad',
        'score_cluster_ajuste', 'score_cluster_categoria', 'score_final', 'motivo', 'motivo_detalle'
    ]
    recomendaciones = candidatos[columnas_salida].sort_values('score_final', ascending=False).head(n_recomendaciones).reset_index(drop=True)

    cliente_nombre_mostrar = None if historial_cliente.empty else str(
        historial_cliente.iloc[0].get('cliente_nombre_completo')
        or historial_cliente.iloc[0].get('cliente_nombre')
        or ''
    ).strip()
    if not cliente_nombre_mostrar:
        cliente_nombre_mostrar = str(cliente_id)

    vecino_similar_mostrar = None
    if not vecinos.empty:
        vecino_ref = str(vecinos.index[0])
        fila_vecino = df[df['cliente_ref'] == vecino_ref]
        nombre_vecino = '' if fila_vecino.empty else str(
            fila_vecino.iloc[0].get('cliente_nombre_completo')
            or fila_vecino.iloc[0].get('cliente_nombre')
            or ''
        ).strip()
        vecino_similar_mostrar = nombre_vecino if nombre_vecino else vecino_ref

    contexto = {
        'algoritmo': 'Hibrido: similitud coseno + contenido + ajuste por cluster KMeans',
        'cliente_mostrar': cliente_nombre_mostrar,
        'historico_total': int(len(historial_cliente)),
        'peliculas_unicas_vistas': int(historial_cliente['pelicula_ref'].nunique()),
        'vecinos_consultados': int(len(vecinos)),
        'vecino_mas_similar_mostrar': vecino_similar_mostrar,
        'similitud_maxima': None if vecinos.empty else float(vecinos.iloc[0]),
        'categorias_preferidas': categorias_preferidas.head(5).to_dict(),
        'segmento_cliente': cluster_nombre_cliente,
    }
    return recomendaciones, contexto


def main() -> None:
    df = preparar_datos(cargar_datos_mongo())
    dataframes_modelo = construir_dataframes_modelo(df)

    clientes = sorted(df['cliente_ref'].dropna().astype(str).unique().tolist())
    if not clientes:
        raise ValueError('No hay clientes disponibles para recomendar.')

    cliente_id = clientes[0]
    recomendaciones, contexto = recomendar_peliculas(
        df,
        cliente_id=cliente_id,
        n_recomendaciones=10,
        n_vecinos=20,
        dataframes_modelo=dataframes_modelo,
    )

    print('=== SISTEMA DE RECOMENDACION - LOGICA ===')
    print(f"Algoritmo: {contexto.get('algoritmo', 'N/A')}")
    print(f"Cliente: {contexto.get('cliente_mostrar', cliente_id)}")
    print(f"Segmento: {contexto.get('segmento_cliente', 'N/A')}")
    print(recomendaciones[['pelicula_titulo', 'categoria_nombre', 'score_final']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
