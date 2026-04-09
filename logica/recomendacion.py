"""Motor de recomendación híbrido del proyecto.

La interfaz permite elegir entre tres estrategias:
similitud de coseno, item-item y Slope One.
"""

import numpy as np
import pandas as pd
import sys
from pymongo import MongoClient
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from logica.clusters import construir_clusters_clientes
from logica.dataframes import construir_dataframes_desde_mongo, normalizar_serie


MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'BI_Final'
COLLECTION_NAME = 'fact_alquiler'


def cargar_datos_mongo() -> pd.DataFrame:
    """Carga la colección de MongoDB que alimenta los algoritmos."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    df = pd.DataFrame(list(collection.find({}, {'_id': 0})))
    if df.empty:
        raise ValueError('No hay datos cargados en MongoDB para recomendar.')
    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos y genera referencias legibles para clientes y películas."""
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


def normalizar_matriz_filas(matriz: pd.DataFrame) -> pd.DataFrame:
    """Normaliza cada fila de una matriz para que su suma sea 1."""
    suma_filas = matriz.sum(axis=1).replace(0, np.nan)
    return matriz.div(suma_filas, axis=0).fillna(0.0)


def construir_matriz_clientes_peliculas(df: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz cliente-película usada por el filtrado colaborativo."""
    return pd.pivot_table(
        df,
        index='cliente_ref',
        columns='pelicula_ref',
        values='ingreso',
        aggfunc='count',
        fill_value=0,
    )


def construir_matriz_valoraciones(df: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz de valoraciones medias cliente-película."""
    return pd.pivot_table(
        df,
        index='cliente_ref',
        columns='pelicula_ref',
        values='ingreso',
        aggfunc='mean',
    )


def construir_df_clientes(df: pd.DataFrame) -> pd.DataFrame:
    """Resume el comportamiento de cada cliente en indicadores entendibles."""
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
    """Resume cada película con métricas útiles para explicar la recomendación."""
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
    """Calcula el perfil global de cada categoría."""
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
    """Construye la distribución de categorías por cliente."""
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
    """Construye el perfil temporal de consumo por cliente."""
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
    """Construye el perfil temporal de consumo por película."""
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
    """Agrupa los DataFrames auxiliares que usan los tres algoritmos."""
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
    """Calcula la similitud coseno entre un cliente y el resto."""
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
    """Obtiene la distribución de categorías vistas por un cliente."""
    historial = df[df['cliente_ref'] == cliente_id]
    if historial.empty:
        return pd.Series(dtype=float)
    return historial['categoria_nombre'].astype(str).value_counts(normalize=True).sort_values(ascending=False)


def construir_motivo(fila: pd.Series, categorias_preferidas: pd.Series = None) -> str:
    """Genera una explicación corta y fácil de leer para cada recomendación."""
    score = float(fila.get('score_final', 0.0))
    categoria = str(fila.get('categoria_nombre', '')).strip()

    if score >= 0.80:
        base = 'Muy alineada con tu perfil'
    elif score >= 0.60:
        base = 'Buena coincidencia con tus gustos'
    elif score >= 0.40:
        base = 'Coincidencia moderada con tu perfil'
    else:
        base = 'Afinidad parcial con tu perfil'

    if categoria and categorias_preferidas is not None and categoria in categorias_preferidas.index:
        return f'{base} en {categoria}'
    if categoria:
        return f'{base} para {categoria}'
    return base


def construir_motivo_detallado(fila: pd.Series, lineas_extra=None) -> str:
    """Explica el cálculo del score final con sus ponderaciones."""
    lineas = [
        f"Pelicula: {fila.get('pelicula_titulo', '')}",
        f"Categoria: {fila.get('categoria_nombre', '')}",
        f"Score final: {float(fila.get('score_final', 0.0)):.4f}",
        f"Total de alquileres: {int(float(fila.get('total_alquileres', 0.0)))}",
        f"Ingreso promedio: {float(fila.get('ingreso_promedio', 0.0)):.2f}",
    ]

    duracion = fila.get('duracion_promedio', np.nan)
    if pd.notna(duracion):
        lineas.append(f"Duracion promedio: {float(duracion):.2f}")

    if lineas_extra:
        lineas.append('')
        lineas.extend([str(linea) for linea in lineas_extra if str(linea).strip()])

    return '\n'.join(lineas)


def _base_candidatos(
    df: pd.DataFrame,
    cliente_id: str,
    dataframes_modelo: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    resumen_peliculas = dataframes_modelo['peliculas']
    historial_cliente = df[df['cliente_ref'] == cliente_id]
    peliculas_vistas = set(historial_cliente['pelicula_ref'].astype(str).tolist())
    categorias_preferidas = construir_preferencias_categoria(df, cliente_id)

    candidatos = resumen_peliculas[~resumen_peliculas['pelicula_ref'].isin(peliculas_vistas)].copy()
    if candidatos.empty:
        raise ValueError(f'El cliente {cliente_id} ya vio todas las peliculas disponibles en el conjunto.')

    return candidatos, categorias_preferidas, historial_cliente, resumen_peliculas


def _armar_salida_basica(
    candidatos: pd.DataFrame,
    score: pd.Series,
    motivo_texto: str,
    n_recomendaciones: int,
    motivo_builder=None,
    detalle_por_pelicula=None,
) -> pd.DataFrame:
    candidatos = candidatos.copy()
    candidatos['score_final'] = normalizar_serie(score).fillna(0.0)

    if motivo_builder is not None:
        candidatos['motivo'] = candidatos.apply(motivo_builder, axis=1)
    else:
        candidatos['motivo'] = motivo_texto

    if detalle_por_pelicula is not None:
        referencias = candidatos['pelicula_ref'].astype(str)
        candidatos['motivo_detalle'] = referencias.map(detalle_por_pelicula).fillna(candidatos['motivo'])
    else:
        candidatos['motivo_detalle'] = candidatos['motivo']

    columnas_base = ['pelicula_titulo', 'categoria_nombre', 'total_alquileres', 'ingreso_promedio', 'duracion_promedio']
    for col in columnas_base:
        if col not in candidatos.columns:
            candidatos[col] = 0.0 if col != 'pelicula_titulo' and col != 'categoria_nombre' else ''

    columnas_salida = columnas_base + ['score_final', 'motivo', 'motivo_detalle']
    return candidatos[columnas_salida].sort_values('score_final', ascending=False).head(n_recomendaciones).reset_index(drop=True)


def _recomendar_coseno_usuario(
    df: pd.DataFrame,
    cliente_id: str,
    n_recomendaciones: int,
    n_vecinos: int,
    dataframes_modelo: dict,
):
    matriz_val = construir_matriz_valoraciones(df)
    if cliente_id not in matriz_val.index:
        raise ValueError(f'No existe historial para el cliente {cliente_id}.')

    candidatos, categorias_preferidas, historial_cliente, _ = _base_candidatos(df, cliente_id, dataframes_modelo)
    matriz_fill = matriz_val.fillna(0.0)
    vecinos = calcular_similitud_coseno(matriz_fill, cliente_id).head(n_vecinos)
    vecinos_positivos = vecinos[vecinos > 0]
    similitud_media_vecinos = float(vecinos_positivos.mean()) if not vecinos_positivos.empty else 0.0

    score = pd.Series(0.0, index=candidatos.index)
    datos_para_detalle = {}
    
    if not vecinos.empty:
        denominador = float(np.abs(vecinos.values).sum())
        if denominador > 0:
            for idx, pelicula_ref in candidatos['pelicula_ref'].astype(str).items():
                total = 0.0
                peso_total = 0.0
                valoraciones_vecinos = []
                vecinos_usados = 0
                for vecino_id, sim in vecinos.items():
                    rating = float(matriz_val.loc[vecino_id, pelicula_ref]) if pelicula_ref in matriz_val.columns else np.nan
                    if pd.notna(rating) and float(sim) > 0:
                        total += float(sim) * rating
                        peso_total += abs(float(sim))
                        valoraciones_vecinos.append(float(rating))
                        vecinos_usados += 1
                score.loc[idx] = (total / peso_total) if peso_total > 0 else 0.0
                datos_para_detalle[pelicula_ref] = (vecinos_usados, valoraciones_vecinos)

    score_normalizado = normalizar_serie(score).fillna(0.0)
    
    detalle_por_pelicula = {}
    
    for idx, pelicula_ref in candidatos['pelicula_ref'].astype(str).items():
        fila = candidatos.loc[idx].copy()
        fila['score_final'] = float(score_normalizado.loc[idx])
        if pelicula_ref in datos_para_detalle:
            vecinos_usados, valoraciones_vecinos = datos_para_detalle[pelicula_ref]
            lineas_extra = [
                'Algoritmo: similitud entre clientes con patrones de consumo parecidos.',
                f'Clientes parecidos que aportaron al score: {vecinos_usados}',
                f'Similitud media de los clientes usados: {similitud_media_vecinos:.3f}',
                (
                    f'Valoracion media observada en esos clientes: {float(np.mean(valoraciones_vecinos)):.2f}'
                    if valoraciones_vecinos
                    else 'Valoracion media observada en esos clientes: 0.00'
                ),
                'La pelicula sube de posicion cuando clientes similares la valoran bien.',
            ]
            detalle_por_pelicula[pelicula_ref] = construir_motivo_detallado(fila, lineas_extra)

    recomendaciones = _armar_salida_basica(
        candidatos,
        score_normalizado,
        motivo_texto='Coincide con tu perfil de consumo',
        n_recomendaciones=n_recomendaciones,
        motivo_builder=construir_motivo,
        detalle_por_pelicula=detalle_por_pelicula,
    )

    contexto = {
        'algoritmo': 'Similitud de coseno (usuario-usuario)',
        'historico_total': int(len(historial_cliente)),
        'peliculas_unicas_vistas': int(historial_cliente['pelicula_ref'].nunique()),
        'vecinos_consultados': int(len(vecinos)),
        'categorias_preferidas': categorias_preferidas.head(5).to_dict(),
    }
    return recomendaciones, contexto


def _recomendar_item_item(
    df: pd.DataFrame,
    cliente_id: str,
    n_recomendaciones: int,
    n_vecinos: int,
    dataframes_modelo: dict,
):
    matriz_val = construir_matriz_valoraciones(df)
    if cliente_id not in matriz_val.index:
        raise ValueError(f'No existe historial para el cliente {cliente_id}.')

    candidatos, categorias_preferidas, historial_cliente, _ = _base_candidatos(df, cliente_id, dataframes_modelo)
    matriz_fill = matriz_val.fillna(0.0)

    usuario_ratings = matriz_val.loc[cliente_id]
    items_vistos = usuario_ratings[usuario_ratings.notna()].index.astype(str).tolist()
    if not items_vistos:
        raise ValueError('El cliente no tiene historial suficiente para item-item.')

    titulo_por_item = {}
    if 'pelicula_titulo' in historial_cliente.columns:
        titulo_por_item = (
            historial_cliente.dropna(subset=['pelicula_ref'])
            .drop_duplicates('pelicula_ref')
            .set_index('pelicula_ref')['pelicula_titulo']
            .astype(str)
            .to_dict()
        )

    matriz_items = matriz_fill.T
    items_disponibles = set(matriz_items.index.astype(str).tolist())
    items_vistos = [it for it in items_vistos if it in items_disponibles]
    candidatos_refs = candidatos['pelicula_ref'].astype(str).tolist()
    candidatos_refs = [it for it in candidatos_refs if it in items_disponibles]

    if not candidatos_refs:
        raise ValueError('No hay peliculas candidatas con informacion suficiente para item-item.')

    seen_mat = matriz_items.loc[items_vistos].to_numpy(dtype=float)
    cand_mat = matriz_items.loc[candidatos_refs].to_numpy(dtype=float)
    seen_norms = np.linalg.norm(seen_mat, axis=1)
    cand_norms = np.linalg.norm(cand_mat, axis=1)
    denom = np.outer(cand_norms, seen_norms)
    sim_matrix = cand_mat @ seen_mat.T
    with np.errstate(divide='ignore', invalid='ignore'):
        sim_matrix = np.divide(sim_matrix, denom, out=np.zeros_like(sim_matrix), where=denom > 0)

    user_seen_ratings = usuario_ratings.loc[items_vistos].to_numpy(dtype=float)
    k = max(1, min(int(n_vecinos), sim_matrix.shape[1]))
    score_por_candidato = {}
    detalle_por_pelicula = {}
    for row_idx, item_candidato in enumerate(candidatos_refs):
        sims = sim_matrix[row_idx]
        if sims.size == 0:
            score_por_candidato[item_candidato] = 0.0
            continue
        top_idx = np.argpartition(sims, -k)[-k:]
        top_sims = sims[top_idx]
        top_ratings = user_seen_ratings[top_idx]
        mask = top_sims > 0
        if not np.any(mask):
            score_por_candidato[item_candidato] = 0.0
            continue
        top_idx = top_idx[mask]
        top_sims = top_sims[mask]
        top_ratings = top_ratings[mask]
        numerador = float(np.dot(top_sims, top_ratings))
        denominador = float(np.abs(top_sims).sum())
        score_por_candidato[item_candidato] = (numerador / denominador) if denominador > 0 else 0.0

    score = candidatos['pelicula_ref'].astype(str).map(score_por_candidato).fillna(0.0)
    score_normalizado = normalizar_serie(score).fillna(0.0)
    
    detalle_por_pelicula = {}
    for row_idx, item_candidato in enumerate(candidatos_refs):
        fila = candidatos.iloc[row_idx].copy()
        fila['score_final'] = float(score_normalizado.iloc[row_idx])
        
        sims = sim_matrix[row_idx]
        if sims.size > 0:
            top_idx_local = np.argpartition(sims, -k)[-k:]
            top_sims = sims[top_idx_local]
            top_ratings = user_seen_ratings[top_idx_local]
            mask = top_sims > 0
            if np.any(mask):
                top_idx_local = top_idx_local[mask]
                top_sims = top_sims[mask]
                top_ratings = top_ratings[mask]
                
                referencias_apoyo = sorted(
                    zip(top_idx_local.tolist(), top_sims.tolist(), top_ratings.tolist()),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
                lineas_extra = [
                    'Algoritmo: similitud entre peliculas ya vistas y peliculas candidatas.',
                    f'Peliculas de referencia usadas: {len(referencias_apoyo)}',
                ]
                for idx_ref, sim, rating in referencias_apoyo:
                    ref = items_vistos[int(idx_ref)]
                    titulo_ref = titulo_por_item.get(ref, ref)
                    lineas_extra.append(f'- {titulo_ref}: similitud {float(sim):.3f}, valoracion {float(rating):.2f}')
                lineas_extra.append('La pelicula gana peso cuando se parece a varias de tus vistas mejor valoradas.')
                detalle_por_pelicula[item_candidato] = construir_motivo_detallado(fila, lineas_extra)

    recomendaciones = _armar_salida_basica(
        candidatos,
        score_normalizado,
        motivo_texto='Se parece a peliculas que ya consumiste',
        n_recomendaciones=n_recomendaciones,
        motivo_builder=construir_motivo,
        detalle_por_pelicula=detalle_por_pelicula,
    )

    contexto = {
        'algoritmo': 'Item-item collaborative filtering',
        'historico_total': int(len(historial_cliente)),
        'peliculas_unicas_vistas': int(historial_cliente['pelicula_ref'].nunique()),
        'vecinos_consultados': int(max(1, n_vecinos)),
        'categorias_preferidas': categorias_preferidas.head(5).to_dict(),
    }
    return recomendaciones, contexto


def _recomendar_slope_one(
    df: pd.DataFrame,
    cliente_id: str,
    n_recomendaciones: int,
    _n_vecinos: int,
    dataframes_modelo: dict,
):
    matriz_val = construir_matriz_valoraciones(df)
    if cliente_id not in matriz_val.index:
        raise ValueError(f'No existe historial para el cliente {cliente_id}.')

    candidatos, categorias_preferidas, historial_cliente, _ = _base_candidatos(df, cliente_id, dataframes_modelo)
    usuario_ratings = matriz_val.loc[cliente_id]
    rated_items = usuario_ratings[usuario_ratings.notna()]
    if rated_items.empty:
        raise ValueError('El cliente no tiene historial suficiente para Slope One.')

    titulo_por_item = {}
    if 'pelicula_titulo' in historial_cliente.columns:
        titulo_por_item = (
            historial_cliente.dropna(subset=['pelicula_ref'])
            .drop_duplicates('pelicula_ref')
            .set_index('pelicula_ref')['pelicula_titulo']
            .astype(str)
            .to_dict()
        )

    dev = {}
    freq = {}
    for _, fila in matriz_val.iterrows():
        fila_rated = fila.dropna()
        items = fila_rated.index.tolist()
        for item_i in items:
            dev.setdefault(item_i, {})
            freq.setdefault(item_i, {})
            for item_j in items:
                if item_i == item_j:
                    continue
                dev[item_i].setdefault(item_j, 0.0)
                freq[item_i].setdefault(item_j, 0)
                dev[item_i][item_j] += float(fila_rated[item_i] - fila_rated[item_j])
                freq[item_i][item_j] += 1

    for item_i in dev:
        for item_j in dev[item_i]:
            if freq[item_i][item_j] > 0:
                dev[item_i][item_j] /= float(freq[item_i][item_j])

    score = pd.Series(0.0, index=candidatos.index)
    datos_para_detalle = {}
    
    for idx, item_candidato in candidatos['pelicula_ref'].astype(str).items():
        numerador = 0.0
        denominador = 0.0
        evidencias = []
        if item_candidato not in dev:
            continue
        for item_visto, rating_visto in rated_items.items():
            if item_visto == item_candidato:
                continue
            if item_visto in dev[item_candidato] and item_visto in freq[item_candidato]:
                f = float(freq[item_candidato][item_visto])
                desviacion = float(dev[item_candidato][item_visto])
                estimado = desviacion + float(rating_visto)
                numerador += estimado * f
                denominador += f
                evidencias.append((item_visto, float(rating_visto), desviacion, f, estimado))
        score.loc[idx] = (numerador / denominador) if denominador > 0 else 0.0
        datos_para_detalle[item_candidato] = evidencias

    score_normalizado = normalizar_serie(score).fillna(0.0)
    
    detalle_por_pelicula = {}
    
    for idx, item_candidato in candidatos['pelicula_ref'].astype(str).items():
        fila = candidatos.loc[idx].copy()
        fila['score_final'] = float(score_normalizado.loc[idx])
        
        if item_candidato in datos_para_detalle and len(datos_para_detalle[item_candidato]) > 0:
            evidencias = datos_para_detalle[item_candidato]
            evidencias_ordenadas = sorted(evidencias, key=lambda x: x[3], reverse=True)[:3]
            lineas_extra = [
                'Algoritmo: desviaciones promedio entre peliculas valoradas.',
                f'Pares comparados con tu historial: {len(evidencias)}',
            ]
            for item_visto, rating_visto, desviacion, frecuencia, estimado in evidencias_ordenadas:
                titulo_ref = titulo_por_item.get(item_visto, item_visto)
                lineas_extra.append(
                    f'- {titulo_ref}: tu valoracion {rating_visto:.2f}, desviacion media {desviacion:+.3f}, frecuencia {int(frecuencia)}'
                )
            lineas_extra.append('La prediccion mejora cuando varios pares historicos apuntan a una valoracion similar.')
            detalle_por_pelicula[item_candidato] = construir_motivo_detallado(fila, lineas_extra)
        else:
            continue

    recomendaciones = _armar_salida_basica(
        candidatos,
        score_normalizado,
        motivo_texto='Encaja con tu historial de valoraciones',
        n_recomendaciones=n_recomendaciones,
        motivo_builder=construir_motivo,
        detalle_por_pelicula=detalle_por_pelicula,
    )

    contexto = {
        'algoritmo': 'Slope One (item-based deviations)',
        'historico_total': int(len(historial_cliente)),
        'peliculas_unicas_vistas': int(historial_cliente['pelicula_ref'].nunique()),
        'vecinos_consultados': 0,
        'categorias_preferidas': categorias_preferidas.head(5).to_dict(),
    }
    return recomendaciones, contexto


def recomendar_peliculas(
    df: pd.DataFrame,
    cliente_id: str,
    n_recomendaciones: int = 10,
    n_vecinos: int = 20,
    dataframes_modelo=None,
    algoritmo: str = 'coseno',
):
    """
    Selector de algoritmos de recomendacion.
    - Similitud de coseno entre clientes.
    - Slope One basado en desviaciones entre pares de peliculas.
    - Item-item basado en similitud entre peliculas vistas y candidatas.
    """
    if dataframes_modelo is None:
        dataframes_modelo = construir_dataframes_modelo(df)
    algoritmo_key = str(algoritmo or 'coseno').strip().lower()

    if algoritmo_key in {'coseno', 'similitud_coseno', 'cosine'}:
        recomendaciones, contexto = _recomendar_coseno_usuario(
            df,
            cliente_id=cliente_id,
            n_recomendaciones=n_recomendaciones,
            n_vecinos=n_vecinos,
            dataframes_modelo=dataframes_modelo,
        )
    elif algoritmo_key in {'slope_one', 'slopeone', 'slope'}:
        recomendaciones, contexto = _recomendar_slope_one(
            df,
            cliente_id=cliente_id,
            n_recomendaciones=n_recomendaciones,
            _n_vecinos=n_vecinos,
            dataframes_modelo=dataframes_modelo,
        )
    elif algoritmo_key in {'item_item', 'item-item', 'item'}:
        recomendaciones, contexto = _recomendar_item_item(
            df,
            cliente_id=cliente_id,
            n_recomendaciones=n_recomendaciones,
            n_vecinos=n_vecinos,
            dataframes_modelo=dataframes_modelo,
        )
    else:
        raise ValueError('Algoritmo no valido. Usa: coseno, slope_one o item_item.')

    historial_cliente = df[df['cliente_ref'] == cliente_id]
    cliente_nombre_mostrar = None if historial_cliente.empty else str(
        historial_cliente.iloc[0].get('cliente_nombre_completo')
        or historial_cliente.iloc[0].get('cliente_nombre')
        or ''
    ).strip()
    if not cliente_nombre_mostrar:
        cliente_nombre_mostrar = str(cliente_id)

    contexto['cliente_mostrar'] = cliente_nombre_mostrar
    contexto.setdefault('segmento_cliente', None)
    contexto.setdefault('vecino_mas_similar_mostrar', None)
    contexto.setdefault('similitud_maxima', None)

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
        algoritmo='coseno',
    )

    print('=== SISTEMA DE RECOMENDACION - LOGICA ===')
    print(f"Algoritmo: {contexto.get('algoritmo', 'N/A')}")
    print(f"Cliente: {contexto.get('cliente_mostrar', cliente_id)}")
    print(f"Segmento: {contexto.get('segmento_cliente', 'N/A')}")
    print(recomendaciones[['pelicula_titulo', 'categoria_nombre', 'score_final']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
