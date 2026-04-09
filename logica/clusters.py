"""Segmentación de clientes mediante KMeans para el sistema de recomendación."""

import pandas as pd
import sys
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from logica.dataframes import construir_dataframes_desde_mongo


COLUMNAS_MODELO = ['gasto_promedio', 'total_alquileres', 'peliculas_unicas', 'categorias_unicas']
COLUMNAS_SALIDA = ['cliente_ref', 'cluster_id', 'cluster_nombre', 'cluster_gasto_centro']
NOMBRES_CLUSTERS = ['Mayor gasto', 'Gasto medio', 'Menor gasto']


def _df_clusters_vacio() -> pd.DataFrame:
    """Devuelve la estructura vacía esperada por el resto del sistema."""
    return pd.DataFrame(columns=COLUMNAS_SALIDA)


def _validar_dataframe_clientes(df_perfil_clientes: pd.DataFrame) -> pd.DataFrame:
    """Prepara el DataFrame de clientes para clustering."""
    if df_perfil_clientes.empty:
        return _df_clusters_vacio()

    base = df_perfil_clientes.copy()
    columnas_modelo = [col for col in COLUMNAS_MODELO if col in base.columns]
    if not columnas_modelo:
        return _df_clusters_vacio()

    for columna in columnas_modelo:
        base[columna] = pd.to_numeric(base[columna], errors='coerce')

    base = base.dropna(subset=columnas_modelo)
    if base.empty:
        return _df_clusters_vacio()

    return base


def _asignar_nombres_clusters(base: pd.DataFrame) -> pd.DataFrame:
    """Asigna nombres legibles a los clusters ordenados por gasto medio."""
    centros_gasto = base.groupby('cluster_id')['gasto_promedio'].mean().sort_values(ascending=False)
    mapa_nombres = {
        cluster_id: NOMBRES_CLUSTERS[idx] if idx < len(NOMBRES_CLUSTERS) else f'Cluster {idx + 1}'
        for idx, cluster_id in enumerate(centros_gasto.index.tolist())
    }

    base = base.copy()
    base['cluster_nombre'] = base['cluster_id'].map(mapa_nombres)
    base['cluster_gasto_centro'] = base['cluster_id'].map(centros_gasto.to_dict()).astype(float)
    return base


def construir_clusters_clientes(df_perfil_clientes: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Agrupa clientes por comportamiento de consumo usando KMeans."""
    base = _validar_dataframe_clientes(df_perfil_clientes)
    if base.empty:
        return _df_clusters_vacio()

    columnas_modelo = [col for col in COLUMNAS_MODELO if col in base.columns]
    k = max(1, min(int(n_clusters), len(base)))
    scaler = StandardScaler()
    matriz_escalada = scaler.fit_transform(base[columnas_modelo].fillna(0.0))

    if k == 1:
        salida = base.copy()
        salida['cluster_id'] = 0
        salida['cluster_nombre'] = 'Perfil unico'
        salida['cluster_gasto_centro'] = float(salida['gasto_promedio'].mean()) if 'gasto_promedio' in salida.columns else 0.0
        return salida[COLUMNAS_SALIDA]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    base['cluster_id'] = kmeans.fit_predict(matriz_escalada)
    return _asignar_nombres_clusters(base)[COLUMNAS_SALIDA]


def obtener_segmentacion_clientes(top_n: int = 10):
    """Obtiene clientes segmentados y un resumen por cluster."""
    dfs = construir_dataframes_desde_mongo(top_n=top_n)
    df_clientes = dfs.get('df_perfil_clientes', pd.DataFrame()).copy()
    df_clusters = construir_clusters_clientes(df_clientes, n_clusters=3)

    if df_clientes.empty or df_clusters.empty:
        raise ValueError('No hay datos suficientes para segmentar clientes.')

    df_segmentado = df_clientes.merge(df_clusters, on='cliente_ref', how='inner')
    if df_segmentado.empty:
        raise ValueError('No se pudo relacionar el perfil de clientes con los clusters.')

    resumen = df_segmentado.groupby('cluster_nombre').agg(
        clientes=('cliente_ref', 'count'),
        gasto_promedio=('gasto_promedio', 'mean'),
        actividad_promedio=('total_alquileres', 'mean'),
    ).sort_values('gasto_promedio', ascending=False)

    return df_segmentado, resumen


def exportar_segmentacion(df: pd.DataFrame, salida_csv: str = 'clusters_fact_alquiler_proyecto.csv') -> str:
    """Exporta el segmento de clientes a CSV con las columnas principales."""
    columnas = ['cliente_ref', 'cluster_id', 'cluster_nombre', 'cluster_gasto_centro', 'gasto_promedio', 'total_alquileres']
    df[columnas].to_csv(salida_csv, index=False)
    return salida_csv


def main() -> None:
    """Muestra el resumen de segmentos y exporta el CSV final."""
    df, resumen = obtener_segmentacion_clientes(top_n=10)
    print('=== SEGMENTACION DE CLIENTES - LOGICA ===')
    print(resumen.to_string())
    salida = exportar_segmentacion(df)
    print(f'Resultados guardados en: {salida}')


if __name__ == '__main__':
    main()
