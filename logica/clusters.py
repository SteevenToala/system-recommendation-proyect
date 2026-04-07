import pandas as pd
import sys
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from logica.dataframes import construir_dataframes_desde_mongo


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


def obtener_segmentacion_clientes(top_n: int = 10):
    dfs = construir_dataframes_desde_mongo(top_n=top_n)
    df_clientes = dfs.get('df_perfil_clientes', pd.DataFrame()).copy()
    df_clusters = construir_clusters_clientes(df_clientes, n_clusters=3)

    if df_clientes.empty or df_clusters.empty:
        raise ValueError('No hay datos suficientes para segmentar clientes.')

    df = df_clientes.merge(df_clusters, on='cliente_ref', how='inner')
    if df.empty:
        raise ValueError('No se pudo relacionar el perfil de clientes con los clusters.')

    resumen = df.groupby('cluster_nombre').agg(
        clientes=('cliente_ref', 'count'),
        gasto_promedio=('gasto_promedio', 'mean'),
        actividad_promedio=('total_alquileres', 'mean'),
    ).sort_values('gasto_promedio', ascending=False)

    return df, resumen


def exportar_segmentacion(df: pd.DataFrame, salida_csv: str = 'clusters_fact_alquiler_proyecto.csv') -> str:
    columnas = ['cliente_ref', 'cluster_id', 'cluster_nombre', 'cluster_gasto_centro', 'gasto_promedio', 'total_alquileres']
    df[columnas].to_csv(salida_csv, index=False)
    return salida_csv


def main() -> None:
    df, resumen = obtener_segmentacion_clientes(top_n=10)
    print('=== SEGMENTACION DE CLIENTES - LOGICA ===')
    print(resumen.to_string())
    salida = exportar_segmentacion(df)
    print(f'Resultados guardados en: {salida}')


if __name__ == '__main__':
    main()
