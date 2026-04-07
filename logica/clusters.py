import pandas as pd
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from logica.dataframes import construir_dataframes_desde_mongo


def obtener_segmentacion_clientes(top_n: int = 10):
    dfs = construir_dataframes_desde_mongo(top_n=top_n)
    df_clientes = dfs.get('df_perfil_clientes', pd.DataFrame()).copy()
    df_clusters = dfs.get('df_clusters_clientes', pd.DataFrame()).copy()

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
