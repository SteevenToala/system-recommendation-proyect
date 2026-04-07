import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from logica.clusters import exportar_segmentacion, obtener_segmentacion_clientes


plt.style.use('seaborn-v0_8-whitegrid')


def main() -> None:
    df, resumen = obtener_segmentacion_clientes(top_n=10)

    print('=== INFORME DE SEGMENTACION BI - CLIENTES ===\n')
    print(resumen.to_string())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(resumen.index, resumen['clientes'], color=['#2E8B57', '#1F77B4', '#D62728'][:len(resumen)])
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y, f'{int(y)}', ha='center', va='bottom')
    plt.title('Clientes por segmento')
    plt.xlabel('Segmento')
    plt.ylabel('Cantidad de clientes')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    datos_box = [
        df[df['cluster_nombre'] == nombre]['gasto_promedio'].dropna().to_numpy()
        for nombre in resumen.index
    ]
    plt.boxplot(datos_box, labels=resumen.index)
    plt.title('Distribucion de gasto promedio por segmento')
    plt.xlabel('Segmento')
    plt.ylabel('Gasto promedio del cliente')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.show()

    salida_csv = exportar_segmentacion(df, salida_csv='clusters_fact_alquiler_proyecto.csv')
    print(f'Resultados guardados en: {salida_csv}')


if __name__ == '__main__':
    main()
