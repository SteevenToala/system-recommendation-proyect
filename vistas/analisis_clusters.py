import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from logica.clusters import exportar_segmentacion, obtener_segmentacion_clientes


plt.style.use('seaborn-v0_8-whitegrid')


def _anotar_barras(ax, bars):
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y, f'{int(round(y))}', ha='center', va='bottom', fontsize=9)


def main() -> None:
    df, resumen = obtener_segmentacion_clientes(top_n=10)

    print('=== INFORME DE SEGMENTACION BI - CLIENTES ===\n')
    print(resumen.to_string())

    colores = ['#2E8B57', '#1F77B4', '#D62728', '#FF7F0E', '#17BECF']

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axs.flatten()

    bars = ax1.bar(resumen.index, resumen['clientes'], color=colores[:len(resumen)])
    _anotar_barras(ax1, bars)
    ax1.set_title('Clientes por segmento')
    ax1.set_xlabel('Segmento')
    ax1.set_ylabel('Cantidad de clientes')
    ax1.grid(axis='y', alpha=0.25)

    datos_box = [
        df[df['cluster_nombre'] == nombre]['gasto_promedio'].dropna().to_numpy()
        for nombre in resumen.index
    ]
    bp = ax2.boxplot(datos_box, labels=resumen.index, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colores[i % len(colores)])
        patch.set_alpha(0.35)
    ax2.set_title('Distribucion de gasto promedio por segmento')
    ax2.set_xlabel('Segmento')
    ax2.set_ylabel('Gasto promedio')
    ax2.grid(axis='y', alpha=0.25)

    resumen_plot = resumen[['gasto_promedio', 'actividad_promedio']].copy()
    resumen_plot.plot(kind='bar', ax=ax3, color=['#0B6E4F', '#E36414'], width=0.75)
    ax3.set_title('Gasto y actividad promedio por cluster')
    ax3.set_xlabel('Segmento')
    ax3.set_ylabel('Promedio')
    ax3.tick_params(axis='x', rotation=15)
    ax3.grid(axis='y', alpha=0.25)

    centros = df.groupby('cluster_nombre').agg(
        gasto_promedio=('gasto_promedio', 'mean'),
        total_alquileres=('total_alquileres', 'mean'),
        peliculas_unicas=('peliculas_unicas', 'mean'),
        categorias_unicas=('categorias_unicas', 'mean'),
    ).reindex(resumen.index)
    centros_norm = (centros - centros.min()) / (centros.max() - centros.min()).replace(0, 1)
    heat = ax4.imshow(centros_norm.to_numpy(), aspect='auto', cmap='YlGnBu')
    fig.colorbar(heat, ax=ax4, fraction=0.046, pad=0.04, label='Escala normalizada')
    ax4.set_yticks(range(len(centros_norm.index)))
    ax4.set_yticklabels(centros_norm.index)
    ax4.set_xticks(range(len(centros_norm.columns)))
    ax4.set_xticklabels(centros_norm.columns, rotation=20)
    ax4.set_title('Perfil relativo de cada cluster')
    for i in range(len(centros_norm.index)):
        for j in range(len(centros_norm.columns)):
            ax4.text(j, i, f"{centros_norm.iloc[i, j]:.2f}", ha='center', va='center', fontsize=8)

    plt.suptitle('Resumen visual de clusters KMeans', fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    plt.figure(figsize=(10, 6))
    for idx, nombre in enumerate(resumen.index.tolist()):
        subset = df[df['cluster_nombre'] == nombre]
        plt.scatter(
            subset['total_alquileres'],
            subset['gasto_promedio'],
            s=np.clip(subset['peliculas_unicas'] * 15, 30, 280),
            alpha=0.7,
            color=colores[idx % len(colores)],
            label=nombre,
            edgecolors='black',
            linewidths=0.3,
        )

        if not subset.empty:
            centro_x = float(subset['total_alquileres'].mean())
            centro_y = float(subset['gasto_promedio'].mean())
            plt.scatter([centro_x], [centro_y], marker='X', s=180, color=colores[idx % len(colores)], edgecolors='black')
            plt.text(centro_x, centro_y, f' Centro {nombre}', fontsize=9)

    plt.title('Dispersion KMeans: actividad vs gasto (tamano = peliculas unicas)')
    plt.xlabel('Total de alquileres')
    plt.ylabel('Gasto promedio')
    plt.legend(title='Segmento')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    salida_csv = exportar_segmentacion(df, salida_csv='clusters_fact_alquiler_proyecto.csv')
    print(f'Resultados guardados en: {salida_csv}')


if __name__ == '__main__':
    main()
