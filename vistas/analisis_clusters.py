"""Visualización del análisis de clusters de clientes."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from logica.clusters import exportar_segmentacion, obtener_segmentacion_clientes


plt.style.use('seaborn-v0_8-whitegrid')


def _normalizar_dataframe(df):
    """Escala un DataFrame al rango 0-1 por columna."""
    return (df - df.min()) / (df.max() - df.min()).replace(0, 1)


def _preparar_colores(resumen):
    """Asigna un color estable a cada cluster."""
    paleta = ['#2E8B57', '#1F77B4', '#D62728', '#FF7F0E', '#17BECF', '#9467BD']
    return {nombre: paleta[idx % len(paleta)] for idx, nombre in enumerate(resumen.index.tolist())}


def _dimensionar_y_centrar_figura(fig, ancho_px=820, alto_px=520):
    """Ajusta el tamaño de la figura y la centra si el backend lo permite."""
    try:
        manager = plt.get_current_fig_manager()
        ventana = getattr(manager, 'window', None)
        dpi = fig.dpi or 100
        fig.set_size_inches(ancho_px / dpi, alto_px / dpi, forward=True)

        if ventana is not None and hasattr(ventana, 'winfo_screenwidth') and hasattr(ventana, 'winfo_screenheight'):
            if hasattr(ventana, 'update_idletasks'):
                ventana.update_idletasks()
            screen_w = int(ventana.winfo_screenwidth())
            screen_h = int(ventana.winfo_screenheight())
            x = max(0, (screen_w - int(ancho_px)) // 2)
            y = max(0, (screen_h - int(alto_px)) // 2)
            if hasattr(ventana, 'geometry'):
                ventana.geometry(f'{int(ancho_px)}x{int(alto_px)}+{x}+{y}')
    except Exception:
        pass


def _mostrar_figura_ajustada(fig, ancho_px, alto_px):
    """Muestra la figura fijando su tamaño después del primer render."""
    _dimensionar_y_centrar_figura(fig, ancho_px=ancho_px, alto_px=alto_px)

    # Algunos backends reajustan la ventana tras el primer render; volvemos a fijar tamano.
    cid_holder = {'cid': None}

    def _fijar_en_primer_dibujo(_event):
        _dimensionar_y_centrar_figura(fig, ancho_px=ancho_px, alto_px=alto_px)
        cid = cid_holder.get('cid')
        if cid is not None:
            fig.canvas.mpl_disconnect(cid)

    cid_holder['cid'] = fig.canvas.mpl_connect('draw_event', _fijar_en_primer_dibujo)

    try:
        fig.canvas.draw_idle()
    except Exception:
        pass
    plt.show()


def _agregar_descripcion(ax, texto):
    """Coloca una breve explicación dentro del gráfico."""
    ax.text(
        0.01,
        0.01,
        texto,
        transform=ax.transAxes,
        fontsize=8,
        va='bottom',
        ha='left',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#F8F9FA', edgecolor='#BBBBBB'),
    )


def main() -> None:
    """Genera el reporte de clusters y abre las cuatro visualizaciones."""
    df, resumen = obtener_segmentacion_clientes(top_n=10)

    print('=== INFORME DE SEGMENTACION BI - CLIENTES ===\n')
    print(resumen.to_string())

    colores = _preparar_colores(resumen)

    base = df[['total_alquileres', 'gasto_promedio', 'peliculas_unicas', 'categorias_unicas']].copy().fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(base)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df_plot = df.copy()
    df_plot['pca_1'] = X_pca[:, 0]
    df_plot['pca_2'] = X_pca[:, 1]

    perfil_cluster = df.groupby('cluster_nombre').agg(
        clientes=('cliente_ref', 'count'),
        gasto_promedio=('gasto_promedio', 'mean'),
        actividad_promedio=('total_alquileres', 'mean'),
        peliculas_unicas=('peliculas_unicas', 'mean'),
        categorias_unicas=('categorias_unicas', 'mean'),
    ).reindex(resumen.index)

    perfil_normalizado = _normalizar_dataframe(
        perfil_cluster[['gasto_promedio', 'actividad_promedio', 'peliculas_unicas', 'categorias_unicas']]
    )
    resumen_normalizado = _normalizar_dataframe(resumen[['clientes', 'gasto_promedio', 'actividad_promedio']])

    perfil_normalizado = perfil_normalizado.rename(
        columns={
            'gasto_promedio': 'Nivel de gasto',
            'actividad_promedio': 'Nivel de actividad',
            'peliculas_unicas': 'Variedad de peliculas',
            'categorias_unicas': 'Variedad de categorias',
        }
    )
    resumen_normalizado = resumen_normalizado.rename(
        columns={
            'clientes': 'Cantidad de clientes',
            'gasto_promedio': 'Gasto promedio',
            'actividad_promedio': 'Actividad promedio',
        }
    )

    # Ventana 1: tamano de segmentos
    fig1, ax1 = plt.subplots(figsize=(8.4, 5.2))
    bars = ax1.bar(resumen.index, resumen['clientes'], color=[colores[n] for n in resumen.index])
    for bar in bars:
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, y, f'{int(round(y))}', ha='center', va='bottom', fontsize=8)
    ax1.set_title('Cantidad de clientes por segmento')
    ax1.set_xlabel('Segmento')
    ax1.set_ylabel('Cantidad de clientes')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(axis='y', alpha=0.25)
    _agregar_descripcion(
        ax1,
        'Este grafico muestra cuantos clientes hay en cada segmento.',
    )
    _mostrar_figura_ajustada(fig1, ancho_px=800, alto_px=500)

    # Ventana 2: mapa visual de similitud entre clientes
    fig2, ax2 = plt.subplots(figsize=(8.6, 5.3))
    for nombre in resumen.index:
        subset = df_plot[df_plot['cluster_nombre'] == nombre]
        ax2.scatter(
            subset['pca_1'],
            subset['pca_2'],
            s=np.clip(subset['gasto_promedio'] * 2.2, 16, 110),
            alpha=0.76,
            color=colores[nombre],
            label=nombre,
            edgecolors='white',
            linewidths=0.4,
        )
        if not subset.empty:
            centro_x = float(subset['pca_1'].mean())
            centro_y = float(subset['pca_2'].mean())
            ax2.scatter([centro_x], [centro_y], marker='X', s=160, color=colores[nombre], edgecolors='black')
            ax2.text(centro_x, centro_y, f' {nombre}', fontsize=8, weight='bold')

    ax2.set_title('Mapa de similitud de comportamiento de clientes')
    ax2.set_xlabel('Nivel total de consumo del cliente')
    ax2.set_ylabel('Estilo de consumo del cliente')
    ax2.grid(alpha=0.2)
    ax2.legend(title='Segmentos', fontsize=8, title_fontsize=9)
    _agregar_descripcion(
        ax2,
        'Cada punto representa un cliente.\nNivel total de consumo (eje X) = intensidad general (mas/menos consumo).\nEstilo de consumo (eje Y) = forma de consumir (mas diverso, mas concentrado, mas activo con menor gasto, etc.).\nSi dos puntos estan cerca, esos clientes se comportan de forma parecida.',
    )
    _mostrar_figura_ajustada(fig2, ancho_px=820, alto_px=520)

    # Ventana 3: perfil normalizado
    fig3, ax3 = plt.subplots(figsize=(8.6, 5.3))
    perfil_normalizado.plot(kind='bar', ax=ax3, color=['#0B6E4F', '#E36414', '#6A4C93', '#2A9D8F'], width=0.78)
    ax3.set_title('Comparacion de perfil por segmento')
    ax3.set_xlabel('Segmento')
    ax3.set_ylabel('Nivel relativo (0 a 1)')
    ax3.tick_params(axis='x', rotation=15)
    ax3.grid(axis='y', alpha=0.25)
    ax3.legend(title='Variables', fontsize=8, title_fontsize=9)
    _agregar_descripcion(
        ax3,
        'Compara cada segmento en gasto, actividad y variedad.\nMas cerca de 1 significa mayor nivel en esa metrica frente a los demas segmentos.',
    )
    _mostrar_figura_ajustada(fig3, ancho_px=820, alto_px=520)

    # Ventana 4: mapa de calor resumen
    fig4, ax4 = plt.subplots(figsize=(8.4, 5.2))
    heat = ax4.imshow(resumen_normalizado.to_numpy(), aspect='auto', cmap='YlOrBr')
    fig4.colorbar(heat, ax=ax4, fraction=0.045, pad=0.03, label='Valor normalizado')
    ax4.set_xticks(range(len(resumen_normalizado.columns)))
    ax4.set_xticklabels(resumen_normalizado.columns, rotation=20)
    ax4.set_yticks(range(len(resumen_normalizado.index)))
    ax4.set_yticklabels(resumen_normalizado.index)
    ax4.set_title('Resumen rapido por segmento')
    for i in range(len(resumen_normalizado.index)):
        for j in range(len(resumen_normalizado.columns)):
            ax4.text(j, i, f'{resumen_normalizado.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7)
    _agregar_descripcion(
        ax4,
        'Cada celda resume el nivel relativo del segmento.\nMas cerca de 1 significa valor mas alto frente a los otros segmentos.',
    )
    _mostrar_figura_ajustada(fig4, ancho_px=800, alto_px=500)

    salida_csv = exportar_segmentacion(df, salida_csv='clusters_fact_alquiler_proyecto.csv')
    print(f'Resultados guardados en: {salida_csv}')


if __name__ == '__main__':
    main()
