"""Interfaz gráfica para ejecutar y comparar los algoritmos de recomendación."""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from logica.recomendacion import (
    cargar_datos_mongo,
    construir_dataframes_modelo,
    preparar_datos,
    recomendar_peliculas,
)

class AppRecomendacion:
    """Aplicación Tkinter que permite ejecutar los tres recomendadores."""
    def __init__(self, root: tk.Tk):
        """Inicializa la ventana principal y carga los datos."""
        self.root = root
        self.root.title('Sistema de Recomendacion BI Final')
        self.root.geometry('1360x860')
        self.root.minsize(1180, 760)
        self.root.state('zoomed')

        self.df = None
        self.dataframes_modelo = None
        self.recomendaciones_actuales = pd.DataFrame()
        self.algoritmos_disponibles = {
            'Similitud de coseno': 'coseno',
            'Slope One': 'slope_one',
            'Item a Item': 'item_item',
        }

        self._build_ui()
        self._cargar_datos_iniciales()

    def _build_ui(self):
        """Construye la interfaz visual principal."""
        frame_superior = ttk.Frame(self.root, padding=12)
        frame_superior.pack(fill=tk.X)

        ttk.Label(frame_superior, text='Cliente:').grid(row=0, column=0, padx=4, pady=4, sticky='w')
        self.combo_clientes = ttk.Combobox(frame_superior, width=50, state='readonly')
        self.combo_clientes.grid(row=0, column=1, padx=4, pady=4, sticky='w')

        ttk.Label(frame_superior, text='Recomendaciones:').grid(row=0, column=2, padx=4, pady=4, sticky='w')
        self.entry_n = ttk.Entry(frame_superior, width=8)
        self.entry_n.insert(0, '10')
        self.entry_n.grid(row=0, column=3, padx=4, pady=4, sticky='w')

        ttk.Label(frame_superior, text='Vecinos:').grid(row=0, column=4, padx=4, pady=4, sticky='w')
        self.entry_vecinos = ttk.Entry(frame_superior, width=8)
        self.entry_vecinos.insert(0, '20')
        self.entry_vecinos.grid(row=0, column=5, padx=4, pady=4, sticky='w')

        ttk.Label(frame_superior, text='Algoritmo:').grid(row=0, column=6, padx=4, pady=4, sticky='w')
        self.combo_algoritmo = ttk.Combobox(frame_superior, width=24, state='readonly')
        self.combo_algoritmo['values'] = list(self.algoritmos_disponibles.keys())
        self.combo_algoritmo.set('Similitud de coseno')
        self.combo_algoritmo.grid(row=0, column=7, padx=4, pady=4, sticky='w')

        self.btn_recomendar = ttk.Button(frame_superior, text='Generar recomendaciones', command=self.generar_recomendaciones)
        self.btn_recomendar.grid(row=0, column=8, padx=8, pady=4)

        self.btn_exportar = ttk.Button(frame_superior, text='Exportar', command=self.exportar_csv)
        self.btn_exportar.grid(row=0, column=9, padx=8, pady=4)

        self.btn_recargar = ttk.Button(frame_superior, text='Recargar datos', command=self._cargar_datos_iniciales)
        self.btn_recargar.grid(row=0, column=10, padx=8, pady=4)

        frame_estado = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        frame_estado.pack(fill=tk.X)
        self.lbl_estado = ttk.Label(frame_estado, text='Listo.')
        self.lbl_estado.pack(anchor='w')

        frame_contexto = ttk.LabelFrame(self.root, text='Resumen del Cliente', padding=10)
        frame_contexto.pack(fill=tk.X, padx=12, pady=(0, 8))

        self.txt_contexto = tk.Text(frame_contexto, height=6, wrap='word')
        self.txt_contexto.pack(fill=tk.X)

        frame_tabla = ttk.LabelFrame(self.root, text='Recomendaciones', padding=10)
        frame_tabla.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 6))

        columnas = ('pelicula_titulo', 'categoria_nombre', 'score_final', 'motivo')
        self.tree = ttk.Treeview(frame_tabla, columns=columnas, show='headings', height=14)

        self.tree.heading('pelicula_titulo', text='Pelicula')
        self.tree.heading('categoria_nombre', text='Categoria')
        self.tree.heading('score_final', text='Score')
        self.tree.heading('motivo', text='Motivo principal')

        self.tree.column('pelicula_titulo', width=300, anchor='w')
        self.tree.column('categoria_nombre', width=130, anchor='w')
        self.tree.column('score_final', width=95, anchor='center')
        self.tree.column('motivo', width=560, anchor='w')

        scrollbar_y = ttk.Scrollbar(frame_tabla, orient='vertical', command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(frame_tabla, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree.bind('<<TreeviewSelect>>', self._mostrar_detalle_recomendacion)

        frame_detalle = ttk.LabelFrame(self.root, text='Detalle de la recomendacion seleccionada', padding=10)
        frame_detalle.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.txt_detalle = ScrolledText(frame_detalle, height=14, wrap='word', font=('Segoe UI', 10))
        self.txt_detalle.pack(fill=tk.BOTH, expand=True)
        self.txt_detalle.insert('1.0', 'Selecciona una recomendacion para ver una explicacion mas clara.')

    def _cargar_datos_iniciales(self):
        """Carga datos desde Mongo y prepara el selector de clientes."""
        try:
            self.df = preparar_datos(cargar_datos_mongo())
            self.dataframes_modelo = construir_dataframes_modelo(self.df)
        except Exception as exc:
            messagebox.showerror('Error de carga', f'No se pudieron cargar los datos desde MongoDB.\n\nDetalle: {exc}')
            self.lbl_estado.config(text='Error al cargar datos.')
            return

        clientes = self.df[['cliente_ref']].drop_duplicates().copy()
        if 'cliente_nombre_completo' in self.df.columns:
            nombres = self.df[['cliente_ref', 'cliente_nombre_completo']].drop_duplicates(subset=['cliente_ref'])
            clientes = clientes.merge(nombres, on='cliente_ref', how='left')
            clientes['etiqueta'] = clientes.apply(
                lambda fila: str(fila['cliente_nombre_completo']).strip() if pd.notna(fila['cliente_nombre_completo']) and str(fila['cliente_nombre_completo']).strip() else str(fila['cliente_ref']),
                axis=1,
            )
        else:
            clientes['etiqueta'] = clientes['cliente_ref'].astype(str)

        clientes = clientes.sort_values('etiqueta')
        self.mapa_etiqueta_a_cliente = {fila['etiqueta']: str(fila['cliente_ref']) for _, fila in clientes.iterrows()}

        etiquetas = list(self.mapa_etiqueta_a_cliente.keys())
        self.combo_clientes['values'] = etiquetas
        if etiquetas:
            self.combo_clientes.set(etiquetas[0])

        self._limpiar_tabla()
        self.txt_contexto.delete('1.0', tk.END)
        self.txt_contexto.insert(tk.END, 'Selecciona un cliente y presiona "Generar recomendaciones".')
        self.lbl_estado.config(text=f'Datos cargados: {len(self.df)} registros, {len(etiquetas)} clientes.')

    def _limpiar_tabla(self):
        """Vacía la tabla de resultados antes de una nueva recomendación."""
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _resumir_motivo(self, texto: str, max_chars: int = 95) -> str:
        """Recorta el texto del motivo para mostrarlo en la tabla."""
        txt = str(texto or '').strip()
        if len(txt) <= max_chars:
            return txt
        return txt[: max_chars - 3].rstrip() + '...'

    def _obtener_cliente_seleccionado(self):
        """Convierte la etiqueta visible del cliente en su identificador interno."""
        etiqueta = self.combo_clientes.get().strip()
        if not etiqueta:
            raise ValueError('Debes seleccionar un cliente.')
        if etiqueta not in self.mapa_etiqueta_a_cliente:
            raise ValueError('Cliente no valido en la lista.')
        return self.mapa_etiqueta_a_cliente[etiqueta]

    def generar_recomendaciones(self):
        """Ejecuta el algoritmo seleccionado y actualiza la vista."""
        try:
            if self.df is None:
                raise ValueError('No hay datos cargados.')
            if self.dataframes_modelo is None:
                self.dataframes_modelo = construir_dataframes_modelo(self.df)

            cliente_id = str(self._obtener_cliente_seleccionado())
            n = int(self.entry_n.get().strip())
            vecinos = int(self.entry_vecinos.get().strip())
            algoritmo_label = self.combo_algoritmo.get().strip()
            algoritmo = self.algoritmos_disponibles.get(algoritmo_label, 'coseno')
            if n <= 0 or vecinos <= 0:
                raise ValueError('Los campos Recomendaciones y Vecinos deben ser mayores a cero.')

            recomendaciones, contexto = recomendar_peliculas(
                self.df,
                cliente_id=cliente_id,
                n_recomendaciones=n,
                n_vecinos=vecinos,
                dataframes_modelo=self.dataframes_modelo,
                algoritmo=algoritmo,
            )

            self.recomendaciones_actuales = recomendaciones.copy()
            self._limpiar_tabla()

            for idx, fila in recomendaciones.reset_index(drop=True).iterrows():
                self.tree.insert(
                    '',
                    tk.END,
                    iid=str(idx),
                    values=(
                        fila.get('pelicula_titulo', ''),
                        fila.get('categoria_nombre', ''),
                        f"{float(fila.get('score_final', 0.0)):.4f}",
                        self._resumir_motivo(fila.get('motivo', '')),
                    ),
                )

            self._mostrar_contexto(contexto)
            self.txt_detalle.delete('1.0', tk.END)
            self.txt_detalle.insert('1.0', 'Selecciona una recomendacion para ver por que aparece en tu lista.')
            self.lbl_estado.config(text=f"Recomendaciones generadas para {contexto.get('cliente_mostrar', 'cliente seleccionado')}.")

        except Exception as exc:
            messagebox.showerror('Error', str(exc))
            self.lbl_estado.config(text='No se pudieron generar recomendaciones.')

    def _mostrar_contexto(self, contexto):
        """Muestra un resumen textual del cliente y del algoritmo usado."""
        lineas = [
            f"Algoritmo: {contexto.get('algoritmo', '')}",
            f"Cliente: {contexto.get('cliente_mostrar', '')}",
            f"Registros historicos: {contexto.get('historico_total', '')}",
            f"Peliculas unicas vistas: {contexto.get('peliculas_unicas_vistas', '')}",
            f"Vecinos consultados: {contexto.get('vecinos_consultados', '')}",
        ]

        if contexto.get('segmento_cliente'):
            lineas.append(f"Segmento del cliente: {contexto.get('segmento_cliente')}")

        if contexto.get('vecino_mas_similar_mostrar') is not None:
            lineas.append(
                f"Cliente mas similar: {contexto.get('vecino_mas_similar_mostrar')} (similitud {contexto.get('similitud_maxima', 0):.3f})"
            )

        categorias = contexto.get('categorias_preferidas', {})
        if categorias:
            top = ', '.join([f"{cat}: {peso:.1%}" for cat, peso in categorias.items()])
            lineas.append(f'Categorias preferidas: {top}')

        self.txt_contexto.delete('1.0', tk.END)
        self.txt_contexto.insert(tk.END, '\n'.join(lineas))

    def _mostrar_detalle_recomendacion(self, _event=None):
        """Muestra el detalle de la recomendación seleccionada en la tabla."""
        seleccion = self.tree.selection()
        if not seleccion or self.recomendaciones_actuales.empty:
            return

        idx = int(seleccion[0])
        if idx < 0 or idx >= len(self.recomendaciones_actuales):
            return

        fila = self.recomendaciones_actuales.iloc[idx]
        lineas = [
            f"Pelicula: {fila.get('pelicula_titulo', '')}",
            f"Categoria: {fila.get('categoria_nombre', '')}",
            f"Score final: {float(fila.get('score_final', 0.0)):.4f}",
            '',
            'Motivo principal:',
            f"- {fila.get('motivo', '')}",
            '',
            'Detalle de la recomendacion:',
            f"{str(fila.get('motivo_detalle', '')).replace(' | ', '\\n- ')}",
        ]
        self.txt_detalle.delete('1.0', tk.END)
        self.txt_detalle.insert('1.0', '\n'.join(lineas))

    def exportar_csv(self):
        """Exporta la última recomendación generada a un archivo CSV."""
        if self.recomendaciones_actuales.empty:
            messagebox.showwarning('Sin datos', 'Primero genera recomendaciones para poder exportar.')
            return

        ruta = filedialog.asksaveasfilename(
            title='Guardar recomendaciones',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')],
            initialfile='recomendaciones.csv',
        )
        if not ruta:
            return

        try:
            self.recomendaciones_actuales.to_csv(ruta, index=False)
            self.lbl_estado.config(text=f'CSV exportado: {os.path.basename(ruta)}')
            messagebox.showinfo('Exportacion completa', f'Se guardo el archivo:\n{ruta}')
        except Exception as exc:
            messagebox.showerror('Error al exportar', str(exc))


def main():
    root = tk.Tk()
    AppRecomendacion(root)
    root.mainloop()


if __name__ == '__main__':
    main()
