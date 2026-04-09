"""Libro didactico claro para explicar 3 algoritmos de recomendacion.

Interfaz nueva:
- 3 pestanas (una por algoritmo)
- En cada pestana hay exactamente 3 subapartados:
  1) Entradas: campos, dataframes, clusters
  2) Proceso: formula + pasos
  3) Ejemplo real: calculo numerico + motivo/motivo_detalle
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import sys
import tkinter as tk

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from logica.recomendacion import (
    calcular_similitud_coseno,
    cargar_datos_mongo,
    construir_matriz_valoraciones,
    preparar_datos,
    recomendar_peliculas,
)


class LibroAlgoritmosClaroApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Libro Claro de Recomendacion')
        self.root.geometry('1400x920')
        self.root.minsize(1180, 760)

        self.df = pd.DataFrame()
        self.matriz_val = pd.DataFrame()
        self.titulo_por_ref: dict[str, str] = {}

        self._build_ui()
        self._load_data()
        self._refresh_all_examples()

    # ---------------------- Base UI ----------------------
    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        titulo = ttk.Label(top, text='Libro de Algoritmos - Version Clara', font=('Segoe UI', 18, 'bold'))
        titulo.pack(anchor='w')

        subtitulo = ttk.Label(
            top,
            text='Estructura fija: 3 algoritmos x 3 subapartados. Cada algoritmo muestra campos, proceso y calculo completo.',
            font=('Segoe UI', 10),
        )
        subtitulo.pack(anchor='w', pady=(2, 4))

        barra = ttk.Frame(top)
        barra.pack(fill=tk.X)

        self.lbl_estado = ttk.Label(barra, text='Cargando datos...', font=('Segoe UI', 10))
        self.lbl_estado.pack(side=tk.LEFT)

        ttk.Button(barra, text='Recargar datos', command=self._on_reload).pack(side=tk.RIGHT)

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.tab_cos = ttk.Frame(self.nb, padding=10)
        self.tab_item = ttk.Frame(self.nb, padding=10)
        self.tab_slope = ttk.Frame(self.nb, padding=10)
        self.nb.add(self.tab_cos, text='1) Coseno usuario-usuario')
        self.nb.add(self.tab_item, text='2) Item-item (VSM)')
        self.nb.add(self.tab_slope, text='3) Slope One')

        self._build_tab_coseno(self.tab_cos)
        self._build_tab_item(self.tab_item)
        self._build_tab_slope(self.tab_slope)

    def _section(self, parent: tk.Widget, title: str) -> ttk.LabelFrame:
        sec = ttk.LabelFrame(parent, text=title, padding=10)
        sec.pack(fill=tk.X, pady=(0, 10))
        return sec

    def _add_bullets(self, parent: tk.Widget, lines: list[str]) -> None:
        text = '\n'.join([f'- {line}' for line in lines])
        ttk.Label(parent, text=text, justify='left').pack(anchor='w', fill=tk.X)

    def _add_steps(self, parent: tk.Widget, lines: list[str]) -> None:
        text = '\n'.join([f'{i + 1}. {line}' for i, line in enumerate(lines)])
        ttk.Label(parent, text=text, justify='left').pack(anchor='w', fill=tk.X)

    def _set_text(self, widget: ScrolledText, text: str) -> None:
        widget.configure(state='normal')
        widget.delete('1.0', tk.END)
        widget.insert('1.0', text)
        widget.configure(state='disabled')

    def _set_tree(self, tree: ttk.Treeview, columns: list[str], rows: list[list[str]]) -> None:
        tree.delete(*tree.get_children())
        tree.configure(columns=columns)
        for c in columns:
            tree.heading(c, text=c)
            tree.column(c, width=150, anchor='center')
        for row in rows:
            tree.insert('', tk.END, values=row)

    def _preview_df(self, parent: tk.Widget, df: pd.DataFrame, max_rows: int = 6, max_cols: int = 8) -> None:
        if df is None or df.empty:
            ttk.Label(parent, text='(sin datos)').pack(anchor='w')
            return

        view = df.iloc[:max_rows, :max_cols].copy().reset_index()
        view.columns = [str(c) for c in view.columns]
        cols = view.columns.tolist()

        tree = ttk.Treeview(parent, columns=cols, show='headings', height=min(max_rows, len(view)))
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor='center')

        for _, row in view.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if pd.isna(v):
                    vals.append('-')
                elif isinstance(v, float):
                    vals.append(f'{v:.3f}')
                else:
                    vals.append(str(v))
            tree.insert('', tk.END, values=vals)
        tree.pack(fill=tk.X, pady=(6, 0))

    def _draw_flow(self, parent: tk.Widget, steps: list[str]) -> None:
        width = 1260
        height = 86
        cv = tk.Canvas(parent, width=width, height=height, bg='#FFFFFF', highlightthickness=1, highlightbackground='#D6DEE6')
        cv.pack(fill=tk.X, pady=(8, 0))

        n = max(1, len(steps))
        gap = (width - 60) / n
        box_w = min(200, int(gap - 22))
        y = 18
        for i, step in enumerate(steps):
            x = 30 + i * gap
            cv.create_rectangle(x, y, x + box_w, y + 42, outline='#1D4ED8', width=1, fill='#EFF6FF')
            cv.create_text(x + box_w / 2, y + 21, text=step, width=box_w - 10, fill='#1E3A8A', font=('Segoe UI', 9))
            if i < n - 1:
                cv.create_line(x + box_w, y + 21, x + gap - 4, y + 21, fill='#64748B', arrow=tk.LAST)

    # ---------------------- Data helpers ----------------------
    def _load_data(self) -> None:
        try:
            self.df = preparar_datos(cargar_datos_mongo())
            self.matriz_val = construir_matriz_valoraciones(self.df)
            self._build_title_map()
            self.lbl_estado.config(
                text=f'Datos OK: {len(self.df)} filas | Matriz R(u,i): {self.matriz_val.shape[0]}x{self.matriz_val.shape[1]}'
            )
        except Exception as exc:
            self.df = pd.DataFrame()
            self.matriz_val = pd.DataFrame()
            self.titulo_por_ref = {}
            self.lbl_estado.config(text=f'Sin datos reales ({exc})')

    def _build_title_map(self) -> None:
        self.titulo_por_ref = {}
        if self.df.empty or 'pelicula_ref' not in self.df.columns:
            return
        if 'pelicula_titulo' not in self.df.columns:
            for ref in self.df['pelicula_ref'].astype(str).unique().tolist():
                self.titulo_por_ref[str(ref)] = str(ref)
            return
        aux = self.df[['pelicula_ref', 'pelicula_titulo']].dropna().drop_duplicates('pelicula_ref')
        for _, row in aux.iterrows():
            self.titulo_por_ref[str(row['pelicula_ref'])] = str(row['pelicula_titulo'])

    def _dense_submatrix(self, max_rows: int = 8, max_cols: int = 8) -> pd.DataFrame:
        m = self.matriz_val
        if m.empty:
            return pd.DataFrame()

        row_order = m.notna().sum(axis=1).sort_values(ascending=False)
        rows = row_order[row_order > 0].index.tolist()[:max_rows]
        if not rows:
            return pd.DataFrame()

        sub = m.loc[rows]
        col_order = sub.notna().sum(axis=0).sort_values(ascending=False)
        cols = col_order[col_order > 0].index.tolist()[:max_cols]
        if not cols:
            return pd.DataFrame()

        return m.loc[rows, cols]

    def _demo_cliente(self, min_items: int = 2) -> str | None:
        if self.matriz_val.empty:
            return None
        counts = self.matriz_val.notna().sum(axis=1)
        ok = counts[counts >= min_items]
        if ok.empty:
            return None
        return str(ok.sort_values(ascending=False).index[0])

    def _titulo(self, ref: str) -> str:
        return self.titulo_por_ref.get(str(ref), str(ref))

    # ---------------------- Tab builders ----------------------
    def _build_tab_coseno(self, parent: ttk.Frame) -> None:
        s1 = self._section(parent, 'Subapartado 1/3 - Entradas: campos, dataframes, clusters')
        self._add_bullets(s1, [
            'Campos: cliente_ref, pelicula_ref, ingreso, categoria_nombre.',
            'DataFrames: df preparado, matriz_val, matriz_fill, candidatos.',
            'Clusters: no participan en la formula del score de coseno.',
        ])
        self.cos_matrix_holder = ttk.Frame(s1)
        self.cos_matrix_holder.pack(fill=tk.X)

        s2 = self._section(parent, 'Subapartado 2/3 - Proceso + formula + flujo')
        ttk.Label(
            s2,
            text='Formula:\nsim(u,v)=(u.v)/(||u||*||v||)\nscore(i)=sum_v(sim(u,v)*r(v,i))/sum_v(|sim(u,v)|)',
            justify='left',
            font=('Consolas', 10),
        ).pack(anchor='w')
        self._add_steps(s2, [
            'Construir R(u,i) con ingreso promedio.',
            'Calcular similitud entre cliente objetivo y vecinos.',
            'Tomar vecinos con similitud positiva.',
            'Armar numerador y denominador por pelicula candidata.',
            'Obtener score crudo, normalizar y ordenar.',
            'Construir motivo y motivo_detalle.',
        ])
        self._draw_flow(s2, ['R(u,i)', 'sim(u,v)', 'vecinos', 'score crudo', 'score_final', 'motivo'])

        s3 = self._section(parent, 'Subapartado 3/3 - Ejemplo real de calculo hasta score final')
        ttk.Button(s3, text='Actualizar ejemplo coseno', command=self._refresh_coseno_example).pack(anchor='w')
        self.cos_text = ScrolledText(s3, height=10, wrap='word', font=('Consolas', 10))
        self.cos_text.pack(fill=tk.X, pady=(6, 6))
        self.cos_tree = ttk.Treeview(s3, show='headings', height=6)
        self.cos_tree.pack(fill=tk.X)

    def _build_tab_item(self, parent: ttk.Frame) -> None:
        s1 = self._section(parent, 'Subapartado 1/3 - Entradas: campos, dataframes, clusters')
        self._add_bullets(s1, [
            'Campos: cliente_ref, pelicula_ref, ingreso.',
            'DataFrames: matriz_val y matriz_items=matriz_val.fillna(0).T.',
            'Clusters: no participan en la formula item-item.',
        ])
        self.item_matrix_holder = ttk.Frame(s1)
        self.item_matrix_holder.pack(fill=tk.X)

        s2 = self._section(parent, 'Subapartado 2/3 - Proceso + formula + flujo')
        ttk.Label(
            s2,
            text='Formula:\nsim(i,j)=(i.j)/(||i||*||j||)\nscore(i)=sum_j(sim(i,j)*r(u,j))/sum_j(|sim(i,j)|)',
            justify='left',
            font=('Consolas', 10),
        ).pack(anchor='w')
        self._add_steps(s2, [
            'Construir matriz VSM (peliculas x clientes).',
            'Calcular similitud entre candidata i y peliculas vistas j.',
            'Filtrar similitudes positivas y top-k.',
            'Calcular numerador y denominador.',
            'Obtener score crudo, normalizar y ordenar.',
            'Construir motivo y motivo_detalle.',
        ])
        self._draw_flow(s2, ['R(u,i)', 'VSM', 'sim(i,j)', 'score crudo', 'score_final', 'motivo'])

        s3 = self._section(parent, 'Subapartado 3/3 - Ejemplo real de calculo hasta score final')
        ttk.Button(s3, text='Actualizar ejemplo item-item', command=self._refresh_item_example).pack(anchor='w')
        self.item_text = ScrolledText(s3, height=10, wrap='word', font=('Consolas', 10))
        self.item_text.pack(fill=tk.X, pady=(6, 6))
        self.item_tree = ttk.Treeview(s3, show='headings', height=6)
        self.item_tree.pack(fill=tk.X)

    def _build_tab_slope(self, parent: ttk.Frame) -> None:
        s1 = self._section(parent, 'Subapartado 1/3 - Entradas: campos, dataframes, clusters')
        self._add_bullets(s1, [
            'Campos: cliente_ref, pelicula_ref, ingreso.',
            'DataFrames: matriz_val y estructuras dev/freq por pares de peliculas.',
            'Clusters: no participan en la formula slope one.',
        ])
        self.slope_matrix_holder = ttk.Frame(s1)
        self.slope_matrix_holder.pack(fill=tk.X)

        s2 = self._section(parent, 'Subapartado 2/3 - Proceso + formula + flujo')
        ttk.Label(
            s2,
            text='Formula:\ndev(i,j)=avg_u(r(u,i)-r(u,j))\npred(u,i)=sum_j((dev(i,j)+r(u,j))*freq(i,j))/sum_j(freq(i,j))',
            justify='left',
            font=('Consolas', 10),
        ).pack(anchor='w')
        self._add_steps(s2, [
            'Recorrer usuarios y acumular diferencias por par (i,j).',
            'Acumular frecuencias por par.',
            'Promediar para obtener dev(i,j).',
            'Para una candidata i, combinar evidencias de items vistos j.',
            'Calcular score crudo, normalizar y ordenar.',
            'Construir motivo y motivo_detalle.',
        ])
        self._draw_flow(s2, ['R(u,i)', 'dev/freq', 'pred_par', 'score crudo', 'score_final', 'motivo'])

        s3 = self._section(parent, 'Subapartado 3/3 - Ejemplo real de calculo hasta score final')
        ttk.Button(s3, text='Actualizar ejemplo slope one', command=self._refresh_slope_example).pack(anchor='w')
        self.slope_text = ScrolledText(s3, height=10, wrap='word', font=('Consolas', 10))
        self.slope_text.pack(fill=tk.X, pady=(6, 6))
        self.slope_tree = ttk.Treeview(s3, show='headings', height=6)
        self.slope_tree.pack(fill=tk.X)

    # ---------------------- Refresh events ----------------------
    def _on_reload(self) -> None:
        self._load_data()
        self._refresh_all_examples()

    def _refresh_all_examples(self) -> None:
        self._refresh_matrix_previews()
        self._refresh_coseno_example()
        self._refresh_item_example()
        self._refresh_slope_example()

    def _refresh_matrix_previews(self) -> None:
        for holder in [self.cos_matrix_holder, self.item_matrix_holder, self.slope_matrix_holder]:
            for w in holder.winfo_children():
                w.destroy()

        sub = self._dense_submatrix()
        ttk.Label(self.cos_matrix_holder, text='Vista R(u,i) observada').pack(anchor='w')
        self._preview_df(self.cos_matrix_holder, sub)
        ttk.Label(self.cos_matrix_holder, text='Vista R(u,i) con fillna(0) para calculo').pack(anchor='w', pady=(6, 0))
        self._preview_df(self.cos_matrix_holder, sub.fillna(0.0) if not sub.empty else pd.DataFrame())

        ttk.Label(self.item_matrix_holder, text='Vista R(u,i)').pack(anchor='w')
        self._preview_df(self.item_matrix_holder, sub)
        ttk.Label(self.item_matrix_holder, text='Vista VSM = R(u,i).T').pack(anchor='w', pady=(6, 0))
        self._preview_df(self.item_matrix_holder, sub.fillna(0.0).T if not sub.empty else pd.DataFrame())

        ttk.Label(self.slope_matrix_holder, text='Vista R(u,i) para formar pares').pack(anchor='w')
        self._preview_df(self.slope_matrix_holder, sub)

    # ---------------------- Coseno example ----------------------
    def _refresh_coseno_example(self) -> None:
        ex = self._calc_coseno_example()
        if ex is None:
            self._set_text(self.cos_text, 'No se pudo construir ejemplo de coseno con los datos actuales.')
            self._set_tree(self.cos_tree, ['vecino', 'sim', 'r(v,i)', 'sim*r(v,i)'], [])
            return

        lineas = [
            f'Cliente objetivo: {ex["cliente"]}',
            f'Pelicula candidata: {ex["titulo"]} ({ex["candidata"]})',
            '',
            'Formula aplicada:',
            'score(i)=sum_v(sim(u,v)*r(v,i))/sum_v(|sim(u,v)|)',
            '',
            f'Numerador = {ex["numerador"]:.6f}',
            f'Denominador = {ex["denominador"]:.6f}',
            f'Score crudo = {ex["score_crudo"]:.6f}',
            '',
            'Construccion de motivo:',
            '- score crudo -> normalizacion -> score_final',
            '- score_final + categoria -> motivo',
            '- motivo + lineas_extra de vecinos -> motivo_detalle',
        ]
        if ex.get('motivo_detalle'):
            lineas.extend(['', 'Motivo detalle real del motor:', ex['motivo_detalle']])

        self._set_text(self.cos_text, '\n'.join(lineas))
        rows = [[v, f'{s:.4f}', f'{r:.4f}', f'{a:.4f}'] for v, s, r, a in ex['aportes']]
        self._set_tree(self.cos_tree, ['vecino', 'sim', 'r(v,i)', 'sim*r(v,i)'], rows)

    def _calc_coseno_example(self) -> dict | None:
        if self.matriz_val.empty:
            return None
        cliente = self._demo_cliente(min_items=2)
        if cliente is None:
            return None

        base = self.matriz_val
        sims = calcular_similitud_coseno(base.fillna(0.0), cliente).head(20)
        if sims.empty:
            return None

        vistos = set(base.loc[cliente].dropna().index.astype(str).tolist())
        candidatas = [str(c) for c in base.columns.astype(str).tolist() if str(c) not in vistos]
        for cand in candidatas:
            aportes = []
            for vecino, sim in sims.items():
                v = str(vecino)
                rating = base.loc[v, cand] if cand in base.columns else np.nan
                if pd.notna(rating) and float(sim) > 0:
                    aportes.append((v, float(sim), float(rating), float(sim) * float(rating)))
            if not aportes:
                continue

            numerador = float(sum(a[3] for a in aportes))
            denominador = float(sum(abs(a[1]) for a in aportes))
            score_crudo = numerador / denominador if denominador > 0 else 0.0

            motivo_detalle = ''
            try:
                recs, _ = recomendar_peliculas(self.df, cliente, 1, 10, algoritmo='coseno')
                if not recs.empty:
                    motivo_detalle = str(recs.iloc[0].get('motivo_detalle', ''))
            except Exception:
                motivo_detalle = ''

            return {
                'cliente': cliente,
                'candidata': cand,
                'titulo': self._titulo(cand),
                'aportes': sorted(aportes, key=lambda x: x[1], reverse=True)[:8],
                'numerador': numerador,
                'denominador': denominador,
                'score_crudo': score_crudo,
                'motivo_detalle': motivo_detalle,
            }
        return None

    # ---------------------- Item-item example ----------------------
    def _refresh_item_example(self) -> None:
        ex = self._calc_item_example()
        if ex is None:
            self._set_text(self.item_text, 'No se pudo construir ejemplo de item-item con los datos actuales.')
            self._set_tree(self.item_tree, ['item_ref', 'sim(i,j)', 'r(u,j)', 'sim*r(u,j)'], [])
            return

        lineas = [
            f'Cliente objetivo: {ex["cliente"]}',
            f'Pelicula candidata i: {ex["titulo"]} ({ex["candidata"]})',
            '',
            'Formula aplicada:',
            'score(i)=sum_j(sim(i,j)*r(u,j))/sum_j(|sim(i,j)|)',
            '',
            f'Numerador = {ex["numerador"]:.6f}',
            f'Denominador = {ex["denominador"]:.6f}',
            f'Score crudo = {ex["score_crudo"]:.6f}',
            '',
            'Construccion de motivo:',
            '- score crudo -> normalizacion -> score_final',
            '- score_final + categoria -> motivo',
            '- motivo + lineas_extra de items de referencia -> motivo_detalle',
        ]
        if ex.get('motivo_detalle'):
            lineas.extend(['', 'Motivo detalle real del motor:', ex['motivo_detalle']])

        self._set_text(self.item_text, '\n'.join(lineas))
        rows = [[i, f'{s:.4f}', f'{r:.4f}', f'{a:.4f}'] for i, s, r, a in ex['aportes']]
        self._set_tree(self.item_tree, ['item_ref', 'sim(i,j)', 'r(u,j)', 'sim*r(u,j)'], rows)

    def _calc_item_example(self) -> dict | None:
        if self.matriz_val.empty:
            return None
        cliente = self._demo_cliente(min_items=2)
        if cliente is None:
            return None

        base = self.matriz_val
        user = base.loc[cliente]
        vistos = user.dropna()
        if vistos.empty:
            return None

        vsm = base.fillna(0.0).T
        candidatas = [str(c) for c in base.columns.astype(str).tolist() if pd.isna(user[c])]
        for cand in candidatas:
            if cand not in vsm.index:
                continue
            vec_i = vsm.loc[cand].to_numpy(dtype=float)
            aportes = []
            for j, rating in vistos.items():
                j = str(j)
                if j not in vsm.index:
                    continue
                vec_j = vsm.loc[j].to_numpy(dtype=float)
                ni = float(np.linalg.norm(vec_i))
                nj = float(np.linalg.norm(vec_j))
                sim = 0.0 if ni == 0 or nj == 0 else float(np.dot(vec_i, vec_j) / (ni * nj))
                if sim > 0:
                    aportes.append((j, sim, float(rating), sim * float(rating)))
            if not aportes:
                continue

            aportes = sorted(aportes, key=lambda x: x[1], reverse=True)[:8]
            numerador = float(sum(a[3] for a in aportes))
            denominador = float(sum(abs(a[1]) for a in aportes))
            score_crudo = numerador / denominador if denominador > 0 else 0.0

            motivo_detalle = ''
            try:
                recs, _ = recomendar_peliculas(self.df, cliente, 1, 10, algoritmo='item_item')
                if not recs.empty:
                    motivo_detalle = str(recs.iloc[0].get('motivo_detalle', ''))
            except Exception:
                motivo_detalle = ''

            return {
                'cliente': cliente,
                'candidata': cand,
                'titulo': self._titulo(cand),
                'aportes': aportes,
                'numerador': numerador,
                'denominador': denominador,
                'score_crudo': score_crudo,
                'motivo_detalle': motivo_detalle,
            }
        return None

    # ---------------------- Slope One example ----------------------
    def _refresh_slope_example(self) -> None:
        ex = self._calc_slope_example()
        if ex is None:
            self._set_text(self.slope_text, 'No se pudo construir ejemplo de slope one con los datos actuales.')
            self._set_tree(self.slope_tree, ['item_ref', 'dev(i,j)', 'freq(i,j)', 'pred_par', 'pred_par*freq'], [])
            return

        lineas = [
            f'Cliente objetivo: {ex["cliente"]}',
            f'Pelicula candidata i: {ex["titulo"]} ({ex["candidata"]})',
            '',
            'Formula aplicada:',
            'dev(i,j)=avg_u(r(u,i)-r(u,j))',
            'pred(u,i)=sum_j((dev(i,j)+r(u,j))*freq(i,j))/sum_j(freq(i,j))',
            '',
            f'Numerador = {ex["numerador"]:.6f}',
            f'Denominador = {ex["denominador"]:.6f}',
            f'Score crudo = {ex["score_crudo"]:.6f}',
            '',
            'Construccion de motivo:',
            '- score crudo -> normalizacion -> score_final',
            '- score_final + categoria -> motivo',
            '- motivo + lineas_extra con desviaciones/frecuencias -> motivo_detalle',
        ]
        if ex.get('motivo_detalle'):
            lineas.extend(['', 'Motivo detalle real del motor:', ex['motivo_detalle']])

        self._set_text(self.slope_text, '\n'.join(lineas))
        rows = [
            [ref, f'{dev:+.4f}', str(freq), f'{pred:.4f}', f'{w:.4f}']
            for ref, _rating, dev, freq, pred, w in ex['evidencias']
        ]
        self._set_tree(self.slope_tree, ['item_ref', 'dev(i,j)', 'freq(i,j)', 'pred_par', 'pred_par*freq'], rows)

    def _calc_slope_example(self) -> dict | None:
        if self.matriz_val.empty:
            return None
        cliente = self._demo_cliente(min_items=2)
        if cliente is None:
            return None

        base = self.matriz_val
        dev_sum = defaultdict(float)
        freq = defaultdict(int)

        for _, fila in base.iterrows():
            rated = fila.dropna()
            items = [str(x) for x in rated.index.tolist()]
            for i in items:
                for j in items:
                    if i == j:
                        continue
                    dev_sum[(i, j)] += float(rated[i] - rated[j])
                    freq[(i, j)] += 1

        dev = {}
        for key, total in dev_sum.items():
            f = freq.get(key, 0)
            if f > 0:
                dev[key] = total / f

        user_rated = base.loc[cliente].dropna()
        candidatas = [str(c) for c in base.columns.astype(str).tolist() if pd.isna(base.loc[cliente, c])]
        for cand in candidatas:
            evidencias = []
            for ref, rating in user_rated.items():
                ref = str(ref)
                key = (cand, ref)
                if key not in dev:
                    continue
                f = int(freq[key])
                d = float(dev[key])
                pred_par = d + float(rating)
                evidencias.append((ref, float(rating), d, f, pred_par, pred_par * f))
            if not evidencias:
                continue

            evidencias = sorted(evidencias, key=lambda x: x[3], reverse=True)[:8]
            numerador = float(sum(e[5] for e in evidencias))
            denominador = float(sum(e[3] for e in evidencias))
            score_crudo = numerador / denominador if denominador > 0 else 0.0

            motivo_detalle = ''
            try:
                recs, _ = recomendar_peliculas(self.df, cliente, 1, 10, algoritmo='slope_one')
                if not recs.empty:
                    motivo_detalle = str(recs.iloc[0].get('motivo_detalle', ''))
            except Exception:
                motivo_detalle = ''

            return {
                'cliente': cliente,
                'candidata': cand,
                'titulo': self._titulo(cand),
                'evidencias': evidencias,
                'numerador': numerador,
                'denominador': denominador,
                'score_crudo': score_crudo,
                'motivo_detalle': motivo_detalle,
            }
        return None


# Compatibilidad con imports existentes en la interfaz principal.
LibroAlgoritmosApp = LibroAlgoritmosClaroApp


def main() -> None:
    root = tk.Tk()
    LibroAlgoritmosClaroApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
