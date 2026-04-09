"""Carga y enriquecimiento de la tabla fact para el proyecto."""

import json
from pathlib import Path

import pandas as pd
from pymongo import MongoClient


BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_EXCEL = BASE_DIR / 'BI-FINAL.xlsx'
RUTA_JSON = BASE_DIR / 'fact_alquiler_proyecto.json'
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'BI_Final'
COLLECTION_NAME = 'fact_alquiler'

COLUMNAS_FACT_NUMERICAS = [
    'id_hecho', 'id_tiempo', 'id_cliente', 'id_pelicula', 'id_tienda',
    'id_categoria', 'id_ciudad', 'cantidad_alquiler', 'ingreso', 'duracion_alquiler',
]


def cargar_hoja(excel: pd.ExcelFile, nombre_hoja: str) -> pd.DataFrame:
    """Carga una hoja específica del archivo Excel."""
    if nombre_hoja not in excel.sheet_names:
        raise ValueError(f'No existe la hoja requerida: {nombre_hoja}')
    return excel.parse(nombre_hoja)


def leer_tablas() -> dict:
    """Lee todas las tablas del modelo estrella desde el Excel."""
    excel = pd.ExcelFile(RUTA_EXCEL)
    return {
        'fact': cargar_hoja(excel, 'FACT_ALQUILER'),
        'actor': cargar_hoja(excel, 'DIM_ACTOR'),
        'categoria': cargar_hoja(excel, 'DIM_CATEGORIA'),
        'ciudad': cargar_hoja(excel, 'DIM_CIUDAD'),
        'cliente': cargar_hoja(excel, 'DIM_CLIENTE'),
        'pelicula': cargar_hoja(excel, 'DIM_PELICULA'),
        'tiempo': cargar_hoja(excel, 'DIM_TIEMPO'),
        'tienda': cargar_hoja(excel, 'DIM_TIENDA'),
        'puente': cargar_hoja(excel, 'PUENTE_PELICULA_ACTOR'),
    }


def preparar_fact(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas clave a numéricas y elimina registros sin ingreso."""
    df = df.copy()
    for columna in COLUMNAS_FACT_NUMERICAS:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors='coerce')

    return df.dropna(subset=['ingreso'])


def _seleccionar_y_renombrar(df: pd.DataFrame, columnas: list[str], renombres: dict[str, str]) -> pd.DataFrame:
    """Selecciona columnas útiles y aplica nombres más legibles."""
    return df[columnas].rename(columns=renombres)


def _construir_dimensiones(tablas: dict) -> dict[str, pd.DataFrame]:
    """Prepara cada dimensión con sus nombres finales para el merge."""
    return {
        'cliente': _seleccionar_y_renombrar(
            tablas['cliente'],
            ['id_cliente', 'nombre', 'apellido', 'email', 'activo'],
            {
                'nombre': 'cliente_nombre',
                'apellido': 'cliente_apellido',
                'email': 'cliente_email',
                'activo': 'cliente_activo',
            },
        ),
        'categoria': _seleccionar_y_renombrar(
            tablas['categoria'],
            ['id_categoria', 'nombre_categoria'],
            {'nombre_categoria': 'categoria_nombre'},
        ),
        'ciudad': _seleccionar_y_renombrar(
            tablas['ciudad'],
            ['id_ciudad', 'ciudad', 'pais'],
            {'ciudad': 'ciudad_nombre', 'pais': 'ciudad_pais'},
        ),
        'pelicula': _seleccionar_y_renombrar(
            tablas['pelicula'],
            ['id_pelicula', 'titulo', 'duracion', 'clasificacion', 'anio_lanzamiento', 'idioma', 'precio_renta', 'costo_reposicion'],
            {
                'titulo': 'pelicula_titulo',
                'duracion': 'pelicula_duracion',
                'clasificacion': 'pelicula_clasificacion',
                'anio_lanzamiento': 'pelicula_anio_lanzamiento',
                'idioma': 'pelicula_idioma',
                'precio_renta': 'pelicula_precio_renta',
                'costo_reposicion': 'pelicula_costo_reposicion',
            },
        ),
        'tiempo': _seleccionar_y_renombrar(
            tablas['tiempo'],
            ['id_tiempo', 'fecha', 'dia', 'mes', 'nombre_mes', 'trimestre', 'anio'],
            {
                'fecha': 'tiempo_fecha',
                'dia': 'tiempo_dia',
                'mes': 'tiempo_mes',
                'nombre_mes': 'tiempo_nombre_mes',
                'trimestre': 'tiempo_trimestre',
                'anio': 'tiempo_anio',
            },
        ),
        'tienda': _seleccionar_y_renombrar(
            tablas['tienda'],
            ['id_tienda', 'nombre_tienda'],
            {'nombre_tienda': 'tienda_nombre'},
        ),
    }


def _agrupar_actores(tablas: dict) -> pd.DataFrame:
    """Agrupa actores por película para facilitar la explicación del dataset."""
    puente = tablas['puente'].merge(
        tablas['actor'][['id_actor', 'nombre_actor']],
        on='id_actor',
        how='left',
    )

    return puente.groupby('id_pelicula').agg(
        pelicula_actores=('nombre_actor', lambda serie: sorted({str(valor) for valor in serie.dropna()})),
        cantidad_actores=('id_actor', 'nunique'),
    ).reset_index()


def unir_dimensiones(tablas: dict) -> pd.DataFrame:
    """Enriquece la tabla fact con las dimensiones y métricas derivadas."""
    fact = preparar_fact(tablas['fact'])
    dimensiones = _construir_dimensiones(tablas)

    for nombre_dim in ['cliente', 'categoria', 'ciudad', 'pelicula', 'tiempo', 'tienda']:
        columna_id = f'id_{nombre_dim}'
        fact = fact.merge(dimensiones[nombre_dim], on=columna_id, how='left')

    fact = fact.merge(_agrupar_actores(tablas), on='id_pelicula', how='left')

    fact['cliente_nombre_completo'] = (
        fact['cliente_nombre'].fillna('').astype(str).str.strip()
        + ' '
        + fact['cliente_apellido'].fillna('').astype(str).str.strip()
    ).str.strip()
    fact.loc[fact['cliente_nombre_completo'] == '', 'cliente_nombre_completo'] = None

    return fact


def guardar_json_y_mongo(df: pd.DataFrame) -> None:
    """Guarda el fact enriquecido en JSON y en MongoDB."""
    columnas_sin_ids = [col for col in df.columns if not col.lower().startswith('id_')]
    registros = df[columnas_sin_ids].to_dict(orient='records')

    with open(RUTA_JSON, 'w', encoding='utf-8') as archivo:
        json.dump(registros, archivo, ensure_ascii=False, indent=4)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.drop()
    if registros:
        collection.insert_many(registros)


def main() -> None:
    """Ejecuta la carga completa del archivo Excel hacia JSON y MongoDB."""
    if not RUTA_EXCEL.exists():
        raise FileNotFoundError(f'No se encontro el archivo Excel: {RUTA_EXCEL}')

    tablas = leer_tablas()
    df = unir_dimensiones(tablas)
    guardar_json_y_mongo(df)

    print(f'Archivo JSON creado en: {RUTA_JSON}')
    print(f'¡Exito! Se insertaron {len(df)} documentos en MongoDB ({DB_NAME}.{COLLECTION_NAME}).')
    print('El fact enriquecido se guarda sin IDs tecnicos y agrega nombres como cliente_nombre_completo, categoria_nombre y pelicula_titulo.')


if __name__ == '__main__':
    main()
