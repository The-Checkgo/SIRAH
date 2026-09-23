import sqlite3
import os
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RUTA_DB = os.path.join(
    BASE_DIR,
    "database",
    "thaps.db"
)

CARPETA_MODELO = os.path.join(
    BASE_DIR,
    "modelo"
)

RUTA_MODELO = os.path.join(
    CARPETA_MODELO,
    "modelo_sirah.pkl"
)

RUTA_GRAFICA = os.path.join(
    CARPETA_MODELO,
    "matriz_confusion.png"
)


# Crear carpeta modelo si no existe
os.makedirs(
    CARPETA_MODELO,
    exist_ok=True
)


# ============================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================

SEMILLA = 42

PORCENTAJE_PRUEBA = 0.20

NUM_ARBOLES = 200


# ============================================================
# COMPROBAR BASE DE DATOS
# ============================================================

if not os.path.exists(RUTA_DB):

    print("ERROR: No se encontró la base de datos.")

    print(RUTA_DB)

    exit()


print("=" * 70)
print("             ENTRENAMIENTO DEL MODELO SIRAH")
print("=" * 70)

print("\nBase de datos:")
print(RUTA_DB)


# ============================================================
# CONECTAR A SQLITE
# ============================================================

conexion = sqlite3.connect(RUTA_DB)


# ============================================================
# LEER CARACTERÍSTICAS
# ============================================================

consulta = """
SELECT
    caracteristicas_audio.id,
    caracteristicas_audio.sonido_id,
    caracteristicas_audio.segmento,
    caracteristicas_audio.inicio,
    caracteristicas_audio.fin,
    caracteristicas_audio.energia,
    caracteristicas_audio.frecuencia_dominante,
    caracteristicas_audio.mfcc,
    sonidos.nombre AS nombre_audio,
    categorias.nombre AS categoria

FROM caracteristicas_audio

INNER JOIN sonidos
    ON caracteristicas_audio.sonido_id = sonidos.id

INNER JOIN categorias
    ON sonidos.categoria_id = categorias.id

ORDER BY caracteristicas_audio.id
"""


df = pd.read_sql_query(
    consulta,
    conexion
)


conexion.close()


# ============================================================
# COMPROBAR DATOS
# ============================================================

print("\nRegistros encontrados:")

print(len(df))

if len(df) == 0:

    print("\nERROR: No hay características en la base de datos.")

    exit()


print("\nPrimeros registros:")

print(
    df[
        [
            "sonido_id",
            "segmento",
            "energia",
            "frecuencia_dominante",
            "categoria"
        ]
    ].head()
)


# ============================================================
# CONVERTIR MFCC DE JSON A NÚMEROS
# ============================================================

print("\nProcesando MFCC...")


def convertir_mfcc(mfcc_texto):

    try:

        valores = json.loads(mfcc_texto)

        return valores

    except Exception:

        return None


df["mfcc_lista"] = df["mfcc"].apply(
    convertir_mfcc
)


# ============================================================
# ELIMINAR REGISTROS CON MFCC INVÁLIDOS
# ============================================================

antes = len(df)

df = df[
    df["mfcc_lista"].notna()
].copy()

despues = len(df)


if antes != despues:

    print(
        f"Registros eliminados por MFCC inválido: "
        f"{antes - despues}"
    )


# ============================================================
# CONVERTIR LOS 13 MFCC EN COLUMNAS
# ============================================================

mfcc_df = pd.DataFrame(
    df["mfcc_lista"].tolist(),
    index=df.index
)


# Cambiar nombres:

mfcc_df.columns = [
    f"mfcc_{i + 1}"
    for i in range(mfcc_df.shape[1])
]


# ============================================================
# UNIR MFCC CON ENERGÍA Y FRECUENCIA
# ============================================================

df = pd.concat(
    [
        df,
        mfcc_df
    ],
    axis=1
)


# ============================================================
# CREAR VARIABLES X
# ============================================================

columnas_mfcc = [
    f"mfcc_{i + 1}"
    for i in range(13)
]


columnas_caracteristicas = (
    columnas_mfcc
    + [
        "energia",
        "frecuencia_dominante"
    ]
)


X = df[
    columnas_caracteristicas
].astype(float)


# ============================================================
# CREAR ETIQUETAS Y
# ============================================================

y = df[
    "categoria"
]


# ============================================================
# GRUPOS
# ============================================================
#
# Cada grupo representa un AUDIO ORIGINAL.
#
# Esto es importante porque varios fragmentos pertenecen
# al mismo audio.
#
# No queremos que fragmentos del mismo audio aparezcan
# tanto en entrenamiento como en prueba.
# ============================================================

grupos = df[
    "sonido_id"
]


# ============================================================
# INFORMACIÓN DEL DATASET
# ============================================================

print("\n" + "=" * 70)
print("              INFORMACIÓN DEL DATASET")
print("=" * 70)

print(
    f"\nTotal de fragmentos: {len(df)}"
)

print(
    f"Total de audios originales: "
    f"{df['sonido_id'].nunique()}"
)

print("\nFragmentos por categoría:")

print(
    df["categoria"].value_counts()
)


print("\nCaracterísticas utilizadas:")

for columna in columnas_caracteristicas:

    print(
        f"  ✓ {columna}"
    )


print(
    f"\nTotal de características por fragmento: "
    f"{len(columnas_caracteristicas)}"
)


# ============================================================
# DIVISIÓN ENTRENAMIENTO / PRUEBA
# ============================================================
#
# IMPORTANTE:
#
# Se divide por sonido_id y NO por fragmento.
#
# Aproximadamente:
#
# 80% de los audios → entrenamiento
# 20% de los audios → prueba
#
# ============================================================

print("\n" + "=" * 70)
print("          DIVISIÓN ENTRENAMIENTO / PRUEBA")
print("=" * 70)


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=PORCENTAJE_PRUEBA,
    random_state=SEMILLA
)


indices_entrenamiento, indices_prueba = next(
    splitter.split(
        X,
        y,
        groups=grupos
    )
)


X_entrenamiento = X.iloc[
    indices_entrenamiento
]

X_prueba = X.iloc[
    indices_prueba
]


y_entrenamiento = y.iloc[
    indices_entrenamiento
]

y_prueba = y.iloc[
    indices_prueba
]


grupos_entrenamiento = grupos.iloc[
    indices_entrenamiento
]

grupos_prueba = grupos.iloc[
    indices_prueba
]


# ============================================================
# MOSTRAR DIVISIÓN
# ============================================================

print(
    f"\nFragmentos de entrenamiento: "
    f"{len(X_entrenamiento)}"
)

print(
    f"Fragmentos de prueba: "
    f"{len(X_prueba)}"
)


print(
    f"\nAudios de entrenamiento: "
    f"{grupos_entrenamiento.nunique()}"
)

print(
    f"Audios de prueba: "
    f"{grupos_prueba.nunique()}"
)


# Comprobar que no hay audios compartidos

audios_compartidos = set(
    grupos_entrenamiento
).intersection(
    set(grupos_prueba)
)


print(
    f"\nAudios compartidos entre entrenamiento "
    f"y prueba: {len(audios_compartidos)}"
)


if len(audios_compartidos) > 0:

    print(
        "ADVERTENCIA: Hay audios compartidos."
    )

else:

    print(
        "✓ No hay audios compartidos."
    )


# ============================================================
# CREAR CLASIFICADOR
# ============================================================

print("\n" + "=" * 70)
print("                ENTRENANDO MODELO")
print("=" * 70)


modelo = RandomForestClassifier(

    n_estimators=NUM_ARBOLES,

    random_state=SEMILLA,

    class_weight="balanced",

    n_jobs=-1

)


# ============================================================
# ENTRENAMIENTO
# ============================================================

modelo.fit(
    X_entrenamiento,
    y_entrenamiento
)


print("\n✓ Modelo entrenado correctamente.")


# ============================================================
# PREDICCIONES
# ============================================================

print("\nGenerando predicciones...")


predicciones = modelo.predict(
    X_prueba
)


# ============================================================
# PRECISIÓN
# ============================================================

precision = accuracy_score(
    y_prueba,
    predicciones
)


print("\n" + "=" * 70)
print("                    RESULTADOS")
print("=" * 70)


print(
    f"\nPrecisión general: "
    f"{precision * 100:.2f}%"
)


# ============================================================
# REPORTE DE CLASIFICACIÓN
# ============================================================

print("\nReporte de clasificación:\n")


print(
    classification_report(
        y_prueba,
        predicciones,
        zero_division=0
    )
)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

categorias = [
    "sirena",
    "incendio",
    "sismica",
    "desconocido"
]


matriz = confusion_matrix(
    y_prueba,
    predicciones,
    labels=categorias
)


print("\nMatriz de confusión:")

print(matriz)


# ============================================================
# CREAR GRÁFICA
# ============================================================

plt.figure(
    figsize=(8, 6)
)


sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    xticklabels=categorias,
    yticklabels=categorias
)


plt.xlabel(
    "Categoría predicha"
)


plt.ylabel(
    "Categoría real"
)


plt.title(
    "Matriz de confusión - SIRAH"
)


plt.tight_layout()


plt.savefig(
    RUTA_GRAFICA,
    dpi=300
)


plt.show()


print(
    f"\n✓ Matriz guardada en:"
)

print(
    RUTA_GRAFICA
)


# ============================================================
# GUARDAR MODELO
# ============================================================

print("\nGuardando modelo...")


joblib.dump(
    modelo,
    RUTA_MODELO
)


print(
    f"\n✓ Modelo guardado correctamente:"
)

print(
    RUTA_MODELO
)


# ============================================================
# GUARDAR INFORMACIÓN DEL MODELO
# ============================================================

informacion_modelo = {

    "caracteristicas": columnas_caracteristicas,

    "categorias": categorias,

    "numero_arboles": NUM_ARBOLES,

    "semilla": SEMILLA,

    "fragmentos_entrenamiento":
        int(len(X_entrenamiento)),

    "fragmentos_prueba":
        int(len(X_prueba)),

    "audios_entrenamiento":
        int(grupos_entrenamiento.nunique()),

    "audios_prueba":
        int(grupos_prueba.nunique()),

    "precision":
        float(precision)
}


RUTA_INFO = os.path.join(
    CARPETA_MODELO,
    "informacion_modelo.json"
)


with open(
    RUTA_INFO,
    "w",
    encoding="utf-8"
) as archivo:

    json.dump(
        informacion_modelo,
        archivo,
        indent=4,
        ensure_ascii=False
    )


print(
    f"\n✓ Información del modelo guardada:"
)

print(
    RUTA_INFO
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("             ENTRENAMIENTO TERMINADO")
print("=" * 70)

print(
    "\nArchivos generados:"
)

print(
    f"  ✓ {RUTA_MODELO}"
)

print(
    f"  ✓ {RUTA_GRAFICA}"
)

print(
    f"  ✓ {RUTA_INFO}"
)

print("\nSIRAH ya tiene su primer modelo entrenado.")