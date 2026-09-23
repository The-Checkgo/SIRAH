import sqlite3
import os
import json

import librosa
import numpy as np


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUTA_DB = os.path.join(
    BASE_DIR,
    "database",
    "thaps.db"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

DURACION_FRAGMENTO = 3.0       # segundos
SOLAPAMIENTO = 0.50            # 50%

SR_ESPERADO = 16000

N_MFCC = 13

N_FFT = 2048

HOP_LENGTH = 512


# ============================================================
# CONEXIÓN A SQLITE
# ============================================================

if not os.path.exists(RUTA_DB):
    print("ERROR: No se encontró la base de datos:")
    print(RUTA_DB)
    exit()


conexion = sqlite3.connect(RUTA_DB)
cursor = conexion.cursor()


# ============================================================
# COMPROBAR TABLA
# ============================================================

cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type='table'
AND name='caracteristicas_audio'
""")

if cursor.fetchone() is None:
    print("ERROR: La tabla caracteristicas_audio no existe.")
    conexion.close()
    exit()


# ============================================================
# OBTENER AUDIOS
# ============================================================

cursor.execute("""
SELECT
    sonidos.id,
    sonidos.nombre,
    sonidos.archivo,
    sonidos.categoria_id,
    categorias.nombre
FROM sonidos
INNER JOIN categorias
ON sonidos.categoria_id = categorias.id
ORDER BY sonidos.id
""")

audios = cursor.fetchall()


print("=" * 70)
print("EXTRACCIÓN DE CARACTERÍSTICAS")
print("=" * 70)

print(f"Base de datos:")
print(RUTA_DB)

print(f"\nAudios encontrados: {len(audios)}")

print(f"\nFragmento: {DURACION_FRAGMENTO} segundos")
print(f"Solapamiento: {SOLAPAMIENTO * 100:.0f}%")
print(f"MFCC: {N_MFCC}")

print("=" * 70)


# ============================================================
# PROCESAR CADA AUDIO
# ============================================================

total_fragmentos = 0

for audio in audios:

    sonido_id = audio[0]
    nombre = audio[1]
    archivo = audio[2]
    categoria_id = audio[3]
    categoria = audio[4]

    print("\n" + "-" * 70)

    print(f"Audio: {nombre}")
    print(f"Categoría: {categoria}")
    print(f"ID: {sonido_id}")

    # --------------------------------------------------------
    # COMPROBAR ARCHIVO
    # --------------------------------------------------------

    if not os.path.exists(archivo):

        print("ERROR: No se encontró el archivo:")
        print(archivo)

        continue

    # --------------------------------------------------------
    # CARGAR AUDIO
    # --------------------------------------------------------

    try:

        y, sr = librosa.load(
            archivo,
            sr=SR_ESPERADO,
            mono=True
        )

    except Exception as error:

        print("ERROR al cargar audio:")
        print(error)

        continue

    # --------------------------------------------------------
    # DURACIÓN
    # --------------------------------------------------------

    duracion = len(y) / sr

    print(f"Duración: {duracion:.2f} segundos")
    print(f"Frecuencia de muestreo: {sr} Hz")

    # --------------------------------------------------------
    # CALCULAR TAMAÑO DEL FRAGMENTO
    # --------------------------------------------------------

    muestras_fragmento = int(
        DURACION_FRAGMENTO * sr
    )

    # --------------------------------------------------------
    # CALCULAR SALTO
    # --------------------------------------------------------

    salto = int(
        muestras_fragmento * (1 - SOLAPAMIENTO)
    )

    # --------------------------------------------------------
    # ELIMINAR CARACTERÍSTICAS ANTERIORES
    # --------------------------------------------------------

    cursor.execute("""
    DELETE FROM caracteristicas_audio
    WHERE sonido_id = ?
    """, (sonido_id,))

    # --------------------------------------------------------
    # RECORRER FRAGMENTOS
    # --------------------------------------------------------

    segmento = 0

    inicio_muestra = 0

    while inicio_muestra < len(y):

        fin_muestra = inicio_muestra + muestras_fragmento

        fragmento = y[inicio_muestra:fin_muestra]

        # ----------------------------------------------------
        # IGNORAR FRAGMENTOS DEMASIADO PEQUEÑOS
        # ----------------------------------------------------

        if len(fragmento) < muestras_fragmento * 0.5:
            break

        # ----------------------------------------------------
        # COMPLETAR CON CEROS SI ES NECESARIO
        # ----------------------------------------------------

        if len(fragmento) < muestras_fragmento:

            faltantes = muestras_fragmento - len(fragmento)

            fragmento = np.pad(
                fragmento,
                (0, faltantes)
            )

        # ----------------------------------------------------
        # TIEMPOS DEL FRAGMENTO
        # ----------------------------------------------------

        inicio_segundos = inicio_muestra / sr

        fin_segundos = min(
            (inicio_muestra + muestras_fragmento) / sr,
            duracion
        )

        # ====================================================
        # 1. ENERGÍA
        # ====================================================

        energia = float(
            np.mean(fragmento ** 2)
        )

        # ====================================================
        # 2. MFCC
        # ====================================================

        mfcc = librosa.feature.mfcc(
            y=fragmento,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        # Promedio de cada coeficiente MFCC
        mfcc_promedio = np.mean(
            mfcc,
            axis=1
        )

        # Convertir a lista para guardarlo como JSON
        mfcc_json = json.dumps(
            mfcc_promedio.tolist()
        )

        # ====================================================
        # 3. FRECUENCIA DOMINANTE
        # ====================================================

        espectro = np.abs(
            librosa.stft(
                fragmento,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
        )

        # Promedio de energía por frecuencia
        espectro_promedio = np.mean(
            espectro,
            axis=1
        )

        # Índice de la frecuencia con mayor energía
        indice_frecuencia = np.argmax(
            espectro_promedio
        )

        frecuencias = librosa.fft_frequencies(
            sr=sr,
            n_fft=N_FFT
        )

        frecuencia_dominante = float(
            frecuencias[indice_frecuencia]
        )

        # ====================================================
        # GUARDAR EN SQLITE
        # ====================================================

        cursor.execute("""
        INSERT INTO caracteristicas_audio (
            sonido_id,
            segmento,
            inicio,
            fin,
            energia,
            frecuencia_dominante,
            mfcc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            sonido_id,
            segmento,
            inicio_segundos,
            fin_segundos,
            energia,
            frecuencia_dominante,
            mfcc_json
        ))

        segmento += 1

        total_fragmentos += 1

        # ----------------------------------------------------
        # SIGUIENTE FRAGMENTO
        # ----------------------------------------------------

        inicio_muestra += salto

    print(f"Fragmentos generados: {segmento}")


# ============================================================
# GUARDAR CAMBIOS
# ============================================================

conexion.commit()


# ============================================================
# MOSTRAR RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("PROCESO TERMINADO")
print("=" * 70)

print(f"Total de fragmentos registrados: {total_fragmentos}")


# ============================================================
# CONSULTAR REGISTROS
# ============================================================

cursor.execute("""
SELECT COUNT(*)
FROM caracteristicas_audio
""")

total_bd = cursor.fetchone()[0]

print(f"Total de registros en caracteristicas_audio: {total_bd}")


# ============================================================
# RESUMEN POR CATEGORÍA
# ============================================================

cursor.execute("""
SELECT
    categorias.nombre,
    COUNT(caracteristicas_audio.id)

FROM caracteristicas_audio

INNER JOIN sonidos
ON caracteristicas_audio.sonido_id = sonidos.id

INNER JOIN categorias
ON sonidos.categoria_id = categorias.id

GROUP BY categorias.nombre

ORDER BY categorias.id
""")

resultados = cursor.fetchall()


print("\nFragmentos por categoría:")

for categoria, cantidad in resultados:

    print(
        f"  {categoria}: {cantidad}"
    )


# ============================================================
# CERRAR BASE
# ============================================================

conexion.close()

print("\nBase de datos cerrada correctamente.")