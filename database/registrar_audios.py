import sqlite3
import os
import librosa


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

# Obtiene la carpeta principal del proyecto:
# D:\Trabajos de python\PythonProject
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ruta de la base de datos
RUTA_DB = os.path.join(
    BASE_DIR,
    "database",
    "thaps.db"
)

# Ruta de los audios normalizados
DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "normalizado"
)

# Categorías que vamos a registrar
CATEGORIAS = [
    "sirena",
    "incendio",
    "sismica",
    "desconocido"
]


# ============================================================
# ENCABEZADO
# ============================================================

print("===================================")
print("       REGISTRO DE AUDIOS")
print("===================================")

print(f"\nProyecto:")
print(BASE_DIR)

print(f"\nDataset:")
print(DATASET_DIR)

print(f"\nBase de datos:")
print(RUTA_DB)


# ============================================================
# COMPROBAR QUE EXISTE LA BASE DE DATOS
# ============================================================

if not os.path.exists(RUTA_DB):
    print("\n[ERROR] No existe la base de datos.")
    print("Primero ejecuta:")
    print("database/crear_base_de_datos.py")
    exit()


# ============================================================
# CONEXIÓN CON SQLITE
# ============================================================

conexion = sqlite3.connect(RUTA_DB)
cursor = conexion.cursor()


# ============================================================
# COMPROBAR QUE EXISTEN LAS TABLAS
# ============================================================

cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table'
""")

tablas = [fila[0] for fila in cursor.fetchall()]

if "categorias" not in tablas:
    print("\n[ERROR] La tabla 'categorias' no existe.")
    print("Primero ejecuta:")
    print("database/crear_base_de_datos.py")

    conexion.close()
    exit()


if "sonidos" not in tablas:
    print("\n[ERROR] La tabla 'sonidos' no existe.")
    print("Primero ejecuta:")
    print("database/crear_base_de_datos.py")

    conexion.close()
    exit()


# ============================================================
# REGISTRO DE AUDIOS
# ============================================================

total = 0


for categoria in CATEGORIAS:

    print("\n-----------------------------------")
    print(f"CATEGORÍA: {categoria}")
    print("-----------------------------------")

    # Ruta de la carpeta de la categoría
    carpeta = os.path.join(
        DATASET_DIR,
        categoria
    )

    print(f"Ruta: {carpeta}")

    # Comprobar que existe la carpeta
    if not os.path.exists(carpeta):

        print(f"[ERROR] No existe: {carpeta}")

        continue

    # Obtener archivos WAV
    archivos = [
        archivo
        for archivo in os.listdir(carpeta)
        if archivo.lower().endswith(".wav")
    ]

    print(f"Audios encontrados: {len(archivos)}")

    # Buscar el ID de la categoría en SQLite
    cursor.execute("""
        SELECT id
        FROM categorias
        WHERE nombre = ?
    """, (categoria,))

    resultado = cursor.fetchone()

    if resultado is None:

        print(
            f"[ERROR] La categoría "
            f"'{categoria}' no existe en la base de datos."
        )

        continue

    categoria_id = resultado[0]

    print(f"ID de categoría: {categoria_id}")


    # ========================================================
    # PROCESAR CADA AUDIO
    # ========================================================

    for archivo in archivos:

        ruta = os.path.join(
            carpeta,
            archivo
        )

        try:

            # Cargar audio normalizado
            audio, frecuencia = librosa.load(
                ruta,
                sr=None,
                mono=True
            )

            # Calcular duración
            duracion = len(audio) / frecuencia


            # Comprobar si ya está registrado
            cursor.execute("""
                SELECT id
                FROM sonidos
                WHERE archivo = ?
            """, (ruta,))

            existe = cursor.fetchone()


            if existe:

                print(
                    f"  - Ya existe: {archivo}"
                )

                continue


            # Insertar audio en la base de datos
            cursor.execute("""
                INSERT INTO sonidos
                (
                    nombre,
                    categoria_id,
                    archivo,
                    duracion,
                    frecuencia
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                archivo,
                categoria_id,
                ruta,
                duracion,
                frecuencia
            ))


            print(
                f"  ✓ {archivo} | "
                f"{duracion:.2f}s | "
                f"{frecuencia} Hz"
            )

            total += 1


        except Exception as error:

            print(
                f"  ✗ ERROR en {archivo}: "
                f"{error}"
            )


# ============================================================
# GUARDAR CAMBIOS
# ============================================================

conexion.commit()


# ============================================================
# MOSTRAR RESULTADO
# ============================================================

print("\n===================================")
print(f"   AUDIOS REGISTRADOS: {total}")
print("===================================")


# ============================================================
# MOSTRAR TODOS LOS AUDIOS DE LA BD
# ============================================================

cursor.execute("""
    SELECT
        sonidos.id,
        sonidos.nombre,
        categorias.nombre,
        sonidos.duracion,
        sonidos.frecuencia

    FROM sonidos

    INNER JOIN categorias
        ON sonidos.categoria_id = categorias.id

    ORDER BY sonidos.id
""")

resultados = cursor.fetchall()


print("\nAUDIOS EN LA BASE DE DATOS:")
print()


for audio in resultados:

    print(
        f"ID: {audio[0]} | "
        f"{audio[1]} | "
        f"Categoría: {audio[2]} | "
        f"Duración: {audio[3]:.2f}s | "
        f"Frecuencia: {audio[4]} Hz"
    )


# ============================================================
# RESUMEN POR CATEGORÍA
# ============================================================

print("\n===================================")
print("       RESUMEN POR CATEGORÍA")
print("===================================")

for categoria in CATEGORIAS:

    cursor.execute("""
        SELECT COUNT(*)
        FROM sonidos
        INNER JOIN categorias
            ON sonidos.categoria_id = categorias.id
        WHERE categorias.nombre = ?
    """, (categoria,))

    cantidad = cursor.fetchone()[0]

    print(
        f"{categoria}: {cantidad} audios"
    )


# ============================================================
# CERRAR BASE DE DATOS
# ============================================================

conexion.close()


print("\n===================================")
print("          PROCESO TERMINADO")
print("===================================")