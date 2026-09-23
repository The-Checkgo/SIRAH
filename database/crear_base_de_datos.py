import sqlite3
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================

# Obtener la carpeta raíz del proyecto
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CARPETA_DB = os.path.join(
    BASE_DIR,
    "database"
)

RUTA_DB = os.path.join(
    CARPETA_DB,
    "thaps.db"
)

# Crear carpeta si no existe
os.makedirs(CARPETA_DB, exist_ok=True)


# ==========================================
# CONEXIÓN CON SQLITE
# ==========================================

conexion = sqlite3.connect(RUTA_DB)

cursor = conexion.cursor()

print("===================================")
print("     CREACIÓN BASE DE DATOS THAPS")
print("===================================")

print(f"\nUbicación de la base de datos:")
print(RUTA_DB)


# ==========================================
# TABLA: CATEGORIAS
# ==========================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS categorias (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL UNIQUE,
    led TEXT NOT NULL,
    vibracion TEXT NOT NULL
)
""")


# ==========================================
# TABLA: SONIDOS
# ==========================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS sonidos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    categoria_id INTEGER NOT NULL,
    archivo TEXT NOT NULL,
    duracion REAL,
    frecuencia INTEGER,
    FOREIGN KEY (categoria_id)
        REFERENCES categorias(id)
)
""")


# ==========================================
# TABLA: CARACTERISTICAS DE AUDIO
# ==========================================
# ==========================================
# TABLA: CARACTERISTICAS DE AUDIO
# ==========================================

# Comprobar si la tabla existe y si tiene la columna 'segmento'
cursor.execute("""
SELECT name FROM sqlite_master
WHERE type='table' AND name='caracteristicas_audio'
""")
tabla_existe = cursor.fetchone()

if tabla_existe:
    # Verificar columnas actuales
    cursor.execute("PRAGMA table_info(caracteristicas_audio)")
    columnas = [col[1] for col in cursor.fetchall()]

    if "segmento" not in columnas:
        print("\n⚠ La tabla 'caracteristicas_audio' tiene estructura antigua.")
        print("  Eliminando y recreando con la estructura nueva...")
        cursor.execute("DROP TABLE caracteristicas_audio")
        print("  ✓ Tabla antigua eliminada.")

cursor.execute("""
CREATE TABLE IF NOT EXISTS caracteristicas_audio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    sonido_id INTEGER NOT NULL,

    -- Número del fragmento dentro del audio
    segmento INTEGER NOT NULL,

    -- Tiempo inicial del fragmento en segundos
    inicio REAL NOT NULL,

    -- Tiempo final del fragmento en segundos
    fin REAL NOT NULL,

    -- Energía del fragmento
    energia REAL NOT NULL,

    -- Frecuencia dominante del fragmento
    frecuencia_dominante REAL NOT NULL,

    -- MFCC almacenados como JSON
    mfcc TEXT NOT NULL,

    FOREIGN KEY (sonido_id)
        REFERENCES sonidos(id)
)
""")

# ==========================================
# TABLA: DETECCIONES
# ==========================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS detecciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT NOT NULL,
    categoria_detectada TEXT NOT NULL,
    confianza REAL,
    tiempo_respuesta REAL
)
""")


# ==========================================
# INSERTAR CATEGORÍAS
# ==========================================

categorias = [
    ("sirena", "rojo", "patron_1"),
    ("incendio", "rojo", "patron_2"),
    ("sismica", "blanco", "patron_3"),
    ("desconocido", "sin_alerta", "sin_vibracion")
]

cursor.executemany("""
INSERT OR IGNORE INTO categorias
(nombre, led, vibracion)
VALUES (?, ?, ?)
""", categorias)


# ==========================================
# GUARDAR CAMBIOS
# ==========================================

conexion.commit()

print("\n✓ Base de datos creada correctamente")
print(f"✓ Ubicación: {RUTA_DB}")


# ==========================================
# MOSTRAR CATEGORÍAS
# ==========================================

cursor.execute("""
SELECT id, nombre, led, vibracion
FROM categorias
ORDER BY id
""")

resultados = cursor.fetchall()

print("\nCategorías registradas:")

for categoria in resultados:

    print(
        f"  ID: {categoria[0]} | "
        f"Categoría: {categoria[1]} | "
        f"LED: {categoria[2]} | "
        f"Vibración: {categoria[3]}"
    )


# ==========================================
# MOSTRAR TABLAS CREADAS
# ==========================================

cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type='table'
ORDER BY name
""")

tablas = cursor.fetchall()

print("\nTablas creadas:")

for tabla in tablas:
    print(f"  ✓ {tabla[0]}")


# ==========================================
# CERRAR CONEXIÓN
# ==========================================

conexion.close()

print("\n===================================")
print("          PROCESO TERMINADO")
print("===================================")