import os
import librosa
import soundfile as sf

# Carpeta original
DATASET_DIR = "dataset"

# Carpeta donde guardaremos los audios procesados
SALIDA_DIR = os.path.join(DATASET_DIR, "normalizado")

# Categorías
CATEGORIAS = [
    "sirena",
    "incendio",
    "sismica",
    "desconocido"
]

# Frecuencia que utilizaremos para todos los audios
SR_OBJETIVO = 16000

print("===================================")
print("       NORMALIZACIÓN DE AUDIOS")
print("===================================")

total = 0

for categoria in CATEGORIAS:

    carpeta_entrada = os.path.join(DATASET_DIR, categoria)
    carpeta_salida = os.path.join(SALIDA_DIR, categoria)

    os.makedirs(carpeta_salida, exist_ok=True)

    archivos = [
        archivo for archivo in os.listdir(carpeta_entrada)
        if archivo.lower().endswith(".wav")
    ]

    print(f"\nCategoría: {categoria}")
    print(f"Audios encontrados: {len(archivos)}")

    for archivo in archivos:

        ruta_entrada = os.path.join(carpeta_entrada, archivo)
        ruta_salida = os.path.join(carpeta_salida, archivo)

        try:
            # Cargar audio y convertirlo a mono
            audio, frecuencia = librosa.load(
                ruta_entrada,
                sr=SR_OBJETIVO,
                mono=True
            )

            # Guardar audio normalizado
            sf.write(
                ruta_salida,
                audio,
                SR_OBJETIVO
            )

            duracion = len(audio) / SR_OBJETIVO

            print(
                f"  ✓ {archivo} | "
                f"Duración: {duracion:.2f}s | "
                f"Frecuencia: {SR_OBJETIVO} Hz | "
                f"Mono"
            )

            total += 1

        except Exception as e:
            print(f"  ✗ ERROR en {archivo}: {e}")

print("\n===================================")
print(f"TOTAL PROCESADOS: {total}")
print("===================================")