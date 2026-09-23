import os
import librosa

# Carpeta principal del dataset
DATASET_DIR = "dataset"

# Categorías que utilizaremos
CATEGORIAS = [
    "sirena",
    "incendio",
    "sismica",
    "desconocido"
]

print("===================================")
print("       REVISIÓN DEL DATASET")
print("===================================")

total = 0

for categoria in CATEGORIAS:

    carpeta = os.path.join(DATASET_DIR, categoria)

    if not os.path.exists(carpeta):
        print(f"\n[ERROR] No existe la carpeta: {carpeta}")
        continue

    archivos = [
        archivo for archivo in os.listdir(carpeta)
        if archivo.lower().endswith(".wav")
    ]

    print(f"\nCategoría: {categoria}")
    print(f"Audios encontrados: {len(archivos)}")

    for archivo in archivos:

        ruta = os.path.join(carpeta, archivo)

        try:
            audio, frecuencia = librosa.load(
                ruta,
                sr=None,
                mono=True
            )

            duracion = len(audio) / frecuencia

            print(
                f"  ✓ {archivo} | "
                f"Duración: {duracion:.2f}s | "
                f"Frecuencia: {frecuencia} Hz"
            )

            total += 1

        except Exception as e:
            print(f"  ✗ ERROR en {archivo}: {e}")

print("\n===================================")
print(f"TOTAL DE AUDIOS: {total}")
print("===================================")