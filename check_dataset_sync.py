from google.colab import drive
import os

# Montowanie Dysku Google w środowisku Colab
drive.mount('/content/drive')

# Definicja ścieżek do folderów ze zdjęciami i etykietami
IMG_DIR = '/content/drive/MyDrive/images'
LABEL_DIR = '/content/drive/MyDrive/labels'

# Tworzenie zbioru nazw plików (bez rozszerzeń) dla obrazów (.jpg, .png)
images = {os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))}

# Tworzenie zbioru nazw plików (bez rozszerzeń) dla etykiet (.txt)
labels = {os.path.splitext(f)[0] for f in os.listdir(LABEL_DIR) if f.endswith('.txt')}

# Logika porównywania zbiorów:
# Znalezienie części wspólnej (zdjęcia, które mają etykiety)
matching = images.intersection(labels)

# Znalezienie zdjęć, które nie mają odpowiadającego pliku tekstowego
missing_labels = images - labels

# Wyświetlenie statystyk zbioru danych
print(f"Liczba zdjęć: {len(images)}")
print(f"Liczba etykiet: {len(labels)}")
print(f"Zdjęcia z pasującą etykietą: {len(matching)}")

# Weryfikacja spójności danych przed rozpoczęciem uczenia maszynowego
if missing_labels:
    print(f"⚠️ Uwaga! {len(missing_labels)} zdjęć nie ma plików .txt")
    # Opcjonalnie: print(f"Brakujące pliki: {missing_labels}")
else:
    print("✅ Wszystkie zdjęcia mają swoje etykiety. Możesz zaczynać trening!")