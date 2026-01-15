# --- Sekcja 6: Zaawansowana wizualizacja jakościowa ---

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from PIL import Image
import torch

# Konfiguracja sprzętowa
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def wyswietl_losowe_z_jakoscia(dataset):
    """
    Pobiera losowe zdjęcie z datasetu i wyświetla je w oryginalnej jakości
    wraz ze wszystkimi ramkami (bounding boxes) z pliku .txt.
    """
    if len(dataset) == 0:
        print("Dataset jest pusty!")
        return

    # Losowanie losowego indeksu ze zbioru danych
    idx = np.random.randint(0, len(dataset))

    # Pobieranie podstawowych informacji o pliku z obiektu dataset
    img_tensor, label_idx = dataset[idx]
    img_name = dataset.img_names[idx]
    img_path = os.path.join(dataset.img_dir, img_name)
    label_path = os.path.join(dataset.label_dir, os.path.splitext(img_name)[0] + ".txt")

    # Wczytanie obrazu bezpośrednio z dysku (PIL), aby zachować pełną jakość zamiast 224x224
    img_pil = Image.open(img_path).convert("RGB")
    width, height = img_pil.size # Pobranie oryginalnych wymiarów do przeliczenia ramek

    # Inicjalizacja dużego okna wykresu
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(img_pil)

    # Definicja czytelnych nazw klas i przypisanie im kolorów HEX dla estetyki
    class_names = ["car", "bus", "km", "syg", "znak", "os", "ro", "za"]
    kolory = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#FFFFFF']

    # 2. Logika nakładania ramek (BBoxes)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    # Rozpakowanie współrzędnych YOLO (wartości od 0 do 1)
                    c_id, xc, yc, w, h = map(float, parts)
                    c_id = int(c_id)

                    # Konwersja znormalizowanego formatu YOLO na piksele zdjęcia oryginalnego
                    # xc, yc to środek ramki -> musimy wyliczyć lewy górny róg (x1, y1)
                    x1 = (xc - w/2) * width
                    y1 = (yc - h/2) * height
                    rect_w = w * width
                    rect_h = h * height

                    # Dobór koloru na podstawie ID klasy
                    color = kolory[c_id] if c_id < len(kolory) else 'lime'
                    
                    # Stworzenie prostokąta z grubą krawędzią (linewidth=3)
                    rect = patches.Rectangle((x1, y1), rect_w, rect_h, linewidth=3,
                                           edgecolor=color, facecolor='none')
                    ax.add_patch(rect)

                    # Dodanie eleganckiej etykiety z tłem (bbox) nad każdą ramką
                    plt.text(x1, y1-10, class_names[c_id], color='white', weight='bold',
                             fontsize=12, bbox=dict(facecolor=color, alpha=0.8, edgecolor='none'))

    # Ustawienie tytułu i estetyki wykresu
    plt.title(f"Plik: {img_name} | Rozdzielczość: {width}x{height}", fontsize=15)
    plt.axis('off') # Ukrycie zbędnych osi liczbowych
    plt.tight_layout()
    plt.show()

# Wywołanie funkcji
wyswietl_losowe_z_jakoscia(dataset)