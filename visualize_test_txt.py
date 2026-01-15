# --- Sekcja 5: Wizualizacja danych i weryfikacja ramek (Bounding Boxes) ---

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from torchvision import transforms

# Definicja uproszczonej transformacji (tylko zmiana rozmiaru i zamiana na Tensor)
# Nie używamy tutaj Normalizacji, aby kolory na podglądzie były naturalne
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Inicjalizacja datasetu do celów wizualizacji
dataset = ProjectDataset(IMG_DIR, LABEL_DIR, transform=simple_transform)

def pokaz_przyklad(idx):
    """
    Funkcja rysująca obraz wraz z nałożonymi ramkami otaczającymi (BBoxes).
    """
    # Pobranie danych z datasetu (Tensor obrazu oraz indeks klasy)
    img_tensor, label_idx = dataset[idx]
    img_name = dataset.img_names[idx]
    label_path = os.path.join(dataset.label_dir, os.path.splitext(img_name)[0] + ".txt")

    # Konwersja Tensora (C, H, W) na format NumPy (H, W, C) akceptowany przez Matplotlib
    img_display = img_tensor.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_display)

    # Logika odczytu i rysowania ramek z pliku adnotacji
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    # Rozpakowanie formatu YOLO: class_id, x_center, y_center, width, height
                    # Wartości w plikach YOLO są znormalizowane (zakres 0-1)
                    c, xc, yc, w, h = map(float, parts)

                    # Przeliczenie współrzędnych znormalizowanych na piksele (dla rozmiaru 224x224)
                    # Formuła zamiany x_center na x_min (lewy górny róg ramki)
                    x1 = (xc - w/2) * 224
                    y1 = (yc - h/2) * 224
                    rect_w = w * 224
                    rect_h = h * 224

                    # Stworzenie obiektu ramki (Rectangle)
                    rect = patches.Rectangle((x1, y1), rect_w, rect_h, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    
                    # Dodanie etykiety tekstowej nad ramką
                    ax.text(x1, y1, dataset.classes[int(c)], color='white', backgroundcolor='red', fontsize=12)

    plt.title(f"Podgląd: {img_name} | Główna klasa: {dataset.classes[label_idx]}")
    plt.axis('off') # Ukrycie osi współrzędnych
    plt.show()

# Uruchomienie podglądu dla pierwszego elementu w zbiorze
if len(dataset) > 0:
    pokaz_przyklad(0)
else:
    print("Dataset jest pusty! Sprawdź ścieżki do folderów.")