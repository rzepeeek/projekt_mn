# --- Sekcja 8: Profesjonalna Autodetekcja z Estetycznym Renderowaniem ---

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. Załadowanie modelu Faster R-CNN v2 - jedna z najlepszych architektur do detekcji obiektów
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model_det = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
model_det.to(DEVICE).eval()

# 2. Definicja palety kolorów RGB dla ułatwienia rozróżniania klas na obrazie
COLORS = [
    (255, 0, 0),   # Czerwony
    (0, 255, 0),   # Zielony
    (0, 0, 255),   # Niebieski
    (255, 255, 0), # Żółty
    (255, 0, 255), # Magenta
    (0, 255, 255), # Cyjan
    (255, 165, 0), # Pomarańczowy
    (255, 255, 255)# Biały
]

def auto_detekcja_full_pro(dataset, index=None):
    """
    Funkcja wykonuje pełną detekcję obiektów i renderuje wyniki 
    z wysoką dbałością o detale wizualne (kontrastowe etykiety).
    """
    if index is None:
        index = np.random.randint(0, len(dataset))

    # Pobranie ścieżki i otwarcie oryginalnego obrazu
    img_name = dataset.img_names[index]
    img_path = os.path.join(dataset.img_dir, img_name)
    img_pil = Image.open(img_path).convert("RGB")

    # Przygotowanie danych dla modelu (Konwersja na Tensor i wysłanie na GPU/CPU)
    img_tensor = F.to_tensor(img_pil).to(DEVICE)
    with torch.no_grad():
        prediction = model_det([img_tensor])

    # Inicjalizacja narzędzia do rysowania po obrazie PIL
    draw = ImageDraw.Draw(img_pil)
    pred = prediction[0]

    # Pobranie nazw kategorii (np. 'person', 'car', 'dog') z metadanych modelu
    categories = weights.meta["categories"]

    print(f"--- Znaleziono na zdjęciu {img_name}: ---")

    # Iteracja po wszystkich wykrytych ramkach, etykietach i wynikach pewności
    for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        if score > 0.5: # Wyświetlamy tylko detekcje, gdzie AI ma min. 50% pewności
            b = box.cpu().numpy()
            label_id = label.item()
            label_name = categories[label_id]

            # Wybór koloru (modulo zapewnia, że nie wyjdziemy poza zakres listy COLORS)
            color = COLORS[label_id % len(COLORS)]

            # Rysowanie głównej ramki obiektu
            draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline=color, width=5)

            # Przygotowanie tekstu etykiety
            text = f"{label_name} {score:.2f}"

            # Logika rysowania tła pod tekst dla maksymalnej czytelności:
            # Tworzymy prostokąt wypełniony kolorem klasy nad górną krawędzią ramki
            draw.rectangle([(b[0], b[1]-20), (b[0]+len(text)*10, b[1])], fill=color)
            
            # Dobór koloru czcionki (czarny dla jasnych teł, biały dla ciemnych)
            text_color = (0,0,0) if sum(color) > 400 else (255,255,255)
            draw.text((b[0]+5, b[1]-18), text, fill=text_color)

            print(f"Wykryto: {label_name} ({score:.2%})")

    # Finalne wyświetlenie przetworzonego obrazu za pomocą Matplotlib
    plt.figure(figsize=(12, 10))
    plt.imshow(img_pil)
    plt.title(f"Autonomiczna Detekcja Wieloklasowa\nPlik: {img_name}", fontsize=15)
    plt.axis('off')
    plt.show()

# Wywołanie testowe
auto_detekcja_full_pro(dataset)