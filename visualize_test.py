# --- Sekcja 7: Autonomiczna detekcja (Inference bez etykiet użytkownika) ---

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. Załadowanie zaawansowanego modelu detekcyjnego Faster R-CNN
# Model ten posiada architekturę FPN (Feature Pyramid Network) dla lepszej detekcji małych obiektów
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# box_score_thresh odrzuca na wstępie detekcje o niskim prawdopodobieństwie
model_det = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
model_det.to(DEVICE).eval()

# Mapa pomocnicza: mapowanie kategorii ze zbioru COCO na Twoje klasy projektowe
# Uwaga: Faster R-CNN rozpoznaje 91 klas, tutaj wybieramy te najistotniejsze dla transportu
CLASS_MAP = {1: "car", 6: "bus", 10: "syg", 13: "znak", 1: "km", 1: "os"}

def auto_detekcja_bez_txt(dataset, index=None):
    """
    Przeprowadza detekcję obiektów na losowym zdjęciu, ignorując pliki .txt.
    Służy do weryfikacji, czy obiekty na zdjęciach są w ogóle wykrywalne przez AI.
    """
    if index is None:
        index = np.random.randint(0, len(dataset))

    # Pobranie ścieżki do losowego zdjęcia z Twojego datasetu
    img_name = dataset.img_names[index]
    img_path = os.path.join(dataset.img_dir, img_name)
    img_pil = Image.open(img_path).convert("RGB")

    # Konwersja obrazu PIL na tensor (model Faster R-CNN wymaga wartości 0.0 - 1.0)
    img_tensor = F.to_tensor(img_pil).to(DEVICE)

    # 2. Wykonanie detekcji (Inference)
    # Model zwraca listę słowników z kluczami: 'boxes', 'labels', 'scores'
    with torch.no_grad():
        prediction = model_det([img_tensor])

    # 3. Przygotowanie do rysowania wyników na obrazie
    draw = ImageDraw.Draw(img_pil)
    pred = prediction[0]

    for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        # Wyświetlamy tylko obiekty, których model jest pewien na więcej niż 60%
        if score > 0.6: 
            b = box.cpu().numpy()
            # Rysowanie ramki (bounding box) bezpośrednio na obiekcie PIL Image
            draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="red", width=5)

            # Pobranie oryginalnej nazwy kategorii ze słownika wag modelu
            lbl = weights.meta["categories"][label.item()]
            # Naniesienie tekstu z nazwą i stopniem pewności (Confidence Score)
            draw.text((b[0], b[1]-15), f"{lbl} {score:.2f}", fill="red")

    # Wyświetlenie finalnego wyniku
    plt.figure(figsize=(12, 10))
    plt.imshow(img_pil)
    plt.title(f"Autonomiczna detekcja modelu (bez użycia plików .txt)\nPlik: {img_name}")
    plt.axis('off')
    plt.show()

# URUCHOMIENIE - testowanie modelu na surowych danych
auto_detekcja_bez_txt(dataset)