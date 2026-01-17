import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. Załadowanie modelu i wag (model ResNet50 jest wymieniony jako jeden z najskuteczniejszych w projekcie)
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model_det = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
model_det.to(DEVICE).eval()

# 2. Definicja kolorów
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 165, 0), (255, 255, 255)]

# Klasy do wykluczenia
KLASY_DO_USUNIECIA = ["airplane"] 

def auto_detekcja_full_pro(dataset, index=None):
    if index is None:
        index = np.random.randint(0, len(dataset))

    # Pobranie zdjęcia (na podstawie klasy Dataset z projektu)
    img_name = dataset.img_names[index]
    img_path = os.path.join(dataset.img_dir, img_name)
    img_pil = Image.open(img_path).convert("RGB")

    # Detekcja (klasyfikacja i wyodrębnianie obiektów)
    img_tensor = F.to_tensor(img_pil).to(DEVICE)
    with torch.no_grad():
        prediction = model_det([img_tensor])

    draw = ImageDraw.Draw(img_pil)
    pred = prediction[0]
    categories = weights.meta["categories"]

    print(f"--- Znaleziono na zdjęciu {img_name}: ---")

    for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        if score > 0.5: # Próg pewności
            label_id = label.item()
            label_name = categories[label_id]

            # 1. FILTRACJA - Sprawdzamy czy klasa nie jest na czarnej liście
            if label_name in KLASY_DO_USUNIECIA:
                continue 

            # 2. RYSOWANIE - Wykonywane tylko jeśli klasa nie została pominięta wyżej
            b = box.cpu().numpy()
            color = COLORS[label_id % len(COLORS)]

            # Rysowanie ramki
            draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline=color, width=5)

            # Napis i tło (podobnie jak w wizualizacjach projektowych)
            text = f"{label_name} {score:.2f}"
            draw.rectangle([(b[0], b[1]-20), (b[0]+len(text)*10, b[1])], fill=color)
            draw.text((b[0]+5, b[1]-18), text, fill=(0,0,0) if sum(color)>400 else (255,255,255))

            print(f"Wykryto: {label_name} ({score:.2%})")

    # Wyświetlenie wyniku (zgodnie z procedurą wizualizacji w raporcie)
    plt.figure(figsize=(12, 10))
    plt.imshow(img_pil)
    plt.title(f"Autonomiczna Detekcja\nPlik: {img_name}", fontsize=15)
    plt.axis('off')
    plt.show()

# URUCHOMIENIE
auto_detekcja_full_pro(dataset)