import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
from google.colab import files

# 1. Konfiguracja urządzenia
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Załadowanie modelu i wag
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model_det = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
model_det.to(DEVICE).eval()

# 3. Definicja kolorów i filtrów
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 165, 0), (255, 255, 255)]

KLASY_DO_USUNIECIA = ["airplane"] 

def detekcja_z_filtrem():
    print("Wybierz zdjęcie z komputera:")
    uploaded = files.upload()
    
    if not uploaded:
        print("Nie przesłano pliku.")
        return

    file_path = list(uploaded.keys())[0]
    img_pil = Image.open(file_path).convert("RGB")

    # Przetwarzanie obrazu
    img_tensor = F.to_tensor(img_pil).to(DEVICE)
    with torch.no_grad():
        prediction = model_det([img_tensor])

    draw = ImageDraw.Draw(img_pil)
    pred = prediction[0]
    categories = weights.meta["categories"]

    print(f"\n--- Wyniki dla: {file_path} ---")

    for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
        label_name = categories[label.item()]
        
        # FILTRACJA KLAS
        if label_name in KLASY_DO_USUNIECIA:
            continue # Pomiń tę klasę i przejdź do następnej

        if score > 0.5: 
            b = box.cpu().numpy()
            color = COLORS[label.item() % len(COLORS)]

            # Rysowanie
            draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline=color, width=5)
            text = f"{label_name} {score:.2f}"
            draw.rectangle([(b[0], b[1]-20), (b[0]+len(text)*10, b[1])], fill=color)
            draw.text((b[0]+5, b[1]-18), text, fill=(0,0,0) if sum(color)>400 else (255,255,255))
            
            print(f"Wykryto: {label_name} ({score:.2%})")

    # Pokazanie wyniku
    plt.figure(figsize=(12, 10))
    plt.imshow(img_pil)
    plt.axis('off')
    plt.show()

# URUCHOMIENIE
detekcja_z_filtrem()