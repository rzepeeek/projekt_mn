import torch
import torch.nn as nn
from torchvision import models

# Sprawdzenie dostępności karty graficznej (GPU) dla przyspieszenia obliczeń
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używam urządzenia: {DEVICE}")

def run_comparison():
    # Lista architektur do przetestowania
    names = ['resnet18', 'alexnet', 'vgg11', 'mobilenet_v2', 'squeezenet1_0']
    num_classes = 8 # Twoja liczba klas: car, bus, km, syg, znak, os, ro, za

    for name in names:
        # Pobranie pre-trenowanego modelu z domyślnymi wagami ImageNet
        m = getattr(models, name)(weights='DEFAULT')

        # --- DOSTOSOWANIE WARSTWY WYJŚCIOWEJ (Fine-tuning) ---
        # Każdy model ma inną strukturę końcową, musimy podmienić ostatni klasyfikator
        
        if name == 'squeezenet1_0':
            # SqueezeNet używa warstwy konwolucyjnej jako klasyfikatora
            m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
            m.num_classes = num_classes
            
        elif hasattr(m, 'fc'):
            # Modele typu ResNet mają warstwę wyjściową o nazwie 'fc' (fully connected)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            
        elif hasattr(m, 'classifier') and isinstance(m.classifier, nn.Sequential):
            # Modele typu VGG czy AlexNet mają blok 'classifier'. 
            # Podmieniamy ostatni element (-1) w sekwencji.
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
            
        else:
            # Specjalna obsługa dla MobileNet i podobnych, gdzie struktura jest inna
            m.classifier = nn.Sequential(
                nn.Dropout(p=0.2), # Zapobieganie overfittingowi
                nn.Linear(m.last_channel if hasattr(m, 'last_channel') else 1280, num_classes)
            )

        # Przeniesienie modelu na GPU/CPU i ustawienie w tryb ewaluacji (wyłączenie Dropout)
        m.to(DEVICE).eval()

        correct, total = 0, 0
        # Wyłączenie obliczania gradientów (oszczędność pamięci i czasu przy testowaniu)
        with torch.no_grad():
            for i, l in val_loader:
                i, l = i.to(DEVICE), l.to(DEVICE)
                outputs = m(i) # Przepuszczenie obrazu przez sieć
                _, p = torch.max(outputs, 1) # Wybranie klasy z najwyższym prawdopodobieństwem
                total += l.size(0)
                correct += (p == l).sum().item()

        # Wyświetlenie wyników porównania
        print(f"Model {name:15} | Celność: {100*correct/total:.2f}%")

# Uruchomienie porównania modeli
run_comparison()