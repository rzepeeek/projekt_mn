# --- Sekcja 2: Dataset i Dataloader ---

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Definicja własnej klasy zbioru danych dziedziczącej po torch.utils.data.Dataset
class ProjectDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        # Pobranie listy plików graficznych z folderu źródłowego
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        # Definicja nazw klas (indeksy 0-7 odpowiadają kolejnym nazwom)
        self.classes = ["car", "bus", "km", "syg", "znak", "os", "ro", "za"]

    # Metoda zwracająca całkowitą liczbę próbek w zbiorze
    def __len__(self): 
        return len(self.img_names)

    # Metoda pobierająca konkretną próbkę (obraz + etykieta) o danym indeksie
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        # Tworzenie ścieżki do etykiety poprzez zamianę rozszerzenia obrazu na .txt
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        # Otwarcie obrazu i konwersja do formatu RGB (ważne przy plikach PNG/RGBA)
        image = Image.open(img_path).convert("RGB")
        
        label = 0 # Domyślna klasa (np. tło lub 'car')
        # Odczyt etykiety z pliku tekstowego
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.readline().split()
                if content: 
                    # Pobranie pierwszej wartości z pliku jako ID klasy (konwersja na int)
                    label = int(content[0]) 

        # Nałożenie transformacji (augumentacja, skalowanie, normalizacja)
        if self.transform: 
            image = self.transform(image)
            
        return image, label

# --- Przygotowanie transformacji danych (Preprocessing) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Zmiana rozmiaru do standardu sieci (np. ResNet/MobileNet)
    transforms.ToTensor(),              # Zamiana obrazu PIL na tensor (0.0 - 1.0)
    # Normalizacja ImageNet (średnia i odchylenie standardowe dla kanałów R, G, B)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Inicjalizacja całego zbioru danych
dataset = ProjectDataset(IMG_DIR, LABEL_DIR, transform=transform)

# Podział zbioru na treningowy (80%) i walidacyjny (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# Tworzenie DataLoaderów - obiektów, które w pętli dostarczają dane w paczkach (batches)
# shuffle=True dla treningu zapobiega uczeniu się kolejności danych przez sieć
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)