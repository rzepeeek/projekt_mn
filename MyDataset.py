class ProjectDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        # Twoje klasy zgodnie z kolejnością
        self.classes = ["car", "bus", "km", "syg", "znak", "os", "ro", "za"]

    def __len__(self): return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")
        
        image = Image.open(img_path).convert("RGB")
        label = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.readline().split()
                if content: label = int(content[0]) # Pobiera ID klasy (0-7)

        if self.transform: image = self.transform(image)
        return image, label

# Przygotowanie danych (80% trening, 20% test)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ProjectDataset(IMG_DIR, LABEL_DIR, transform=transform)
train_set, val_set = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)