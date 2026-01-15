import torch.optim as optim

# Wybór konkretnego modelu do treningu (ResNet18) i załadowanie wag pre-trenowanych
model = models.resnet18(weights='DEFAULT')
# Dostosowanie ostatniej warstwy do Twoich 8 klas
model.fc = nn.Linear(model.fc.in_features, 8)
model.to(DEVICE)

# Definicja funkcji kosztu (CrossEntropyLoss jest standardem dla klasyfikacji wieloklasowej)
criterion = nn.CrossEntropyLoss()
# Wybór optymalizatora (Adam często radzi sobie lepiej i szybciej niż standardowy SGD)
# lr (learning rate) = 0.0001 to niska wartość, idealna do "dostrajania" (fine-tuning) modelu
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Słownik do przechowywania historii wyników w celu późniejszego wykreślenia wykresów
history = {'loss': [], 'acc': []}

print("Rozpoczynam proces trenowania...")

for epoch in range(15): # Pętla po epokach (całych przejściach przez zbiór danych)
    model.train() # Ustawienie modelu w tryb treningu (aktywuje Dropout i Batch Normalization)
    running_loss, correct, total = 0.0, 0, 0
    
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        
        # Zerowanie gradientów z poprzedniego kroku, aby się nie kumulowały
        optimizer.zero_grad()
        
        # 1. Forward pass: Przewidywanie wyniku przez model
        out = model(imgs)
        # 2. Obliczenie straty (różnicy między przewidywaniem a rzeczywistością)
        loss = criterion(out, lbls)
        # 3. Backward pass: Obliczenie gradientów (wsteczna propagacja)
        loss.backward()
        # 4. Aktualizacja wag modelu na podstawie obliczonych gradientów
        optimizer.step()

        # Statystyki bieżące
        running_loss += loss.item()
        _, p = torch.max(out, 1) # Wybór najbardziej prawdopodobnej klasy
        total += lbls.size(0)
        correct += (p == lbls).sum().item()

    # Zapisywanie uśrednionych wyników z danej epoki
    history['loss'].append(running_loss/len(train_loader))
    history['acc'].append(100*correct/total)
    
    print(f"Epoka {epoch+1}/15 - Loss: {history['loss'][-1]:.4f}, Acc: {history['acc'][-1]:.2f}%")