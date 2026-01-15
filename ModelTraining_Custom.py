model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, 8)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

history = {'loss': [], 'acc': []}

for epoch in range(15): # 15 epok dla stabilnych wyników
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, p = torch.max(out, 1)
        total += lbls.size(0)
        correct += (p == lbls).sum().item()
    
    history['loss'].append(running_loss/len(train_loader))
    history['acc'].append(100*correct/total)
    print(f"Epoka {epoch+1}/15 - Loss: {history['loss'][-1]:.4f}, Acc: {history['acc'][-1]:.2f}%")