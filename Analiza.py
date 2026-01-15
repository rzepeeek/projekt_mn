model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for i, l in val_loader:
        i, l = i.to(DEVICE), l.to(DEVICE)
        _, p = torch.max(model(i), 1)
        y_true.extend(l.cpu().numpy())
        y_pred.extend(p.cpu().numpy())

# Wykresy i Macierz
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', 
            xticklabels=dataset.classes, yticklabels=dataset.classes, cmap='Blues')
plt.title('Macierz Pomyłek - Twoje Klasy')
plt.show()

print(classification_report(y_true, y_pred, target_names=dataset.classes))