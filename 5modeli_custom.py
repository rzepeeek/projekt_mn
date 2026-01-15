def run_comparison():
    names = ['resnet18', 'alexnet', 'vgg11', 'mobilenet_v2', 'squeezenet1_0']
    for name in names:
        m = getattr(models, name)(weights='DEFAULT')
        if hasattr(m, 'fc'): m.fc = nn.Linear(m.fc.in_features, 8)
        else: m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 8)
        
        m.to(DEVICE).eval()
        # Szybka walidacja
        correct, total = 0, 0
        with torch.no_grad():
            for i, l in val_loader:
                i, l = i.to(DEVICE), l.to(DEVICE)
                _, p = torch.max(m(i), 1)
                total += l.size(0)
                correct += (p == l).sum().item()
        print(f"Model {name:15} | Celność: {100*correct/total:.2f}%")

run_comparison()