# --- Sekcja 9: Finalne zestawienie wyników i wizualizacja wydajności ---

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def final_models_comparison():
    """
    Tworzy wykres słupkowy porównujący celność różnych architektur.
    Wyróżnia wytrenowany model na tle modeli bazowych.
    """
    # 1. Przygotowanie danych (wyniki uzyskane z wcześniejszych testów)
    # Lista zawiera modele pre-trenowane z podmienioną, ale jeszcze nie nauczoną głowicą
    data = {
        'Model': ['ResNet18', 'AlexNet', 'VGG11', 'MobileNet_V2', 'SqueezeNet'],
        'Accuracy (%)': [44.44, 7.69, 5.13, 2.56, 22.22] 
    }

    # Dodanie wyników Twojego modelu po pełnym procesie uczenia (15 epok)
    data['Model'].append('ResNet18 (Trained)')
    data['Accuracy (%)'].append(98.29)

    # Tworzenie obiektu DataFrame dla łatwiejszego zarządzania danymi i sortowania
    df = pd.DataFrame(data)
    # Sortowanie od najlepszego wyniku dla lepszej czytelności wykresu
    df = df.sort_values(by='Accuracy (%)', ascending=False)

    # 2. Konfiguracja wizualna wykresu
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid") # Dodanie siatki w tle dla lepszej orientacji

    # Logika kolorowania: model wytrenowany (zielony), modele bazowe (niebieski)
    colors = ['#2ecc71' if 'Trained' in m else '#3498db' for m in df['Model']]

    # Generowanie wykresu słupkowego (horyzontalnego)
    barplot = sns.barplot(x='Accuracy (%)', y='Model', data=df, palette=colors)

    # Dodanie precyzyjnych etykiet tekstowych na końcach słupków
    for i, p in enumerate(barplot.patches):
        width = p.get_width()
        plt.text(width + 1,                          # Pozycja X (lekko za słupkiem)
                 p.get_y() + p.get_height()/2,      # Pozycja Y (środek słupka)
                 f'{width:.2f}%',                   # Tekst (formatowanie do 2 miejsc po przecinku)
                 va='center',                       # Wyśrodkowanie w pionie
                 fontsize=12, 
                 fontweight='bold')

    # Opisanie osi i nadanie tytułu
    plt.title('Porównanie skuteczności architektur sieci neuronowych', fontsize=16)
    plt.xlabel('Dokładność (Accuracy %)', fontsize=12)
    plt.ylabel('Architektura modelu', fontsize=12)
    
    # Ustawienie limitu osi X do 110, aby napisy nie wychodziły poza obszar
    plt.xlim(0, 110) 
    plt.tight_layout()
    plt.show()

    # 3. Wyświetlenie tabeli w formie tekstowej w konsoli
    print("\n--- ZESTAWIENIE PROCENTOWE SKUTECZNOŚCI ---")
    print(df.to_string(index=False))

# Uruchomienie generowania raportu
final_models_comparison()