import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

def train_model(features, labels):
    """
    Verilen özellikler ve etiketlerle makine öğrenimi modelini eğitir.
    features (list): Öznitelikler.
        labels (list): Etiketler (0: Kadın, 1: Erkek).
    """
    # Veri setini eğitim ve test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Modeli tanımla (random forest modeli) ve eğit
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test verisi ile tahmin yap
    predictions = model.predict(X_test)
    y_pred = model.predict(X_test)

    

    

    # Başarı oranını ve raporu yazdır
    accuracy = accuracy_score(y_test, predictions)
    print("Doğruluk Oranı:", accuracy)
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, predictions))
    print(f"Model Doğruluğu: {accuracy * 100:.2f}%")

# Karışıklık Matrisini Hesapla ve Yazdır
    cm = confusion_matrix(y_test, predictions)
    print("\nKarışıklık Matrisi:")
    print(cm)

    return model
