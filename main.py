from scripts.feature_extraction import extract_combined_features, extract_lbp_features, extract_hog_features
from scripts.load_images import load_images_from_folder
from scripts.feature_extraction import extract_combined_features
from scripts.save_features import save_features_to_file, load_features_from_file
from scripts.train_model import train_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Görüntüleri yükle
    folder_path = r"C:\Users\Emre\Desktop\SOCOFing\Real"
    images, _ = load_images_from_folder(folder_path)

# Farklı öznitelik çıkarma yöntemleri
    methods = {
        "combined": extract_combined_features,
        "lbp": extract_lbp_features,
        "hog": extract_hog_features
    }

    for method_name, method_func in methods.items():
        print(f"\n{method_name.upper()} yöntemi ile öznitelik çıkarma ve model eğitimi...")

    # Öznitelik çıkarma
    features_list = []
    for idx, image in enumerate(images):
        features = extract_combined_features(image)
        features_list.append(features)

    # Özniteliklerin boyutlarını kontrol et
for idx, feature in enumerate(features_list):
    print(f"Öznitelik {idx}: Boyut {len(feature)}")    

    # Öznitelikleri kaydet
    output_file = r"C:\Users\Emre\Desktop\SOCOFing\features.npy"
    save_features_to_file(features_list, output_file)

    # Özellikleri yükle
    features = load_features_from_file(output_file)

    # Etiketleri rastgele oluştur (0 = Kadın, 1 = Erkek)
    labels = np.random.randint(0, 2, len(features))

    # Modeli eğit ve test et
    model = train_model(features, labels)

    # Sabit boyut ayarla
fixed_size = 128

# Öznitelikleri sabit boyuta getir
fixed_features_list = []
for feature in features_list:
    if len(feature) > fixed_size:
        fixed_features_list.append(feature[:fixed_size])  # Kes
    elif len(feature) < fixed_size:
        fixed_features_list.append(np.pad(feature, (0, fixed_size - len(feature)), 'constant'))  # Sıfırla doldur
    else:
        fixed_features_list.append(feature)

# Sabitlenmiş listeyi kaydet
save_features_to_file(fixed_features_list, output_file)

 # Etiketleri rastgele oluştur (0 = Kadın, 1 = Erkek)
labels = np.random.randint(0, 2, len(features))

 # Modeli eğit ve doğruluk oranını test et
model = train_model(features, labels)

accuracy = train_model(features, labels)
performance_data.append({"Method": method_name, "Accuracy": accuracy * 100})

 # Performans tablosunu yazdır
performance_df = pd.DataFrame(performance_data)
print("\nPerformans Karşılaştırma Tablosu:")
print(performance_df)

# Performans grafiğini çiz
plt.figure(figsize=(10, 6))
plt.bar(performance_df["Method"], performance_df["Accuracy"], color=['blue', 'green', 'red'])
plt.title("Farklı Öznitelik Çıkarma Yöntemlerinin Performansı")
plt.xlabel("Yöntem")
plt.ylabel("Doğruluk Oranı (%)")
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

