import os
import cv2

def load_images_from_folder(folder_path):
    images = []
    labels = []  # Şimdilik boş bırakıyoruz; gelecekte etiket eklenebilir
    for filename in os.listdir(folder_path):
        if filename.endswith('.BMP'):  # Parmak izi dosya formatı
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(None)  # Etiket yerine şimdilik 'None' ekliyoruz
    return images, labels


# Test için
if __name__ == "__main__":
    folder_path = r"C:\Users\Emre\Desktop\SOCOFing\Real"
    images = load_images_from_folder(folder_path)
    print(f"Yüklenen görüntü sayısı: {len(images)}")
