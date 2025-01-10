import numpy as np

def save_features_to_file(features, file_path):
    """
    Öznitelikleri bir dosyaya kaydeder.
    
    Args:
        features (list): Öznitelik vektörlerinin listesi.
        file_path (str): Kaydedilecek dosyanın yolu.
    """
    np.save(file_path, np.array(features))
    print(f"Öznitelikler {file_path} dosyasına kaydedildi.")

def load_features_from_file(file_path):
    """
    Dosyadan öznitelikleri yükler.
    
    Args:
        file_path (str): Yüklenecek dosyanın yolu.
    
    Returns:
        numpy.ndarray: Öznitelik vektörleri.
    """
    features = np.load(file_path, allow_pickle=True)
    print(f"Öznitelikler {file_path} dosyasından yüklendi.")
    return features
