from skimage.feature import hog
from skimage.feature import local_binary_pattern
import numpy as np

def extract_hog_features(image):   
    """
    HOG (Histogram of Oriented Gradients) (kenar ve doku bilgileri) özelliklerini çıkarır.
    """
    features, hog_image = hog(image,
                              orientations=9,
                              pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2),
                              visualize=True,
                              block_norm='L2-Hys')
    return features
def extract_lbp_features(image, radius=3, n_points=24):
    """
    Görüntüden LBP öznitelikleri çıkarır.
    
    Args:
        image (ndarray): Giriş görüntüsü (gri tonlama).
        radius (int): LBP yarıçapı.
        n_points (int): Çevresel piksel sayısı.
        
    Returns:
        ndarray: LBP histogramı.
    """
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= hist.sum()  # Normalize et
    return hist

def extract_basic_statistics(image):
    """
    Görüntünün temel istatistiksel özniteliklerini (parlaklık ortalaması ve standart sapması) çıkarır.
    """
    mean = np.mean(image)
    std = np.std(image)
    return [mean, std]

def extract_combined_features(image):
    """
    HOG ve temel istatistik özelliklerini birleştirir.
    """
    hog_features = extract_hog_features(image)
    basic_stats = extract_basic_statistics(image)
    combined_features = np.concatenate((hog_features, basic_stats))
    return combined_features

def extract_combined_features(image):
    """
    Görüntüden sabit boyutlu öznitelikler çıkarır.
    """
    features = []  # Çeşitli özellikleri burada birleştirin

    # Öznitelikleri sabit boyuta dönüştür
    max_feature_size = 128  # Örneğin, sabit bir boyut
    features = np.resize(features, (max_feature_size,))  # Boyutu sabitle
    return features

