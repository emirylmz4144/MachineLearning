import numpy as np

# 3x3 Karışıklık Matrisi
confusion_matrix_3x3 = np.array([
    [50, 2, 1],  # Gerçek Kedi
    [10, 45, 5], # Gerçek Köpek
    [3, 5, 40]   # Gerçek Tavşan
])

# Metrik hesaplama fonksiyonu
def calculate_metrics(confusion_matrix):
    metrics = {}
    total_samples = confusion_matrix.sum()  # Toplam örnek sayısını hesapla
    for i in range(3):  # 3 sınıf için döngü
        TP = confusion_matrix[i, i]  # True Positive for class i
        FP = confusion_matrix[:, i].sum() - TP  # False Positive for class i
        FN = confusion_matrix[i, :].sum() - TP  # False Negative for class i
        TN = total_samples - (TP + FP + FN)  # True Negative for class i

        # Hassasiyet (Precision)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        # Duyarlılık (Recall veya TPR)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Özgünlük (Specificity)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        # Her sınıf için metrikleri kaydet
        metrics[f"Sınıf {i+1}"] = {
            "Doğru pozitif tahminler.": TP,
            "Yanlış pozitif tahminler": FP,
            "Yanlış negatif tahminler": FN,
            "Doğru negatif tahminler": TN,
            "Hassasiyet (Precision)": precision,
            "Duyarlılık (Recall)": recall,
            "Özgünlük (Specificity)": specificity
        }

    return metrics

# Hesaplamaları yap ve çıktıları göster
metrics = calculate_metrics(confusion_matrix_3x3)
for sınıf, değerler in metrics.items():
    print(f"{sınıf} Metrikleri:")
    for metrik, değer in değerler.items():
        print(f"  {metrik}: {değer:.2f}")
    print()
