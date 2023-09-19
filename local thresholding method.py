import cv2
import numpy as np
import matplotlib.pyplot as plt

region_sizes=[4,16,64]
def local_thresholding(image, num_regions):
    height, width = image.shape[:2]
    region_height = height // num_regions
    region_width = width // num_regions

    output = np.zeros_like(image)

    for i in range(num_regions):
        for j in range(num_regions):
            region = image[i * region_height:(i + 1) * region_height, j * region_width:(j + 1) * region_width]

            # Otsu'nun eşikleme yöntemi ile optimum eşik değerini belirle
            _, thresh = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            output[i * region_height:(i + 1) * region_height, j * region_width:(j + 1) * region_width] = thresh

    return output


# Girdi görüntüsünü yükle
image = cv2.imread("Rice.png", 0)  # 0 ile görüntüyü siyah beyaz olarak yüklüyoruz

# 4 bölgeli lokal eşikleme
output_4_regions = local_thresholding(image, 4)

# 16 bölgeli lokal eşikleme
output_16_regions = local_thresholding(image, 16)

# 64 bölgeli lokal eşikleme
output_64_regions = local_thresholding(image, 64)

local_thresholding_results=[output_4_regions,output_16_regions,output_64_regions]

# Bağlantılı bileşen analizi yap
total_rice_count = []
total_rice_area = []
average_rice_area = []

for i, window in enumerate(local_thresholding_results):
    # Bağlantılı bileşenleri etiketle
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(window, connectivity=8)

    # İlk bileşeni dışarıda tut (arka plan)
    num_labels -= 1

    # Bileşen analizi için toplam pirinç sayısı ve alanı hesapla
    rice_count = num_labels
    rice_area = np.sum(stats[1:, cv2.CC_STAT_AREA])

    total_rice_count.append(rice_count)
    total_rice_area.append(rice_area)

    # Ortalama pirinç alanını hesapla
    average_area = rice_area / rice_count
    average_rice_area.append(average_area)

    print(f"Lokal Eşikleme (Region Size={region_sizes[i]}): Rice Count={rice_count}, Total Rice Area={rice_area}, "
          f"Average Rice Area={average_area}")

# Grafik çizdirme
plt.figure(figsize=(10, 6))
plt.errorbar(region_sizes, total_rice_count, yerr=np.sqrt(total_rice_count), fmt='o-', label='Rice Count')
plt.errorbar(region_sizes, total_rice_area, yerr=np.sqrt(total_rice_area), fmt='o-', label='Total Rice Area')
plt.errorbar(region_sizes, average_rice_area, yerr=np.sqrt(average_rice_area), fmt='o-', label='Average Rice Area')
plt.xlabel('Region Size')
plt.ylabel('Measurements')
plt.title('Measurements vs Region Size')
plt.legend()
plt.grid(True)
plt.show()

# Sonuçları görüntüle
cv2.imshow("4 Regions", output_4_regions)
cv2.imshow("16 Regions", output_16_regions)
cv2.imshow("64 Regions", output_64_regions)
cv2.waitKey(0)
cv2.destroyAllWindows()
