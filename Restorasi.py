import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

# ======================================================
# FUNGSI DASAR UNTUK PEMROSESAN CITRA
# ======================================================

def load_image(path, limit=300):
    """
    Membaca gambar dan mengecilkan ukurannya bila terlalu besar.
    """
    img = cv2.imread(path)
    if img is None:
        print(f"WARNING: Tidak dapat membaca file {path}")
        # Return gambar hitam dummy agar program tidak crash saat dijalankan tanpa file
        return np.zeros((100, 100, 3), dtype=np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if max(h, w) > limit:
        scale = limit / max(h, w)
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        print(f"Resize {path}: {w}x{h} -> {resized.shape[1]}x{resized.shape[0]}")
        return resized

    return img


def convert_to_gray(rgb):
    """Mengubah RGB menjadi grayscale."""
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
    return gray


def mse(imgA, imgB):
    """Menghitung nilai Mean Squared Error (MSE)."""
    if imgA.shape != imgB.shape:
        imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))

    return np.mean((imgA.astype(float) - imgB.astype(float)) ** 2)


# ======================================================
# NOISE GENERATORS
# ======================================================

def apply_salt_pepper(img, ratio):
    """Menyisipkan noise salt-pepper."""
    noisy = img.copy()
    rand_map = np.random.rand(*noisy.shape[:2])

    if noisy.ndim == 3:
        noisy[rand_map < ratio / 2] = [255, 255, 255]
        noisy[rand_map > 1 - ratio / 2] = [0, 0, 0]
    else:
        noisy[rand_map < ratio / 2] = 255
        noisy[rand_map > 1 - ratio / 2] = 0

    return noisy


def apply_gaussian(img, mu, sigma):
    """Memberikan noise Gaussian."""
    h, w = img.shape[:2]

    if img.ndim == 3:
        gauss = np.random.normal(mu, sigma, (h, w, 3))
    else:
        gauss = np.random.normal(mu, sigma, (h, w))

    return np.clip(img + gauss, 0, 255).astype(np.uint8)


# ======================================================
# FILTERING MANUAL
# ======================================================

def manual_filter(img, mode, size=3):
    """
    Menjalankan filter min, max, mean, atau median secara manual.
    """
    pad = size // 2
    h, w = img.shape[:2]

    if img.ndim == 3:
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        result = np.zeros_like(img)
        channels = 3
    else:
        padded = np.pad(img, ((pad, pad), (pad, pad)), mode='edge')[:, :, None]
        result = np.zeros((h, w, 1), dtype=np.uint8)
        channels = 1

    for y in range(h):
        for x in range(w):
            for c in range(channels):
                block = padded[y:y+size, x:x+size, c]

                if mode == "min":
                    val = np.min(block)
                elif mode == "max":
                    val = np.max(block)
                elif mode == "mean":
                    val = np.mean(block)
                else:
                    val = np.median(block)

                result[y, x, c] = val

    return result[:, :, 0] if channels == 1 else result


# ======================================================
# MEMBACA GAMBAR
# ======================================================

print("\n--- MEMUAT GAMBAR ---")
# PASTIKAN NAMA FILE SESUAI DENGAN YANG ADA DI FOLDER ANDA
land = load_image("Pemandangan.jpg") 
port = load_image("Potrait.jpg") 

if land is None: land = np.zeros((100,100,3), dtype=np.uint8)
if port is None: port = np.zeros((100,100,3), dtype=np.uint8)

land_g = convert_to_gray(land)
port_g = convert_to_gray(port)

image_sets = {
    "Original": [
        ("Landscape", land),
        ("Portrait", port),
        ("Landscape Gray", land_g),
        ("Portrait Gray", port_g),
    ]
}

# ======================================================
# GENERATE NOISE
# ======================================================

print("--- MEMBUAT VARIASI NOISE ---")

SP1 = 0.02
SP2 = 0.10
GS1 = 10
GS2 = 40

def create_noise_pack(label, func, *params):
    pack = []
    for name, img in image_sets["Original"]:
        pack.append((name, func(img, *params))) 
    return pack

image_sets["SP1"] = create_noise_pack("SP-1", apply_salt_pepper, SP1)
image_sets["SP2"] = create_noise_pack("SP-2", apply_salt_pepper, SP2)
image_sets["GS1"] = create_noise_pack("GS-1", apply_gaussian, 0, GS1)
image_sets["GS2"] = create_noise_pack("GS-2", apply_gaussian, 0, GS2)

# ======================================================
# PENERAPAN FILTERS & PENGUMPULAN DATA
# ======================================================

filters = ["min", "max", "median", "mean"]
noise_groups = ["SP1", "SP2", "GS1", "GS2"]

filtered_results = {}
mse_data_list = [] 

print("\n--- PROSES FILTER MANUAL (Mohon Tunggu) ---\n")

for ng in noise_groups:
    filtered_results[ng] = {}

    for f in filters:
        group_out = []
        
        for name, noisy in image_sets[ng]:
            # Cari gambar original yang sesuai
            if "Landscape Gray" in name:
                ori = land_g
            elif "Portrait Gray" in name:
                ori = port_g
            elif "Landscape" in name:
                ori = land
            else:
                ori = port
            
            filtered = manual_filter(noisy, f)
            score = mse(ori, filtered)
            
            # Simpan hasil untuk visualisasi gambar
            group_out.append((f"{name} + {ng} + {f}\n(MSE={score:.2f})", filtered))

            # Simpan data untuk Grafik & Tabel
            mse_data_list.append({
                "Noise Type": ng,
                "Image Name": name,
                "Filter": f.capitalize(),
                "MSE": score
            })

        filtered_results[ng][f] = group_out

# ======================================================
# FUNGSI VISUALISASI GRAFIK & TABEL (MODIFIKASI: SATU PER SATU)
# ======================================================

def generate_analysis_report(data):
    df = pd.DataFrame(data)
    
    # 1. PRINT TABEL
    print("\n" + "="*50)
    print("TABEL PERBANDINGAN NILAI MSE")
    print("="*50)
    
    pivot_df = df.pivot_table(index=["Noise Type", "Image Name"], 
                              columns="Filter", 
                              values="MSE")
    
    pd.options.display.float_format = '{:.2f}'.format
    print(pivot_df)
    print("\n" + "="*50)

    # 2. GENERATE GRAFIK BATANG SATU PER SATU
    print("\nMenampilkan grafik satu per satu... (Tutup jendela grafik untuk melihat berikutnya)")
    
    noise_types = df["Noise Type"].unique()
    image_names = df["Image Name"].unique()
    
    bar_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'] 
    
    for noise in noise_types:
        for img_name in image_names:
            # Membuat Figure Baru untuk setiap kombinasi
            plt.figure(figsize=(10, 6)) # Ukuran cukup besar agar jelas
            
            subset = df[(df["Noise Type"] == noise) & (df["Image Name"] == img_name)]
            subset = subset.sort_values(by="Filter")
            
            # Plot Bar Chart
            bars = plt.bar(subset["Filter"], subset["MSE"], color=bar_colors, edgecolor='grey', width=0.6)
            
            # Tambahkan label nilai di atas batang
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5), # Jarak teks dari bar
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

            plt.title(f"Perbandingan MSE: {img_name} dengan Noise {noise}", fontweight='bold', fontsize=14)
            plt.ylabel("Nilai MSE (Lebih Rendah Lebih Baik)", fontsize=12)
            plt.xlabel("Jenis Filter", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Atur margin agar tidak terpotong
            plt.tight_layout()
            
            # Tampilkan Grafik
            plt.show() 

# ======================================================
# DISPLAY GAMBAR (VISUALISASI)
# ======================================================

def show_series(data, title):
    """Menampilkan kumpulan gambar sebagai grid."""
    n = len(data)
    cols = 2
    rows = math.ceil(n / cols)

    plt.figure(figsize=(10, rows * 4))
    plt.suptitle(title, fontsize=14, fontweight="bold")

    for i, (label, img) in enumerate(data):
        plt.subplot(rows, cols, i + 1)
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title(label, fontsize=10)
        plt.xticks([]); plt.yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- EKSEKUSI UTAMA ---

# 1. Tampilkan Grafik Analisis Terlebih Dahulu (Satu per satu)
print("Menampilkan Grafik Analisis...")
generate_analysis_report(mse_data_list)

# 2. Tampilkan Gambar Asli & Noise (Tetap grid agar ringkas)
print("Menampilkan Gambar Awal & Noise...")
show_series(image_sets["Original"], "Gambar Awal")
show_series(image_sets["SP1"], "Noise Salt-Pepper Level 1")
show_series(image_sets["SP2"], "Noise Salt-Pepper Level 2")
show_series(image_sets["GS1"], "Gaussian Noise Level 1")
show_series(image_sets["GS2"], "Gaussian Noise Level 2")

# 3. Tampilkan Hasil Filter
print("Menampilkan Hasil Filter...")
for group in noise_groups:
    for f in filters:
        show_series(filtered_results[group][f], f"Hasil Filter {f.upper()} pada {group}")

print("\nSelesai. Semua output telah ditampilkan.")