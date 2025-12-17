import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

# ======================================================
# 1. FUNGSI DASAR & UTILITAS
# ======================================================

def load_image(path, limit=300):
    """Membaca gambar dan resize jika terlalu besar."""
    img = cv2.imread(path)
    if img is None:
        print(f"WARNING: Tidak dapat membaca file {path}")
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
    """Menghitung Mean Squared Error."""
    # Pastikan ukuran sama
    if imgA.shape != imgB.shape:
        imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))
    
    # Jika grayscale vs RGB, konversi salah satu (safety check)
    if imgA.ndim == 3 and imgB.ndim == 2:
        imgB = cv2.cvtColor(imgB, cv2.COLOR_GRAY2RGB)
    elif imgA.ndim == 2 and imgB.ndim == 3:
        imgA = cv2.cvtColor(imgA, cv2.COLOR_GRAY2RGB)
        
    return np.mean((imgA.astype(float) - imgB.astype(float)) ** 2)

# ======================================================
# 2. NOISE GENERATORS
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
# 3. FILTERING MANUAL (DIPERBAIKI: SAFE FOR GRAYSCALE)
# ======================================================

def apply_filter_logic(img, mode, size):
    """Fungsi pembantu untuk menangani logika dimensi (2D/3D)."""
    pad = size // 2
    h, w = img.shape[:2]
    
    # Normalisasi ke 3D sementara agar logic loop sama
    if img.ndim == 2:
        img_proc = img[:, :, None]
        channels = 1
    else:
        img_proc = img
        channels = 3
        
    padded = np.pad(img_proc, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    result = np.zeros_like(img_proc)

    for y in range(h):
        for x in range(w):
            for c in range(channels):
                window = padded[y:y+size, x:x+size, c]
                
                if mode == 'min':
                    val = np.min(window)
                elif mode == 'max':
                    val = np.max(window)
                elif mode == 'mean':
                    val = np.mean(window)
                elif mode == 'median':
                    val = np.median(window)
                
                result[y, x, c] = val

    # Kembalikan ke bentuk asal
    if img.ndim == 2:
        return result[:, :, 0]
    return result

# Wrapper Functions agar bisa dipanggil terpisah
def mean_filter(img, size=3): return apply_filter_logic(img, 'mean', size)
def median_filter(img, size=3): return apply_filter_logic(img, 'median', size)
def min_filter(img, size=3): return apply_filter_logic(img, 'min', size)
def max_filter(img, size=3): return apply_filter_logic(img, 'max', size)

# ======================================================
# 4. MAIN PROGRAM
# ======================================================

print("\n--- MEMUAT GAMBAR ---")
# Pastikan nama file gambar ada di folder yang sama
land = load_image("Pemandangan.jpg") 
port = load_image("Pemandangan.jpg") # Sesuaikan jika ada gambar portrait beda

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

# --- GENERATE NOISE ---
print("--- MEMBUAT VARIASI NOISE ---")
SP1, SP2 = 0.02, 0.10
GS1, GS2 = 10, 40

def create_noise_pack(label, func, *params):
    pack = []
    for name, img in image_sets["Original"]:
        pack.append((name, func(img, *params))) 
    return pack

image_sets["SP1"] = create_noise_pack("SP-1", apply_salt_pepper, SP1)
image_sets["SP2"] = create_noise_pack("SP-2", apply_salt_pepper, SP2)
image_sets["GS1"] = create_noise_pack("GS-1", apply_gaussian, 0, GS1)
image_sets["GS2"] = create_noise_pack("GS-2", apply_gaussian, 0, GS2)

# --- APPLY FILTERS ---
filter_functions = {
    "min": min_filter,
    "max": max_filter,
    "median": median_filter,
    "mean": mean_filter
}

noise_groups = ["SP1", "SP2", "GS1", "GS2"]
filtered_results = {}
mse_data_list = [] 

print("\n--- PROSES FILTER MANUAL (Mohon Tunggu) ---\n")

for ng in noise_groups:
    filtered_results[ng] = {}
    for f_name, f_func in filter_functions.items():
        group_out = []
        for name, noisy in image_sets[ng]:
            # Cari gambar original yang sesuai
            if "Landscape Gray" in name: ori = land_g
            elif "Portrait Gray" in name: ori = port_g
            elif "Landscape" in name: ori = land
            else: ori = port
            
            # PROSES FILTER (Sekarang aman untuk Grayscale)
            filtered = f_func(noisy) 
            score = mse(ori, filtered)
            
            group_out.append((f"{name} | {ng} | {f_name}\n(MSE={score:.2f})", filtered))
            mse_data_list.append({
                "Noise Type": ng,
                "Image Name": name,
                "Filter": f_name.upper(),
                "MSE": score,
                "Category": "Grayscale" if "Gray" in name else "RGB"
            })
        filtered_results[ng][f_name] = group_out

# ======================================================
# 5. VISUALISASI DUA GRAFIK BESAR (RGB & GRAYSCALE)
# ======================================================

def generate_two_charts(data):
    df = pd.DataFrame(data)
    
    # Buat kolom Label gabungan (Noise + Filter) untuk X-Axis
    df['Label'] = df['Noise Type'] + "\n" + df['Filter']
    
    # Pisahkan data RGB dan Grayscale
    df_rgb = df[df['Category'] == "RGB"]
    df_gray = df[df['Category'] == "Grayscale"]
    
    # Fungsi pembantu plotting
    def plot_grouped_bar(dataframe, title):
        if dataframe.empty: return

        # Pivot data: Index=Label, Kolom=NamaGambar, Isi=MSE
        pivot_df = dataframe.pivot_table(index='Label', columns='Image Name', values='MSE')
        
        # Urutkan index agar rapi (SP1 dulu, baru SP2, dst)
        # Kita buat custom sort order
        desired_order = []
        for n in ["SP1", "SP2", "GS1", "GS2"]:
            for f in ["MIN", "MAX", "MEDIAN", "MEAN"]:
                desired_order.append(f"{n}\n{f}")
        
        # Filter hanya label yang ada di data
        existing_order = [x for x in desired_order if x in pivot_df.index]
        pivot_df = pivot_df.reindex(existing_order)
        
        # Plotting
        ax = pivot_df.plot(kind='bar', figsize=(15, 8), width=0.8, edgecolor='black', alpha=0.8)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel("Nilai MSE (Semakin Rendah Semakin Baik)", fontsize=12)
        plt.xlabel("Jenis Noise & Filter", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(title="Citra", loc='upper right')
        
        # Tambahkan label angka di atas bar
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', fontsize=9, padding=3, rotation=90)
            
        plt.tight_layout()
        plt.show()

    # 1. TAMPILKAN TABEL DATA DULU
    print("\n" + "="*60)
    print("TABEL DATA MSE LENGKAP")
    print("="*60)
    print(df.pivot_table(index=['Noise Type', 'Filter'], columns='Image Name', values='MSE'))

    # 2. PLOT GRAFIK 1: RGB
    print("\nMenampilkan Grafik 1: Perbandingan MSE untuk Citra RGB...")
    plot_grouped_bar(df_rgb, "Perbandingan MSE Filter (Kategori RGB)")
    
    # 3. PLOT GRAFIK 2: GRAYSCALE
    print("\nMenampilkan Grafik 2: Perbandingan MSE untuk Citra Grayscale...")
    plot_grouped_bar(df_gray, "Perbandingan MSE Filter (Kategori Grayscale)")

def show_series(data, title):
    n = len(data)
    cols = 2
    rows = math.ceil(n / cols)
    plt.figure(figsize=(10, rows * 4))
    plt.suptitle(title, fontsize=14, fontweight="bold")
    for i, (label, img) in enumerate(data):
        plt.subplot(rows, cols, i + 1)
        if img.ndim == 2: plt.imshow(img, cmap="gray")
        else: plt.imshow(img)
        plt.title(label, fontsize=10)
        plt.xticks([]); plt.yticks([])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- EKSEKUSI ---
print("Menghasilkan Laporan Analisis...")
generate_two_charts(mse_data_list)

# (Opsional) Tampilkan Gambar Hasil Filter
# Uncomment baris di bawah ini jika ingin melihat gambar hasil filter
# print("Menampilkan Gambar Hasil...")
# for group in noise_groups:
#     for f in ["median"]: # Contoh: hanya menampilkan hasil median biar tidak kebanyakan
#         show_series(filtered_results[group][f], f"Hasil Filter {f.upper()} pada {group}")

print("\nSelesai.")
