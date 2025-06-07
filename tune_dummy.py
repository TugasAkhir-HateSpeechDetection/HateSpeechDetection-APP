import time
import json
import random
import re, os


print("🚀 [INFO] Memulai tune_dummy.py...")

def run_tuning():
    print("🧪 [DUMMY] Menjalankan tuning dummy...")

    results = []

    for i in range(3):  # 3 dummy iterations
        time.sleep(0.5)  # delay kecil agar terasa 'nyata'
        params = {
            'epochs': random.choice([5, 10]),
            'units': random.choice([8, 16]),
            'learning_rate': random.choice([0.01, 0.001]),
            'batch_size': random.choice([32, 64])
        }

        train_acc = round(random.uniform(0.7, 0.9), 4)
        val_acc = round(random.uniform(0.6, 0.85), 4)

        print(f"🧪 [ITERASI {i+1}] Dummy Val Acc: {val_acc}")

        results.append({
            'iteration': i + 1,
            'params': params,
            'train_acc': train_acc,
            'val_acc': val_acc
        })

    # Simpan ke file
    save_directory = "./tuneresult/"
    file_name = "Bi-GRU.pt"
    os.makedirs(save_directory, exist_ok=True)

    save_path = os.path.join(save_directory, file_name)
    output_path = os.path.join(save_directory, "tuning_results.json")

    with open(output_path, 'w') as f:
        json.dump({"results": results}, f, indent=4)

    print(f"💾 [SAVED] Hasil tuning disimpan ke {output_path}")
    print("🎉 [DONE] Semua iterasi selesai.")
    return results

if __name__ == "__main__":
    run_tuning()
