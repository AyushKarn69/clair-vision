import json, os, shutil

# Path to the JSON result
with open("results.json", "r") as f:
    data = json.load(f)

output_dir = "sorted_output"
blur_dir = os.path.join(output_dir, "blurrry")
sharp_dir = os.path.join(output_dir, "sharrp")

os.makedirs(blur_dir, exist_ok=True)
os.makedirs(sharp_dir, exist_ok=True)

for r in data["results"]:
    src = r["input_path"]
    dst = blur_dir if r["blurry"] else sharp_dir
    try:
        shutil.copy(src, dst)
        #print(f"✅ {'BLUR' if r['blurry'] else 'SHARP'} → {os.path.basename(src)}")
    except Exception as e:
        print(f"⚠️ Error copying {src}: {e}")

print("\nSorting complete!")
print(f"Blurred images → {blur_dir}")
print(f"Sharp images → {sharp_dir}")

scores = [r['score'] for r in json.load(open('results.json'))['results']]
print("Min:", min(scores), "Max:", max(scores))

