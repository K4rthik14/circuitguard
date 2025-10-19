import os, shutil, random

base = r"C:\Users\Karthik S\Documents\Infosys-Internship\circuitguard\outputs\labeled_rois_jpeg"
train_dir = os.path.join(base, "train")
val_dir = os.path.join(base, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# detect class folders
classes = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and d not in ["train", "val"]]

for cls in classes:
    src_dir = os.path.join(base, cls)
    imgs = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(imgs)
    split = int(0.8 * len(imgs))
    train_imgs = imgs[:split]
    val_imgs = imgs[split:]

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    for img in train_imgs:
        shutil.move(os.path.join(src_dir, img), os.path.join(train_dir, cls, img))
    for img in val_imgs:
        shutil.move(os.path.join(src_dir, img), os.path.join(val_dir, cls, img))

print("✅ Dataset split complete!")
print("Train and Val folders created under:", base)
