import os
import random
import shutil

def split_train_val(img_dir, lbl_dir, val_ratio=0.2):
    os.makedirs(img_dir.replace("train", "val"), exist_ok=True)
    os.makedirs(lbl_dir.replace("train", "val"), exist_ok=True)

    files = os.listdir(img_dir)
    val_size = int(len(files) * val_ratio)
    val_files = random.sample(files, val_size)

    for file in val_files:
        shutil.move(os.path.join(img_dir, file), os.path.join(img_dir.replace("train", "val"), file))
        lbl_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
        shutil.move(os.path.join(lbl_dir, lbl_file), os.path.join(lbl_dir.replace("train", "val"), lbl_file))

split_train_val('indian_number_plates/images/train', 'indian_number_plates/labels/train')
