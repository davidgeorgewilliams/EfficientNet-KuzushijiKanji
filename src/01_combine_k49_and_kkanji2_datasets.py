import os
import numpy as np
import pandas as pd
from PIL import Image
import shutil
from tqdm import tqdm
import time
import hashlib


def load_npz(file_path):
    with np.load(file_path) as data:
        return data['arr_0']


def resize_and_save(img, save_path):
    img_pil = Image.fromarray(img.squeeze(), mode='L')
    img_resized = img_pil.resize((64, 64), Image.LANCZOS)
    img_resized.save(save_path)


def load_label_mapping(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df['index'], df['codepoint']))


def generate_unique_filename():
    unique_string = f"{time.time()}\t{np.random.randint(int(1e10))}"
    return hashlib.md5(unique_string.encode()).hexdigest() + ".png"


def process_k49(imgs, labels, output_dir, prefix, label_mapping):
    for img, label in tqdm(zip(imgs, labels), total=len(imgs), desc=f"Processing {prefix}"):
        codepoint = label_mapping[label]
        class_dir = os.path.join(output_dir, codepoint)
        os.makedirs(class_dir, exist_ok=True)

        img_name = generate_unique_filename()
        save_path = os.path.join(str(class_dir), img_name)
        resize_and_save(img, save_path)


def process_kkanji(kkanji_dir, output_dir):
    for root, _, files in tqdm(os.walk(kkanji_dir), desc="Processing KKanji"):
        for file in files:
            if file.endswith('.png'):
                src_path = os.path.join(root, file)
                class_name = os.path.basename(root)
                dst_dir = os.path.join(output_dir, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(str(dst_dir), file)
                shutil.copy2(str(src_path), str(dst_path))


def main():
    # Define paths
    k49_train_imgs = '../data/k49-train-imgs.npz'
    k49_train_labels = '../data/k49-train-labels.npz'
    k49_test_imgs = '../data/k49-test-imgs.npz'
    k49_test_labels = '../data/k49-test-labels.npz'
    k49_classmap = '../data/k49_classmap.csv'
    kkanji_dir = '../data/kkanji2'
    output_dir = '../data/kuzushiji'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load label mapping
    label_mapping = load_label_mapping(k49_classmap)

    # Process K49 dataset
    train_imgs = load_npz(k49_train_imgs)
    train_labels = load_npz(k49_train_labels)
    test_imgs = load_npz(k49_test_imgs)
    test_labels = load_npz(k49_test_labels)

    process_k49(train_imgs, train_labels, output_dir, 'train', label_mapping)
    process_k49(test_imgs, test_labels, output_dir, 'test', label_mapping)

    # Process KKanji dataset
    process_kkanji(kkanji_dir, output_dir)

    print("Dataset combination and processing complete!")


if __name__ == "__main__":
    main()