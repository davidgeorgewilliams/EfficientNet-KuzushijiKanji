from kkanji import *

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
