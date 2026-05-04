import zipfile
import os

base_dir = r"c:\Users\swaya\OneDrive\Desktop\image caption"
dataset_dir = os.path.join(base_dir, "dataset")

os.makedirs(dataset_dir, exist_ok=True)

images_zip = os.path.join(base_dir, "Images.zip")
captions_zip = os.path.join(base_dir, "captions.txt.zip")

print("Extracting Images...")
with zipfile.ZipFile(images_zip, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(dataset_dir, "Images"))
print("Images extracted.")

print("Extracting Captions...")
with zipfile.ZipFile(captions_zip, 'r') as zip_ref:
    zip_ref.extractall(dataset_dir)
print("Captions extracted.")

print("Dataset setup complete.")
