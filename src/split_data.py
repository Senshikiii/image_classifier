import os
import shutil
import random

cat_src = "Petimages/Cat"
dog_src = "Petimages/Dog"

output_dirs = [
    "data_2/train/cat",
    "data_2/train/dog",
    "data_2/test/cat",
    "data_2/test/dog",
]

for dir in output_dirs:
    os.makedirs(dir, exist_ok=True)


cat_files = os.listdir(cat_src)
dog_files = os.listdir(dog_src)

random.shuffle(cat_files)
random.shuffle(dog_files)


def split_and_copy(files, src, train_dst, test_dst, ratio=0.8):
    files = [
        f
        for f in files
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
            )
        )
    ]
    split = int(len(files) * ratio)
    for f in files[:split]:
        shutil.copy(os.path.join(src, f), os.path.join(train_dst, f))

    for f in files[split:]:
        shutil.copy(os.path.join(src, f), os.path.join(test_dst, f))


split_and_copy(cat_files, cat_src, "data_2/train/cat", "data_2/test/cat")
split_and_copy(dog_files, dog_src, "data_2/train/dog", "data_2/test/dog")

print("Done splitting")
