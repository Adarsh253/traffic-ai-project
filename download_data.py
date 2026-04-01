import os
import numpy as np
from PIL import Image

# Create folders
for split in ['train', 'test']:
    for i in range(5):
        path = f"data/{split}/{i}"
        os.makedirs(path, exist_ok=True)

# Generate dummy images
def generate_images(folder, num_images):
    for i in range(num_images):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(folder, f"{i}.jpg"))

# Generate data
for i in range(5):
    generate_images(f"data/train/{i}", 50)
    generate_images(f"data/test/{i}", 10)

print("✅ Dummy dataset created successfully!")