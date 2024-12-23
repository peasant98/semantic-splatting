import os
from tqdm import tqdm
from PIL import Image
import cv2

# Input and output directories
input_dir = 'kitchen/depths/'
output_dir = 'kitchen/depths_2/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for filename in tqdm(os.listdir(input_dir), desc="Processing frames"):        
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        # Get new dimensions for downscaling
        height, width = img.shape[:2]
        new_width, new_height = width // 2, height // 2
        
        # Resize the image
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Save the resized image
        cv2.imwrite(output_path, img_resized)

print("All images have been downscaled and saved in the 'depth_2/' folder.")
