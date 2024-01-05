import argparse
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", type=str, default=None, help=".npz file for images")
    parser.add_argument("--image_dir",  type=str, default=None, help="directory to save images")
    args = parser.parse_args()
    os.makedirs(args.image_dir, exist_ok=True)
    images = np.load(args.image_file)['arr_0']
    # print(images)
    for i in tqdm(range(images.shape[0])):
        image = Image.fromarray(np.uint8(images[i])).convert('RGB')
        save_path = os.path.join(args.image_dir, str(i) + '.jpg')
        image.save(save_path)
    
    
if __name__ == "__main__":
    main()