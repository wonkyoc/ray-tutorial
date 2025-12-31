# counter_fails.py
import ray
import numpy as np
import time

ray.init()

total_pixels_processed = 0 # Our global counter.

@ray.remote
def process_image_and_count(image: np.ndarray) -> np.ndarray:
    global total_pixels_processed
    # This update is only visible inside this specific task.
    total_pixels_processed += image.size 
    print(f"Task processing {image.size} pixels. Local total: {total_pixels_processed}")
    
    time.sleep(1)
    return 255 - image

images = [np.random.randint(0, 255, (10, 10, 3)) for _ in range(4)]
image_size = images[0].size # 10*10*3 = 300 pixels per image

# Launch the tasks.
ray.get([process_image_and_count.remote(img) for img in images])

print(f"\nExpected total pixels: {image_size * 4}")
print(f"Actual total pixels in main script: {total_pixels_processed}")