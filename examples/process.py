import numpy as np
import time

def process_image(image: np.ndarray) -> np.ndarray:
    """Invert the image colors and takes 1 second to simulate processing time."""
    time.sleep(1)  # Simulate a time-consuming processing task
    return 255 - image  # Invert colors

# Input
images = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(5)]

# Action
start_time = time.time()
results = [process_image(img) for img in images]
end_time = time.time()

print(f"Processed {len(images)} images in {end_time - start_time:.2f} seconds.")
