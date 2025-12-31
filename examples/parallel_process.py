import ray
import numpy as np
import time

# 1. Initialize Ray. It will automatically use all your CPU cores.
ray.init()

# 2. Decorate the function to make it a remote task.
@ray.remote
def process_image(image: np.ndarray) -> np.ndarray:
    """Inverts the image colors and takes 1 second."""
    time.sleep(1)
    return 255 - image

images = [np.random.randint(0, 255, (10, 10, 3)) for _ in range(8)]

start_time = time.time()
# 3. Launch tasks in parallel with .remote().
result_refs = [process_image.remote(img) for img in images]

# 4. Retrieve the results with ray.get(). This is a blocking call.
results = ray.get(result_refs)
end_time = time.time()

print(f"Processed {len(results)} images in {end_time - start_time:.2f} seconds.")

# Clean up Ray.
ray.shutdown()