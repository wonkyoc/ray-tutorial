# counter_works.py
import ray
import numpy as np
import time

ray.init()

# 1. Define the stateful worker as an Actor.
@ray.remote
class PixelCounter:
    def __init__(self):
        self.total_pixels = 0

    def add(self, num_pixels: int):
        self.total_pixels += num_pixels

    def get_total(self) -> int:
        return self.total_pixels

# 2. Modify the task to accept and use the actor handle.
@ray.remote
def process_image_with_actor(image: np.ndarray, counter_actor: "ActorHandle"):
    # Remotely call the actor's method to update state.
    counter_actor.add.remote(image.size)
    time.sleep(1)
    # This task doesn't need to return anything for this example.

# --- Main Script ---
images = [np.random.randint(0, 255, (10, 10, 3)) for _ in range(8)]
image_size = images[0].size

# 3. Create a single instance of the Actor.
counter = PixelCounter.remote()

# 4. Launch tasks, passing the actor handle to each one.
task_refs = [process_image_with_actor.remote(img, counter) for img in images]

# Wait for all the image processing tasks to complete.
ray.get(task_refs)

# 5. Check the final state of the actor.
expected_total = image_size * len(images)

# Call the actor's get_total() method and fetch the result.
final_total_ref = counter.get_total.remote()
final_total = ray.get(final_total_ref)

print(f"Expected total pixels: {expected_total}")
print(f"Actual total from actor: {final_total}")
assert final_total == expected_total

ray.shutdown()