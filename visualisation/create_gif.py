import os
import imageio


def create_segmentation_gif(image_dir, prefix, gif_path, duration=0.3, loop=0):
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.startswith(prefix) and f.endswith(".png")],
        key=lambda x: int(x.split("_slice_")[1].split(".")[0])
    )

    images = [imageio.imread(os.path.join(image_dir, img)) for img in image_files]
    imageio.mimsave(gif_path, images, duration=duration, loop=loop)
    print(f"GIF saved at: {gif_path}")
