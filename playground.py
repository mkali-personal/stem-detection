from areas import areas


# Change this filename when you want to debug a different image.
TARGET_FILENAME = "10 - slice00659.tif"
# Set to True to recompute even if this filename already exists in areas.csv.
FORCE = True


if __name__ == "__main__":
    results = areas(filename=TARGET_FILENAME, force=FORCE)
    print(results)

# Show data/figures/TARGET_FILENAME (without the .tif extension) as an image.:
import matplotlib.pyplot as plt
image_name = TARGET_FILENAME.rsplit(".", 1)[0]  # Remove .tif extension
image = plt.imread(f"data/figures/{image_name}.png")
plt.imshow(image)
plt.show()