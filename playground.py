import matplotlib.pyplot as plt

from areas import areas

# Change this filename when you want to debug a different image.
TARGET_FILENAME = None  # "160 - slice00659.tif"
# Set to True to recompute even if this filename already exists in areas.csv.
FORCE = False


if __name__ == "__main__":
    results = areas(filename=TARGET_FILENAME, force=FORCE, gaussian_sigma = 4,
    transition_sigma = 5,
    edge_emission_prob = 0.9,
    non_edge_emission_prob = 0.5,
    contrast_radial_weight = 1.0,
    contrast_threshold_pctile = 99.5,
    radial_contrast_sign='+')
    print(results)

    if TARGET_FILENAME is not None:
        image_name = TARGET_FILENAME.rsplit(".", 1)[0]
        image = plt.imread(f"data/figures/{image_name}.png")

        plt.figure(figsize=(16, 16))
        plt.imshow(image)
        plt.show()