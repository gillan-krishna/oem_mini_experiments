from skimage.io import imread
import matplotlib.pyplot as plt

def imshow(image_path, mask_path):
    image = imread(image_path)
    mask = imread(mask_path)
    plt.imshow(image)
    if mask:
        plt.imshow(mask, alpha=0.5)
    text = "with" if mask else "without"
    plt.title(f"Image {text} mask")
    plt.show()