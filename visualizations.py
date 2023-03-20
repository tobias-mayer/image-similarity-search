import matplotlib.pyplot as plt
import cv2


def show_images(images, rows = 1, cols=1,figsize=(12, 12), filename='show_images.png'):
    fig, axes = plt.subplots(ncols=cols, nrows=rows,figsize=figsize)
    
    for index, name in enumerate(images):
        axes.ravel()[index].imshow(cv2.cvtColor(images[name], cv2.COLOR_BGR2RGB))
        axes.ravel()[index].set_title(name)
        axes.ravel()[index].set_axis_off()
        
    plt.tight_layout() 
    plt.savefig(filename)
