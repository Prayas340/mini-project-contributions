import os
import matplotlib.pyplot as plt
import matplotlib.image as img

dir_path = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(dir_path, "picture.jpg")
if not os.path.exists(image_path):
    image_path = "Finding_Lanes/picture.jpg"

image = img.imread(image_path)
plt.imshow(image)
plt.show()
