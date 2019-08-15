from matplotlib import pyplot as plt

def show_image_tensor(dataset, index):
    plt.imshow(dataset[index][0].permute(1, 2, 0))
    plt.show()