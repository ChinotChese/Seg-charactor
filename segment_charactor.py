import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import glob
from sklearn.cluster import KMeans
from collections import Counter

# take colors of charactor
def take_color(folder_screenshot_path):
    screen_shots = list(glob.glob(folder_screenshot_path + '/*.jpg'))
    centroids = []
    for shot in screen_shots:
        img_shot = img.imread(shot)
        img_2D = img_shot.reshape(-1, 3)
        quantity = img_2D.shape[0]
        color = (np.sum(img_2D, axis=0)/quantity).round(0).astype(int)
        centroids.append(color)
    return centroids

# take index color of character in frame
def find_color_character_index(centroid, centroid_of_charactor):
    quantity = len(centroid)
    min_dis = []
    for center in centroid:
        m_dis = np.min([np.linalg.norm(center - color) for color in centroid_of_charactor])
        min_dis.append(m_dis)
    ind_of_character_color = np.argsort(min_dis)[:quantity]
    return ind_of_character_color


path_sc_sh = "./Seg-charactor/screen_shots"
centroid_character = take_color(path_sc_sh)
# binary mask
def binary_mask(path_img, n_cluster):
    # convert 2D
    img_tensor = img.imread(path_img)
    img_size = img_tensor.shape
    img_2d = img_tensor.reshape(-1, 3)

    # clustering
    kmeans_model = KMeans(n_clusters=n_cluster) 
    cluster_labels = kmeans_model.fit_predict(img_2d)
    centroid = kmeans_model.cluster_centers_.round(0).astype(int)

    # take index color of character
    ind = find_color_character_index(centroid, centroid_character)

    # convert to binary color
    bcolor = np.copy(centroid)
    for i in range(len(centroid)):
        if i in ind:
            bcolor[i] = [255, 255, 255]
        else:
            bcolor[i] = [0, 0, 0]
    img_bm = np.reshape(bcolor[cluster_labels],img_size)
    return img_bm

