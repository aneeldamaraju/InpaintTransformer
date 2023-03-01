import numpy as np
import cv2
from utils.shape_utils import *


#Code from Dor Verbin 2022
def perlin(x, y, seed=0):
    # permutation table
    # np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here


def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)


def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y


def get_curvature(rad):
    rad_t_plus_1 = np.concatenate([rad[1:], rad[:1]], axis=0)
    rad_t_minus_1 = np.concatenate([rad[-1:], rad[:-1]], axis=0)

    drad = 0.5 * (rad_t_plus_1 - rad_t_minus_1) / (angle[1] - angle[0])
    d2rad = (rad_t_plus_1 - 2 * rad + rad_t_minus_1) / (angle[1] - angle[0]) ** 2

    curvature = np.abs(rad ** 2 + 2 * drad ** 2 - rad * d2rad) / np.sqrt((rad ** 2 + drad ** 2) ** 3)

    return curvature


def generate_blob(H=256, W=256, blobs=1):
    maxrad = H / 3.0
    minrad = H / 4.0



    # np.random.seed(2020)
    img_np = np.zeros((H, W), dtype=np.uint8)

    if blobs == 2:
        colors = [255, 128] if np.random.rand() > 0.5 else [128, 255]
    elif blobs == 1:
        colors = [255]

    sharpness = .6
    for blob_id, clr in enumerate(colors):
        angle = np.arange(0, 2*np.pi, 0.001)[:-1]

        xp = 2.5 + sharpness * np.cos(angle[np.newaxis, :])
        yp = 2.5 + sharpness * np.sin(angle[np.newaxis, :])

        p = perlin(xp, yp)

        rad = minrad + (maxrad - minrad) * (0.5 + p)[0, :]

        xlist = rad * np.cos(angle) + 0.5 * H  # + 0.1 * H * np.cos(axis_angle + np.pi * (blob_id == 1))
        ylist = rad * np.sin(angle) + 0.5 * W  # + 0.1 * W * np.sin(axis_angle + np.pi * (blob_id == 1))

        pts = np.array([[int(x), int(y)] for (x, y) in zip(xlist, ylist)])
        pts = pts.reshape((1, -1, 2))

        #         np.save(f'datasets/blobs_small/rad_{blob_id}_{i:03}.npy', rad)
        #         np.save(f'datasets/blobs_small/pts_{blob_id}_{i:03}.npy', pts)

        cv2.fillPoly(img_np, pts, color=clr, lineType=cv2.LINE_AA)


    # img_np = cv2.resize(img_np, (55, 55), interpolation=cv2.INTER_AREA)
    img = img_np.reshape(img_np.shape[0], img_np.shape[1], 1)
    img = np.asarray(img / 255.0, dtype=np.float32)
    img = np.asarray(img > .5, dtype=np.float32)
    return img, pts


def generate_line(H, W,use_slope=True):
    thresh = int(.2 * W)

    img_np = np.zeros((H, W), dtype=np.uint8)

    x_c = int(np.random.uniform(.2, .8) * H)
    ylist = np.arange(thresh, W - thresh, 0.001)
    if use_slope:
        slope = 1 / np.random.uniform(-.2, .2)
        xlist = x_c + ylist / slope
        pts = np.array([[x_c, 0], [int(x_c + W / slope), W], [H, W], [H, 0]])
    else:
        xlist = x_c + ylist *0
        pts = np.array([[x_c, 0], [x_c, W], [H, W], [H, 0]])

    out_pts = np.array([[x, y] for (x, y) in zip(xlist, ylist)])
    out_pts = out_pts.reshape((1, -1, 2))

    # print(pts)
    pts = pts.reshape((1, -1, 2))
    cv2.fillPoly(img_np, pts, color=255, lineType=cv2.LINE_AA)

    img = img_np.reshape(img_np.shape[0], img_np.shape[1], 1)
    img = np.asarray(img / 255.0, dtype=np.float32)
    img = np.asarray(img > .5, dtype=np.float32)
    return img, out_pts


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for itr in range(1, 21):
        plt.subplot(4, 5, itr)
        img, xy = generate_blob(256,256)
        pts = get_pts_on_curve(xy, P=30)
        plt.imshow(img[:, :, 0], cmap='gray')
        plt.scatter(pts[0,:,0],pts[0,:,1],s=2)
        # print
        plt.axis('off')
    plt.suptitle('Sample Blobs')
    plt.tight_layout()
    plt.show()
