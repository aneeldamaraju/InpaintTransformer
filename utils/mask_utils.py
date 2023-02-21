import numpy as np

# Here we generate a mask rectangle plot it on top of the other rectangle
def gen_mask_rect(pts, H=256, W=256, thresh=.4):
    '''
    Generate a masking rectangle, that is centered close to an (edge) point in the image
    :param pts: list of candidate (edge) points in an image
    :param H: Height of image in pixels
    :param W: Width of image in pixels
    :param thresh: maximum threshold for height and width of mask. Decimal from 0 to 1
    :return: mask_rect: the mask in image space.
             mask_coords: a 4-list of the coordinates of the mask scaled from 0 to 1 in the order [l,b,r,t]
    '''

    pt = pts[np.random.choice(pts.shape[0]), :]
    w_pt = pt[0]
    h_pt = pt[1]
    # height and width of the mask
    w_mask = np.random.randint(int(thresh * W * .6), int(thresh * W))
    h_mask = np.random.randint(int(thresh * H * .6), int(thresh * H))

    rand_x = np.random.uniform(.3, .7)
    rand_y = np.random.uniform(.3, .7)
    #Generate mask coordinates
    mask_l = np.max([w_pt - int(w_mask * rand_x), 0])
    mask_r = np.min([w_pt + int(w_mask * (1 - rand_x)), W])
    mask_b = np.max([h_pt - int(h_mask * rand_y), 0])
    mask_t = np.min([h_pt + int(h_mask * (1 - rand_y)), H])

    #Create a meshgrid and fill iti in
    x_pos, y_pos = np.meshgrid(np.linspace(0, H - 1, H), np.linspace(0, W - 1, W))
    left_wall = x_pos > mask_l
    right_wall = x_pos < mask_r
    top_wall = y_pos < mask_t
    bot_wall = y_pos > mask_b
    mask_rect = left_wall * right_wall * top_wall * bot_wall

    return mask_rect, [mask_l, mask_b, mask_r, mask_t]


def pt_in_rect(pt, rect_coords):
    '''
    Check if a point is in a rectangle
    :return: boolean
    '''
    l, b, r, t = rect_coords
    pt_x, pt_y = pt
    if pt_x > l and pt_x < r:
        if pt_y > b and pt_y < t:
            return True
    return False

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from utils.generate_blob import *

    plt.figure(figsize=(8, 8))
    for itr in range(16):
        plt.subplot(4, 4, itr + 1)
        img, rad = generate_blob()
        pts = get_pts_on_curve(rad, P=30).squeeze()
        mask_rect, mask_coords = gen_mask_rect(pts)

        plt.imshow(img)
        mask_img = np.copy(img) * 1.0
        mask_img[mask_rect] = .5
        plt.imshow(mask_img, cmap='gray_r')
        plt.xticks([])
        plt.yticks([])
    plt.suptitle('Sample masks')
    plt.tight_layout()
    plt.show()