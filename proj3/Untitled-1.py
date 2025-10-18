# %%
from PIL import Image, ImageOps
import numpy as np

def load_img_consistent(path, max_size=1200):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    
    w, h = img.size
    scale = max(w, h) / max_size
    print(scale)
    if scale > 1:
        new_size = (int(w / scale), int(h / scale))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"Resized from {w}×{h} → {new_size}")
    
    return np.array(img)

def normalize_points(pts):
    mean = pts.mean(axis=0)
    std  = pts.std(axis=0).mean()
    s = np.sqrt(2) / std
    T = np.array([[s, 0, -s*mean[0]],
                  [0, s, -s*mean[1]],
                  [0, 0, 1]])
    pts_h = np.c_[pts, np.ones(len(pts))].T
    npts = (T @ pts_h).T[:, :2]
    return npts, T

import numpy as np

def reprojection_errors(H, im1_pts, im2_pts):
    #im1 -> im2
    n = im1_pts.shape[0]
    p1 = np.c_[im1_pts, np.ones(n)]
    p2_hat = (H @ p1.T).T
    p2_hat = p2_hat[:, :2] / p2_hat[:, 2:3]

    #im2 -> im1 (consistency)
    Hinv = np.linalg.inv(H)
    p2 = np.c_[im2_pts, np.ones(n)]
    p1_hat = (Hinv @ p2.T).T
    p1_hat = p1_hat[:, :2] / p1_hat[:, 2:3]

    err_fwd = np.linalg.norm(p2_hat - im2_pts, axis=1)
    err_bwd = np.linalg.norm(p1_hat - im1_pts, axis=1)
    return err_fwd, err_bwd

def warp_bounds(im, H):
    h, w = im.shape[:2]
    corners = np.array([[0,0,1],[w,0,1],[0,h,1],[w,h,1]]).T
    wc = H @ corners
    wc /= wc[2]
    xs, ys = wc[0], wc[1]
    min_x, max_x = np.floor(xs.min()), np.ceil(xs.max())
    min_y, max_y = np.floor(ys.min()), np.ceil(ys.max())
    tx = -min_x if min_x < 0 else 0
    ty = -min_y if min_y < 0 else 0
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]])
    out_w = int(max_x - min_x)
    out_h = int(max_y - min_y)
    return (out_h, out_w), T



# %%
def computeH(im1_pts, im2_pts):
    # nnormalize
    p1n, T1 = normalize_points(im1_pts)
    p2n, T2 = normalize_points(im2_pts)

    N = im1_pts.shape[0]
    A = []
    for i in range(N):
        x, y = p1n[i]
        u, v = p2n[i]
        A.append([ x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([ 0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3,3)
    # Denormalize:
    # # H = T2^{-1} Hn T1
    H = np.linalg.inv(T2) @ Hn @ T1
    return H / H[2,2]

# %%
%matplotlib qt

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def pick_points(image_path, n, title):
    img = load_img_consistent(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)  
    # no rotation now
    ax.set_title(title)
    pts = np.array(plt.ginput(n, timeout=-1, show_clicks=True), dtype=float)
    plt.close(fig)
    return pts


N = 6  # 
# im1_path = "/Users/jaitegchahal/cs180/jaitegchahal123.github.io/proj3/media/IMG_5781.jpg"
# im2_path = "/Users/jaitegchahal/cs180/jaitegchahal123.github.io/proj3/media/IMG_5782.jpg"
im1_path = "/Users/jaitegchahal/cs180/jaitegchahal123.github.io/proj3/media/IMG_5863.jpg"
im2_path = "/Users/jaitegchahal/cs180/jaitegchahal123.github.io/proj3/media/IMG_5864.jpg"

print(f"Select {N} points on LEFT image (in order).")
im1_pts = pick_points(im1_path, N, "LEFT image")

print(f"Select {N} MATCHING points on RIGHT image in the SAME order.")
im2_pts = pick_points(im2_path, N, "RIGHT image")

# im1_pts[i] ↔ im2_pts[i]


# %%
print(im1_pts)
print(im2_pts)

# %%
H = computeH(im1_pts, im2_pts)
print(H)

# %%
import matplotlib.pyplot as plt
im1 = load_img_consistent(im1_path)
im2 = load_img_consistent(im2_path)
def visualize_projection(H, im1, im2, im1_pts, im2_pts, title=""):
    # project im1_pts -> im2 frame
    p1 = np.c_[im1_pts, np.ones(len(im1_pts))]
    proj = (H @ p1.T).T
    proj = proj[:, :2] / proj[:, 2:3]

    plt.figure(figsize=(7,7))
    plt.imshow(im2)
    plt.scatter(im2_pts[:,0], im2_pts[:,1], s=40, c='lime', label='clicked im2_pts')
    plt.scatter(proj[:,0],   proj[:,1],   s=40, facecolors='none', edgecolors='red', label='H * im1_pts')
    for i,(a,b) in enumerate(zip(im2_pts, proj)):
        plt.plot([a[0], b[0]], [a[1], b[1]], 'y-', linewidth=1)
        plt.text(a[0]+3, a[1]+3, str(i), color='white', fontsize=9)
        plt.text(b[0]+3, b[1]+3, str(i), color='red', fontsize=9)
    plt.legend(); plt.title(title); plt.axis('off'); plt.show()

visualize_projection(H, im1, im2, im1_pts, im2_pts, "Projection check: im1 -> im2")


# %%
def warpImageNearestNeighbor(im, H, out_shape):
    H_inv = np.linalg.inv(H)
    h_out, w_out = out_shape
    warped_im = np.zeros((h_out, w_out, im.shape[2]), dtype=im.dtype)

    for i in range(h_out):
        for j in range(w_out):
            p_out = np.array([j, i, 1])
            p_in = H_inv @ p_out
            p_in /= p_in[2]
            x_in, y_in = int(round(p_in[0])), int(round(p_in[1]))

            if 0 <= x_in < im.shape[1] and 0 <= y_in < im.shape[0]:
                warped_im[i, j] = im[y_in, x_in]

    return warped_im

# %%
def warpImageBilinear(im, H, out_shape):
    H_inv = np.linalg.inv(H)
    h_out, w_out = out_shape
    warped_im = np.zeros((h_out, w_out, im.shape[2]), dtype=im.dtype)

    for i in range(h_out):
        for j in range(w_out):
            p_out = np.array([j, i, 1])
            p_in = H_inv @ p_out
            p_in /= p_in[2]
            x_in, y_in = p_in[0], p_in[1]

            if 0 <= x_in < im.shape[1]-1 and 0 <= y_in < im.shape[0]-1:
                x0, y0 = int(np.floor(x_in)), int(np.floor(y_in))
                x1, y1 = x0 + 1, y0 + 1

                dx, dy = x_in - x0, y_in - y0

                for c in range(im.shape[2]):
                    top = (1 - dx) * im[y0, x0, c] + dx * im[y0, x1, c]
                    bottom = (1 - dx) * im[y1, x0, c] + dx * im[y1, x1, c]
                    warped_im[i, j, c] = (1 - dy) * top + dy * bottom

    return warped_im


# %%
im1 = load_img_consistent(im1_path)
im2 = load_img_consistent(im2_path)
h_out, w_out = im2.shape[0], im2.shape[1]  
#canvas = im2’s frame

imwarped_nn = warpImageNearestNeighbor(im1, H, (h_out, w_out))
imwarped_bilinear = warpImageBilinear(im1, H, (h_out, w_out))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(imwarped_nn)
axs[0].set_title("Nearest Neighbor")
axs[1].imshow(imwarped_bilinear)
axs[1].set_title("Bilinear Interpolation")
plt.show()

# %%
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(imwarped_nn)
plt.title("Nearest Neighbor")
plt.axis('off')

rando = Image.fromarray(imwarped_nn)
rando.save("./media/warpednn.png")

plt.figure()
plt.imshow(imwarped_bilinear)
plt.title("Bilinear Interpolation")
plt.axis('off')

rand1 = Image.fromarray(imwarped_bilinear)
rand1.save("./media/warpedbil.png")

plt.show()


# %%
def mosaic_bounds(im1, im2, H):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    corners1 = np.array([[0,0,1],[w1,0,1],[0,h1,1],[w1,h1,1]]).T
    corners2 = np.array([[0,0,1],[w2,0,1],[0,h2,1],[w2,h2,1]]).T

    warped_corners1 = H @ corners1
    warped_corners1 /= warped_corners1[2]

    all_x = np.hstack([warped_corners1[0], corners2[0]])
    all_y = np.hstack([warped_corners1[1], corners2[1]])

    min_x, max_x = np.floor(all_x.min()), np.ceil(all_x.max())
    min_y, max_y = np.floor(all_y.min()), np.ceil(all_y.max())

    tx = -min_x if min_x < 0 else 0
    ty = -min_y if min_y < 0 else 0
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]])
    out_w, out_h = int(max_x - min_x), int(max_y - min_y)
    return (out_h, out_w), T


# %%
out_shape, T = mosaic_bounds(im1, im2, H)
H_total = T @ H
im1_warped = warpImageBilinear(im1, H_total, out_shape)


# %%
mosaic = np.zeros_like(im1_warped)
h2, w2 = im2.shape[:2]

for i in range(h2):
    for j in range(w2):
        x, y = int(j + T[0,2]), int(i + T[1,2])
        if 0 <= y < mosaic.shape[0] and 0 <= x < mosaic.shape[1]:
            mosaic[y, x] = im2[i, j]


# %%
import numpy as np
from scipy.ndimage import distance_transform_edt

def alpha_from_valid(mask):
    return distance_transform_edt(mask) / (distance_transform_edt(~mask) + 1e-6)

valid1 = (im1_warped.sum(axis=2) > 0)
valid2 = (mosaic.sum(axis=2) > 0)

a1 = alpha_from_valid(valid1)
a2 = alpha_from_valid(valid2)

bias = 1.5
a2 *= bias

W = (a1 + a2 + 1e-6)[..., None]
mosaic_blend = (im1_warped * a1[...,None] + mosaic * a2[...,None]) / W
mosaic_blend = np.clip(mosaic_blend, 0, 255).astype(np.uint8)

# %%
plt.imshow(mosaic_blend.astype(np.uint8))
plt.title("Blended Mosaic")
plt.axis('off')
plt.show()

# %%
img = Image.fromarray(mosaic_blend.astype(np.uint8))
img.save("./media/blended_mosaic1.png")


