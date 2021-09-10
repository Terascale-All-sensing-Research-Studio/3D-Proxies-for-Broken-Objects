import cv2
import numpy as np
from scipy import interpolate

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

ENCODER = None


def normalize_features(features):
    """
    Normalize a list of features. Will return:
        [centroid, standard deviation]
    """
    if features is None:
        return None
    cent = np.mean(features, axis=0)
    stdev = np.std(features, axis=0)

    return np.concatenate((cent, stdev), axis=0)


def edsift_descriptor(image):
    """
    Compute a SIFT descriptor for an image.
    """

    def create_circular_mask(h, w, radius):
        center = (int(w / 2), int(h / 2))
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = dist_from_center <= radius
        return mask

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()

    pts = np.random.random((500, 2)) * np.array(image.shape[:2][::-1])
    kpts = [cv2.KeyPoint(r[0], r[1], 1) for r in pts]

    im_h, im_w = gray.shape[:2]
    r = 5
    bins = 10
    hists = []
    for k in pts.astype(int):
        sub_img = gray[
            max(k[0] - 5, 0) : min(k[0] + 5, im_h),
            max(k[1] - 5, 0) : min(k[1] + 5, im_w),
        ]

        cur_hist = []
        for r_ in np.linspace(0, r, bins + 1)[1:]:
            mask = create_circular_mask(sub_img.shape[0], sub_img.shape[1], r_)
            if len(cur_hist) == 0:
                cur_hist.append((mask * sub_img).sum())
            else:
                cur_hist.append((mask * sub_img).sum() - cur_hist[-1])

        cur_hist = interpolate.interp1d(np.linspace(0, r, bins), np.array(cur_hist))(
            np.linspace(0, r, 128)
        )
        cur_hist /= (2 * r) ** 2
        hists.append(np.expand_dims(cur_hist, axis=0))

    import matplotlib.pyplot as plt

    hists = np.vstack(hists)
    desc = sift.compute(gray, kpts)[1]
    desc = np.hstack((desc, hists))
    return desc


def sift_descriptor(image):
    """
    Compute a SIFT descriptor for an image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Do sift
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray, None)[1]


def orb_descriptor(image):
    """
    Compute a SIFT descriptor for an image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Do surf
    orb = cv2.ORB_create(nfeatures=500)
    kp, desc = orb.detectAndCompute(gray, None)

    return desc


def vgg_descriptor(image):
    global ENCODER
    # This prevents the encoder from being loaded multiple times
    if ENCODER is None:
        model = VGG16(weights="imagenet")
        ENCODER = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    # prepare the image for VGG
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = image[np.newaxis, :, :, :]

    # call feature extraction
    return ENCODER.predict(image)


def process_eDSIFT(f_in, f_out):
    image = cv2.imread(f_in)
    np.savez(f_out, features=edsift_descriptor(image))


def process_SIFT(f_in, f_out):
    image = cv2.imread(f_in)
    np.savez(f_out, features=sift_descriptor(image))


def process_ORB(f_in, f_out):
    image = cv2.imread(f_in)
    np.savez(f_out, features=orb_descriptor(image))


def process_VGG(f_in, f_out):
    image = cv2.imread(f_in)
    np.savez(f_out, features=vgg_descriptor(image))
