import random
import cv2
import numpy as np


class Remaper(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, word_img, text_box_pnts, word_color):
        """
        :param word_img:  word image with big background
        :param text_box_pnts: left-top, right-top, right-bottom, left-bottom of text word
        :return:
        """
        max_val = np.random.uniform(self.cfg.curve.min, self.cfg.curve.max)

        h = word_img.shape[0]
        w = word_img.shape[1]

        xmin = text_box_pnts[0][0]
        xmax = text_box_pnts[1][0]
        ymin = text_box_pnts[0][1]
        ymax = text_box_pnts[2][1]

        # Vectorized remap (much faster than nested Python loops).
        # Keep the same math as _remap_y(): int(max_val * sin(2*3.14*x/period)).
        x = np.arange(w, dtype=np.float32)
        offset = max_val * np.sin(2 * 3.14 * x / self.cfg.curve.period)
        # Casting float->int in numpy truncates toward 0, consistent with Python int().
        offset_i = offset.astype(np.int32)

        img_x = np.tile(x, (h, 1)).astype(np.float32)
        img_y = (np.arange(h, dtype=np.float32)[:, None] + offset_i[None, :].astype(np.float32)).astype(np.float32)

        # bbox remap range only depends on x-offset.
        off_min = int(offset_i.min()) if w > 0 else 0
        off_max = int(offset_i.max()) if w > 0 else 0
        remap_y_min = ymin + min(0, off_min)
        remap_y_max = ymax + max(0, off_max)

        remaped_text_box_pnts = [
            [xmin, remap_y_min],
            [xmax, remap_y_min],
            [xmax, remap_y_max],
            [xmin, remap_y_max],
        ]

        # TODO: use cuda::remap
        dst = cv2.remap(word_img, img_x, img_y, cv2.INTER_CUBIC)
        return dst, remaped_text_box_pnts

    def _remap_y(self, x, max_val):
        return int(max_val * np.math.sin(2 * 3.14 * x / self.cfg.curve.period))
