import cv2
import numpy as np
import copy

from .parameter import BRUSH_NUM, BRUSH_SIZE, RANDOM_COLOR


class Gene:

    def __init__(self, color, size, y, x, rotation, brush_number):
        self.color = color
        self.size = size
        self.y, self.x = y, x
        self.rotation = rotation
        self.brush_number = brush_number
        self.cached_image, self.cached_area, self.cached_alpha = None, None, 1.0

    def copy(self):
        """
        get a copy of this Gene
        :return: the copy
        """
        new_gene = Gene(color=self.color, size=self.size, y=self.y, x=self.x,
                        rotation=self.rotation, brush_number=self.brush_number)
        return new_gene

    @classmethod
    def from_random(cls, bound, brush_range, img_mag, img_angels, sampling_mask):
        """
        get a random Gene
        :param bound: the shape of the original image
        :param brush_range: the range of brushes
        :param img_mag: the magnitudes of the original image
        :param img_angels: the angels of the original image
        :param sampling_mask: the sampling mask
        :return: the Gene
        """
        if sampling_mask is not None:
            pos = Gene.__util_sample_from_img(sampling_mask)
            y, x = pos[0][0], pos[1][0]
        else:
            y, x = np.random.randint(0, bound[0]), np.random.randint(0, bound[1])

        color = np.random.randint(low=0, high=256, size=bound[2]) if RANDOM_COLOR else None
        size = np.random.uniform(brush_range[0], brush_range[1])
        local_mag = img_mag[y][x]
        local_angle = img_angels[y][x] + 90
        rotation = np.random.randint(-180, 180) * (1 - local_mag) + local_angle
        brush_number = np.random.randint(0, BRUSH_NUM * 2)
        return cls(color, size, y, x, rotation, brush_number)

    def refresh(self, original_img, brushes, padding):
        """
        refresh the cached image, the cached area and the cached alpha
        :param original_img: the original image
        :param brushes: all brushes
        :param padding: the size of padding
        :return:
        """
        brush_img = brushes[self.brush_number]
        brush_img = cv2.resize(brush_img, None, fx=self.size, fy=self.size, interpolation=cv2.INTER_CUBIC)
        brush_img = self.__rotateImg(brush_img, self.rotation)
        rows, cols, _ = brush_img.shape

        y_min = int(self.y + padding - rows / 2)
        y_max = y_min + brush_img.shape[0]
        x_min = int(self.x + padding - cols / 2)
        x_max = x_min + brush_img.shape[1]
        color = self.color if RANDOM_COLOR else original_img[self.y, self.x]
        my_clr = np.copy(brush_img)
        my_clr[:, :] = color

        foreground = my_clr[0:rows, 0:cols, :]
        self.cached_alpha = brush_img / 255.0
        self.cached_area = (slice(y_min, y_max), slice(x_min, x_max), slice(None, None))
        self.cached_image = np.multiply(self.cached_alpha, foreground)

    def mutate(self, bound, brush_range, img_mag, img_angels, sampling_mask):
        """
        get a mutation of this Gene
        :param bound: the shape of the original image
        :param brush_range: the range of brushes
        :param img_mag: the magnitudes of the original image
        :param img_angels: the angels of the original image
        :param sampling_mask: the sampling mask
        :return: the mutated Gene
        """
        new_gene = self.copy()
        option = np.random.randint(0, 6) if RANDOM_COLOR else np.random.randint(1, 6)
        if option == 0:
            new_gene = np.random.randint(low=0, high=256, size=bound[2]) if RANDOM_COLOR else None
        elif option == 1 or option == 2:
            shift_size = BRUSH_SIZE * self.size
            y_shift = np.random.randint(-shift_size, shift_size)
            x_shift = np.random.randint(-shift_size, shift_size)
            new_gene.y = np.clip(new_gene.y + y_shift, 0, bound[0] - 1)
            new_gene.x = np.clip(new_gene.x + x_shift, 0, bound[1] - 1)
        elif option == 3:
            new_gene.size = np.random.uniform(brush_range[0], brush_range[1])
        elif option == 4:
            local_mag = img_mag[self.y][self.x]
            local_angle = img_angels[self.y][self.x] + 90
            new_gene.rotation = np.random.randint(-180, 180) * (1 - local_mag) + local_angle
        elif option == 5:
            new_gene.brush_number = np.random.randint(0, BRUSH_NUM * 2)
        return new_gene

    @staticmethod
    def __rotateImg(img, angle):
        rows, cols, channels = img.shape
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, m, (cols, rows))
        return dst

    @staticmethod
    def __util_sample_from_img(img):
        pos = np.indices(dimensions=img.shape)
        pos = pos.reshape(2, pos.shape[1] * pos.shape[2])
        img_flat = np.clip(img.flatten() / img.flatten().sum(), 0.0, 1.0)
        return pos[:, np.random.choice(np.arange(pos.shape[1]), 1, p=img_flat)]
