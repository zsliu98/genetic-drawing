import cv2
import numpy as np
import pickle
from IPython.display import clear_output

from .dna import DNA
from .parameter import BRUSH_PATH, BRUSH_NUM, BRUSH_SIZE, BRUSH_RANGE
from .parameter import DEFAULT_STEPS, DEFAULT_GENERATIONS, DEFAULT_DNA_COUNT, DEFAULT_GENE_COUNT
from .parameter import RANDOM_ADD, CROSS_OVER, MUTATE, HILL_CLIMB, CROSS_OVER_PORTION, MUTATE_PORTION


class Nature:
    def __init__(self, original_img, img_buffer=None, brush_range=BRUSH_RANGE):
        """
        :param original_img: the original image
        :param img_buffer: the canvas
        :param brush_range: the range of brushes
        """
        self.original_img = original_img
        self.img_grey = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        self.img_grads = self.__img_gradient(self.img_grey)

        self.brushes = []
        self.__preload_brushes(BRUSH_PATH)
        self.brush_range = brush_range
        self.sampling_mask = None
        # start with an empty black img
        if img_buffer is None:
            self.img_buffer = [np.zeros_like(self.original_img, dtype=np.uint8)]
        else:
            self.img_buffer = [img_buffer]

        self.best_dna = DNA(self.original_img,
                            self.img_grads,
                            brushes=self.brushes,
                            brush_range=self.__calc_brush_range(0, 1),
                            canvas=self.img_buffer[-1],
                            sampling_mask=self.sampling_mask)
        self.best_dna.refresh()
        print("Error {}".format(self.best_dna.cached_error))
        self.dna_list = []

    def save_state(self, path):
        """
        save current state to path
        :param path:
        :return:
        """
        with open(path, 'wb') as outg:
            state = {'original_img': self.original_img,
                     'brushes_range': self.brush_range,
                     'best_dna': self.best_dna}
            pickle.dump(state, outg, pickle.HIGHEST_PROTOCOL)

    def load_state(self, path):
        """
        load state from path
        :param path:
        :return:
        """
        with open(path, 'rb') as inp:
            state = pickle.load(inp)
            self.original_img = state['original_img']
            self.img_grey = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
            self.img_grads = self.__img_gradient(self.img_grey)
            self.brush_range = state['brushes_range']
            self.best_dna = state['best_dna']
            print("Error {}".format(self.best_dna.cached_error))
            self.img_buffer = [self.best_dna.draw()]

    def __preload_brushes(self, path):
        self.brushes = []
        for i in range(BRUSH_NUM):
            brush = cv2.imread(path + str(i) + '.jpg')
            brush = cv2.resize(brush, (BRUSH_SIZE, BRUSH_SIZE))
            self.brushes.append(brush)
            brush = np.flip(brush, axis=0)
            self.brushes.append(brush)

    def generate(self, steps=DEFAULT_STEPS, generations=DEFAULT_GENERATIONS,
                 dna_count=DEFAULT_DNA_COUNT, gene_count=DEFAULT_GENE_COUNT):
        """
        perform GA
        :param steps: the number of steps
        :param generations: the number of generations per step
        :param dna_count: the number of DNAs
        :param gene_count: the number of Genes per DNA
        :return:
        """
        for s in range(steps):
            # initialize new mask and DNAs
            if self.sampling_mask is not None:
                sampling_mask = self.sampling_mask
            else:
                sampling_mask = self.__create_sampling_mask(s, steps)
            brush_range = self.__calc_brush_range(s, steps)
            self.dna_list = []
            for i in range(dna_count):
                new_dna = DNA(self.original_img,
                              self.img_grads,
                              brushes=self.brushes,
                              brush_range=brush_range,
                              canvas=self.img_buffer[-1],
                              sampling_mask=sampling_mask)
                new_dna.init_random(gene_count=gene_count)
                self.dna_list.append(new_dna)
            # perform evolutions
            for i in range(generations):
                # random add
                for j in range(RANDOM_ADD):
                    new_dna = DNA(self.original_img,
                                  self.img_grads,
                                  brushes=self.brushes,
                                  brush_range=brush_range,
                                  canvas=self.img_buffer[-1],
                                  sampling_mask=sampling_mask)
                    new_dna.init_random(gene_count=gene_count)
                    self.dna_list.append(new_dna)
                self.__step(dna_count)
                clear_output(wait=True)
                print("{:.1f}%\tStage {}\tGeneration {}\tError {}".format(
                    100 * (s * generations + i) / (steps * generations), s, i, self.best_dna.cached_error))
            self.img_buffer.append(self.dna_list[0].cached_image)
            if self.dna_list[0].cached_error < self.best_dna.cached_error:
                self.best_dna = self.dna_list[0].copy(cached=True)

    def __step(self, dna_count):
        """
        cross, mutate, hill climb and select
        :param dna_count: the number of DNAs after one generation
        :return:
        """
        # cross
        if CROSS_OVER:
            dna_list1 = self.dna_list.copy()
            dna_list2 = self.dna_list.copy()
            np.random.shuffle(dna_list2)
            for dna1, dna2 in zip(dna_list1, dna_list2):
                if np.random.random() < CROSS_OVER_PORTION:
                    dna1, dna2 = DNA.cross_over(dna1, dna2)
                    self.dna_list.append(dna1)
                    self.dna_list.append(dna2)
        # mutate
        if MUTATE:
            dna_list1 = self.dna_list.copy()
            self.dna_list = []
            for dna in dna_list1:
                if np.random.random() < MUTATE_PORTION:
                    self.dna_list.append(DNA.mutate(dna))
                else:
                    self.dna_list.append(dna)
        # hill climb
        if HILL_CLIMB:
            dna_list1 = self.dna_list.copy()
            self.dna_list = []
            for dna in dna_list1:
                self.dna_list.append(DNA.hill_climb(dna))
        # select
        self.dna_list.sort(key=lambda x: x.cached_error)
        self.dna_list = self.dna_list[:dna_count]

    def __calc_brush_range(self, stage, total_stages):
        return [self.__calc_brush_size(self.brush_range[0], stage, total_stages),
                self.__calc_brush_size(self.brush_range[1], stage, total_stages)]

    def __create_sampling_mask(self, s, stages):
        percent = 0.2
        start_stage = int(stages * percent)
        sampling_mask = None
        if s >= start_stage:
            t = (1.0 - (s - start_stage) / max(stages - start_stage - 1, 1)) * 0.25 + 0.005
            sampling_mask = self.__calc_sampling_mask(t)
        return sampling_mask

    def __calc_sampling_mask(self, blur_percent):
        img = np.copy(self.img_grey)
        # Calculate gradient
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees )
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # calculate blur level
        w = img.shape[0] * blur_percent
        if w > 1:
            mag = cv2.GaussianBlur(mag, (0, 0), w, cv2.BORDER_DEFAULT)
        # ensure range from 0-255 (mostly for visual debugging, since in sampling we will renormalize it anyway)
        scale = 255.0 / mag.max()
        return mag * scale

    @staticmethod
    def __calc_brush_size(brange, stage, total_stages):
        b_min = brange[0]
        b_max = brange[1]
        t = stage / max(total_stages - 1, 1)
        return (b_max - b_min) * (-t * t + 1) + b_min

    @staticmethod
    def __img_gradient(img):
        # convert to 0 to 1 float representation
        img = np.float32(img) / 255.0
        # Calculate gradient
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees )
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # normalize magnitudes
        mag /= np.max(mag)
        # lower contrast
        mag = np.power(mag, 0.3)
        return mag, angle
