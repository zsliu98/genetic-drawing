import cv2
import numpy as np
import copy

from .parameter import BRUSH_NUM, BRUSH_SIZE, RANDOM_COLOR, MUTATE_RATE, HILL_CLIMB_RATE
from .gene import Gene


class DNA:

    def __init__(self, original_img, img_gradient, brushes, brush_range, canvas=None, sampling_mask=None):
        """
        :param original_img: the original image
        :param img_gradient: the gradient image
        :param brushes: all brushes
        :param brush_range: the range of brushes
        :param canvas: the canvas
        :param sampling_mask: the sampling mask
        """
        self.genes = []
        self.original_img = original_img
        self.bound = original_img.shape

        # CTRLS
        self.brushes = brushes
        self.brush_range = brush_range
        self.padding = int(BRUSH_SIZE * self.brush_range[1] / 2 + 5)

        self.canvas = canvas

        # IMG GRADIENT
        self.img_mag = img_gradient[0]
        self.img_angles = img_gradient[1]

        # OTHER
        self.sampling_mask = sampling_mask

        # CACHE
        self.cached_image = None
        self.cached_error = None

    def copy(self, cached=False):
        """
        get a copy of this DNA
        :param cached: whether copy the cached image and the cached error
        :return: the copy
        """
        new_dna = DNA(original_img=self.original_img,
                      img_gradient=(self.img_mag, self.img_angles),
                      brushes=self.brushes,
                      brush_range=self.brush_range,
                      canvas=self.canvas,
                      sampling_mask=self.sampling_mask)
        new_dna.genes = self.genes[:]
        if cached:
            new_dna.cached_error = self.cached_error
            new_dna.cached_image = self.cached_image
        return new_dna

    def init_random(self, gene_count):
        """
        init the DNA with random Genes
        :param gene_count: the number of Genes
        :return:
        """
        # initialize random DNA sequence
        for i in range(gene_count):
            # random color
            gene = Gene.from_random(self.bound, self.brush_range, self.img_mag, self.img_angles, self.sampling_mask)
            gene.refresh(self.original_img, self.brushes, self.padding)
            self.genes.append(gene)
        # refresh cache error and image
        self.refresh()

    def refresh(self):
        """
        refresh the cached image and the cached error
        :return:
        """
        self.cached_image = self.draw()
        self.cached_error = np.sum(cv2.absdiff(self.original_img, self.cached_image))

    def draw(self):
        """
        draw the image of this DNA
        :return: the image
        """
        # set image to pre generated
        if self.canvas is None:  # if we do not have an image specified
            in_img = np.zeros(shape=self.bound, dtype=np.uint8)
        else:
            in_img = np.copy(self.canvas)
        # apply padding
        p = self.padding
        in_img = np.pad(in_img, ((p, p), (p, p), (0, 0)), mode='constant', constant_values=0)
        # draw every gene
        for gene in self.genes:
            foreground, area, alpha = gene.cached_image, gene.cached_area, gene.cached_alpha
            background = in_img[area]
            background = np.multiply(np.clip((1.0 - alpha), 0.0, 1.0), background)
            in_img[area] = np.clip(np.add(foreground, background), 0.0, 255.0).astype(np.uint8)

        y, x = in_img.shape[0], in_img.shape[1]
        return in_img[p:(y - p), p:(x - p), :]

    @staticmethod
    def mutate(dna):
        dna = dna.copy()
        np.random.shuffle(dna.genes)
        for idx, gene in enumerate(dna.genes[:]):
            if np.random.random() < MUTATE_RATE:
                gene = gene.mutate(dna.bound, dna.brush_range, dna.img_mag, dna.img_angles, dna.sampling_mask)
                gene.refresh(dna.original_img, dna.brushes, dna.padding)
                dna.genes[idx] = gene
        dna.refresh()
        return dna

    @staticmethod
    def cross_over(dna1, dna2):
        dna1, dna2 = dna1.copy(), dna2.copy()
        genes = dna1.genes + dna2.genes
        np.random.shuffle(genes)
        dna1.genes = genes[:len(dna1.genes)]
        dna2.genes = genes[len(dna1.genes):]
        dna1.refresh()
        dna2.refresh()
        return dna1, dna2

    @staticmethod
    def hill_climb(dna):
        current_error = dna.cached_error
        dna = dna.copy()
        old_dna = dna.copy()
        for idx, gene in enumerate(dna.genes[:]):
            if np.random.random() < HILL_CLIMB_RATE:
                gene = gene.mutate(dna.bound, dna.brush_range, dna.img_mag, dna.img_angles, dna.sampling_mask)
                gene.refresh(dna.original_img, dna.brushes, dna.padding)
                dna.genes[idx] = gene
                dna.refresh()
                if dna.cached_error < current_error:
                    old_dna.genes = dna.genes[:]
                else:
                    dna.genes = old_dna.genes[:]
        dna.refresh()
        return dna
