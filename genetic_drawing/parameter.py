BRUSH_PATH = 'brushes/watercolor/'  # path of brushes
BRUSH_NUM = 4  # number of brushes
BRUSH_SIZE = 300  # size of brushes
BRUSH_RANGE = ([0.1, 0.3], [0.3, 0.7])  # how to resize the brushes

DEFAULT_STEPS = 10  # default number of steps
DEFAULT_GENERATIONS = 40  # default number of generations
DEFAULT_DNA_COUNT = 10  # default DNA counts
DEFAULT_GENE_COUNT = 10  # default Gene counts

RANDOM_COLOR = False  # False: use color from the center of brushes; True: use random color

RANDOM_ADD = 1  # the number of DNA added per generation

CROSS_OVER = True  # whether perform cross-over
CROSS_OVER_PORTION = 0.5  # the portion (of DNA) of cross-over

MUTATE = True  # whether perform mutation
MUTATE_PORTION = 0.1  # the portion (of DNA) of mutation
MUTATE_RATE = 0.1  # the portion (of Gene) of mutation

HILL_CLIMB = True  # whether perform hill-climbing
HILL_CLIMB_RATE = 0.2  # the portion (of Gene) of hill-climbing
