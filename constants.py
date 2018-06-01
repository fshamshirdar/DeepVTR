MIN_SHORTCUT_DISTANCE = 5
SHORTCUT_WINDOW = 10
GOAL_SIMILARITY_THRESHOLD = 0.8
SHORTCUT_SIMILARITY_THRESHOLD = 0.8
SEQUENCE_VELOCITIES = [0.3, 0.5, 0.8, 1.0, 1.5]
SEQUENCE_LENGTH = 5

LOCO_IMAGE_SIZE = 224
LOCO_IMAGE_WIDTH = 224
LOCO_IMAGE_HEIGHT = 224
LOCO_NUM_CLASSES = 3

PLACE_IMAGE_SIZE = 227
PLACE_IMAGE_WIDTH = 227
PLACE_IMAGE_HEIGHT = 227
PLACE_NUM_CLASSES = 3

TRAINING_LOCO_IMAGE_SCALE = 224
TRAINING_LOCO_BATCH = 32
TRAINING_LOCO_LR = 0.01
TRAINING_LOCO_MOMENTUM = 0.9
TRAINING_LOCO_LR_SCHEDULER_SIZE = 7
TRAINING_LOCO_LR_SCHEDULER_GAMMA = 0.1

TRAINING_PLACE_IMAGE_SCALE = 227
TRAINING_PLACE_BATCH = 32
TRAINING_PLACE_MARGIN = 0.5 # 300 # 0.2
TRAINING_PLACE_LR = 0.0005
TRAINING_PLACE_MOMENTUM = 0.9
TRAINING_PLACE_LR_SCHEDULER_SIZE = 7
TRAINING_PLACE_LR_SCHEDULER_GAMMA = 0.1

DATA_COLLECTION_ROUNDS = 1000
DATA_COLLECTION_MIN_ANGLE = 0.174533 # 10 deg
DATA_COLLECTION_MAX_ANGLE = 0.92173 # 52 deg
DATA_COLLECTION_MIN_SPEED = 0.5
DATA_COLLECTION_MAX_SPEED = 3.0
DATA_COLLECTION_PLAYING_ROUNG_LENGTH = 100
DATA_COLLECTION_MIN_HEIGHT = 1
DATA_COLLECTION_MAX_HEIGHT = 20

DATASET_MAX_ACTION_DISTANCE = 1
