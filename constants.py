from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = ROOT_DIR / "models" / "face_landmarker.task"

DATASET_DIR = ROOT_DIR / "data" / "sessions"
DEBUG_WINDOW_NAME = "Gaze Debug"
GAZE_PREVIEW_WINDOW_NAME = "Gaze Preview"
STATUS_MESSAGE_SECONDS = 4.0
VISION_CONCAT_MODEL_PATH = ROOT_DIR / "models" / "vision_gaze_latest.pt"
VISION_CONCAT_METRICS_PATH = ROOT_DIR / "models" / "vision_gaze_latest.json"
VISION_ATTENTION_MATCHED_MODEL_PATH = ROOT_DIR / "models" / "vision_gaze_attention_matched.pt"
VISION_ATTENTION_MATCHED_METRICS_PATH = ROOT_DIR / "models" / "vision_gaze_attention_matched.json"
VISION_MODEL_PATH = VISION_CONCAT_MODEL_PATH
VISION_METRICS_PATH = VISION_CONCAT_METRICS_PATH

PREDICTION_ALPHA = 0.35
RIDGE_ALPHA = 1.5
POLYNOMIAL_DEGREE = 2

COLLECTION_RECORD_SECONDS = 1.0
COLLECTION_LOCKOUT_SECONDS = 0.25
COLLECTION_MIN_SAMPLES = 10
EVAL_SPLIT_RATIO = 0.2
EYE_CROP_WIDTH = 96
EYE_CROP_HEIGHT = 64
SPLIT_GRID_COLS = 8
SPLIT_GRID_ROWS = 6
VISION_BATCH_SIZE = 64
VISION_EPOCHS = 28
VISION_LEARNING_RATE = 8e-4
VISION_WEIGHT_DECAY = 1e-4
VISION_AUGMENT_TRAIN = True
VISION_EVAL_EPOCHS = 10
VISION_HEAD_FEATURE_KEYS = [
    "face_center_x",
    "face_center_y",
    "face_scale",
    "head_yaw_deg",
    "head_pitch_deg",
    "head_roll_deg",
    "head_tx",
    "head_ty",
    "head_tz",
]

FACE_OVAL = [
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10),
]

LEFT_EYE = [
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (362, 263),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
]

RIGHT_EYE = [
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (133, 33),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
]

LEFT_IRIS = [(474, 475), (475, 476), (476, 477), (477, 474)]
RIGHT_IRIS = [(469, 470), (470, 471), (471, 472), (472, 469)]

LEFT_IRIS_POINTS = [474, 475, 476, 477]
RIGHT_IRIS_POINTS = [469, 470, 471, 472]
LEFT_EYE_POINTS = sorted({point for connection in LEFT_EYE for point in connection})
RIGHT_EYE_POINTS = sorted({point for connection in RIGHT_EYE for point in connection})

RIGHT_EYE_CORNER_POINTS = (33, 133)
RIGHT_EYE_UPPER_LID_POINTS = [159, 158, 160, 161]
RIGHT_EYE_LOWER_LID_POINTS = [145, 153, 144, 163]

LEFT_EYE_CORNER_POINTS = (263, 362)
LEFT_EYE_UPPER_LID_POINTS = [386, 385, 387, 388]
LEFT_EYE_LOWER_LID_POINTS = [374, 380, 373, 390]

MAPPER_FEATURE_KEYS = [
    "left_x",
    "left_y",
    "left_orth_y",
    "left_openness",
    "left_upper_gap",
    "left_lower_gap",
    "right_x",
    "right_y",
    "right_orth_y",
    "right_openness",
    "right_upper_gap",
    "right_lower_gap",
    "face_center_x",
    "face_center_y",
    "face_scale",
    "head_yaw_deg",
    "head_pitch_deg",
    "head_roll_deg",
    "head_tx",
    "head_ty",
    "head_tz",
]
