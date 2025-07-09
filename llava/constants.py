CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Report and Segmentation_mask
MASK_THRESH = 3000  # mask island smaller than this threshold will be removed
ORGAN_MASK_MAPPING = {
    # --- Diseases mapped to lung-related anatomy ---
    "disease-atelectasis": ["lung halves"],
    "disease-cardiomegaly": ["heart region"],
    "disease-consolidation": ["lung halves"],
    "disease-edema": ["lung halves"],
    "disease-enlarged cardiomediastinum": ["mediastinum"],
    "disease-fracture": ["ribs", "all vertebrae"],
    "disease-lung lesion": ["lung halves"],
    "disease-lung opacity": ["lung halves"],
    "disease-pleural effusion": ["lung halves"],
    "disease-pleural other": ["lung halves"],
    "disease-pneumonia": ["lung halves"],
    "disease-pneumothorax": ["lung halves"],

    # --- Organs mapped directly ---
    "organ-heart": ["heart region"],
    "organ-lungs": ["lung halves"],
    "organ-pleura": ["lung halves"],
    "organ-mediastinum": ["mediastinum"],
    "organ-bones": ["ribs", "all vertebrae"],
    "organ-diaphragm": ["diaphragm"],

    # --- Other topics ---
    "support devices": ["all"],
    "patient status": ["all"],
    "other": ["all"],
}

# BAD_IMAGE_PATHS = "/data/sc159/data/MIMIC_III/segmentation_single/bad_image_path.txt"
BAD_IMAGE_PATHS = "/work/devika/data/MIMIC_III/MIMIC_III/segmentation_single/bad_image_path.txt"  # todo
