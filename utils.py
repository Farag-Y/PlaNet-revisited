import cv2
import numpy as np

def preprocess_frame(frame, size=64):
    frame = cv2.resize(frame, (size, size))        # (size, size, 3)
    frame = np.transpose(frame, (2, 0, 1))         # (3, size, size)
    return frame.astype(np.float32) / 255.0        # normalize to [0, 1]