import cv2
import numpy as np

def model_wrapper(model, *inputs, trailing_dims=3):
    leading = inputs[0].shape[:-trailing_dims]
    reshaped = [obs.reshape(-1, *obs.shape[-trailing_dims:]) for obs in inputs]
    out = model(*reshaped)
    return out.view(*leading, *out.shape[1:])


def preprocess_frame(frame, size=64):
    frame = cv2.resize(frame, (size, size))        # (size, size, 3)
    frame = np.transpose(frame, (2, 0, 1))         # (3, size, size)
    return frame.astype(np.float32) / 255.0        # normalize to [0, 1]