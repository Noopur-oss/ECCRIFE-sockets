# rife_model.py
import torch
import cv2
import numpy as np
from train_log.RIFE_HDv3 import Model # Assuming the RIFE model is in this path

class RIFE:
    def __init__(self):
        self.model = Model()
        self.model.load_model('C:/Users/rupoo/PycharmProjects/ECCRIFE/ECCRIFE/train_log')  # Path to your model weights
        self.model.eval()  # Set model to evaluation mode

    def process(self, frame1, frame2):
        # Normalize input frames (as required by the model)
        frame1 = frame1.astype(np.float32) / 255.0
        frame2 = frame2.astype(np.float32) / 255.0

        # Convert to torch tensors and add batch dimension
        tensor1 = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0)
        tensor2 = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0)

        # Run the model to interpolate
        with torch.no_grad():
            interpolated_frame = self.model.infer(tensor1, tensor2)
        
        # Convert the interpolated frame back to numpy
        interpolated_frame = interpolated_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        interpolated_frame = (interpolated_frame * 255.0).astype(np.uint8)
        return interpolated_frame
