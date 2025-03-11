import cv2
import socket
import struct
import pickle
import numpy as np
import onnxruntime as ort  # ONNX Runtime for running the ONNX model
import torch.nn.functional as F
import threading
from queue import Queue

def receive_frame(client_socket):
    # Unpack the message size
    packed_msg_size = client_socket.recv(8)
    if not packed_msg_size:
        return None
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    # Receive the frame data based on the message size
    data = b""
    while len(data) < msg_size:
        data += client_socket.recv(4096)

    frame = pickle.loads(data)
    return frame

def process_frame(frame, session):
    # Convert frame to numpy array and normalize to [0, 1]
    frame = np.array(frame).astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Ensure the frame has the shape (H, W, C) and convert to (1, 3, H, W)
    frame = frame.transpose(2, 0, 1)[None, :]  # Shape: (1, 3, H, W)

    # Ensure we have 8 channels; adjust based on your needs
    if frame.shape[1] < 8:  # If less than 8 channels
        # Repeat channels or pad with zeros
        # Here we can pad with zeros to reach 8 channels
        padding = np.zeros((1, 8 - frame.shape[1], frame.shape[2], frame.shape[3]), dtype=np.float32)
        frame = np.concatenate([frame, padding], axis=1)  # Shape will now be (1, 8, H, W)

    # Prepare dimensions for padding
    n, c, h, w = frame.shape  # n=1, c=8
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = ((0, 0), (0, 0), (0, ph - h), (0, pw - w))
    frame = np.pad(frame, padding, mode='constant')  # Pad frame

    # Convert to float16
    inputs = {session.get_inputs()[0].name: frame.astype(np.float16)}

    # Run inference using the ONNX model
    middle_frame = session.run(None, inputs)[0]

    # Postprocess output to convert back to image (HWC format)
    middle_frame = (middle_frame[0] * 255).astype(np.uint8).transpose(1, 2, 0)[:h, :w]

    return middle_frame


if __name__ == "__main__":
    # Set up sockets for input and output (modify the IP and ports as needed)
    serverSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSock.bind(('0.0.0.0', 5000))
    serverSock.listen()

    # Load the ONNX model
    onnx_model_path = './RIFE.onnx'  # Path to your ONNX model
    session = ort.InferenceSession(onnx_model_path)

    c, addr = serverSock.accept()
    print(f"Got Connection from: {addr}")

    while True:
        frame = receive_frame(c)
        if frame is None:
            print("Error: No frame received")
            break

        # Process frame with the ONNX model
        interpolated_frame = process_frame(frame, session)

        # Show the interpolated frame
        cv2.imshow("Server - Interpolated Frame", interpolated_frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
