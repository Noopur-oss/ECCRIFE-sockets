# main.py
import cv2
import socket
import struct
import pickle
import torch
import numpy
import torch.nn.functional as F
import threading
from queue import Queue
from train_log.RIFE_HDv3


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

def process_frame(frame,model):
    # Convert frame to tensor and preprocess like in your model
    frame = torch.tensor(frame.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255.

    # Dummy second frame for interpolation (as an example)
    img1 = frame.clone()  # Replace with another actual frame

    # Prepare dimensions for padding as shown in your provided code
    n, c, h, w = frame.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    frame = F.pad(frame, padding)
    img1 = F.pad(img1, padding)

    # Perform interpolation
    middle_frame = model.inference(frame, img1)

    # Reverse preprocessing to convert tensor back to image format
    middle_frame = (middle_frame[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

    return middle_frame


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up sockets for input and output (modify the IP and ports as needed
    serverSock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    serverSock.bind(('0.0.0.0',12345))
    serverSock.listen()
    # Initialize the RIFE model
    rife_model = Model()
    rife_model.load_model('./train_log/',-1)
    rife_model.eval()

    rife_model.device()
    c,addr = serverSock.accept()
    print(f"Got Connection from :{addr}")

    while True:
        frame = receive_frame(c)
        if frame is None:
            print("Error: No frame received")
            break

        # Process frame with the interpolation model
        interpolated_frame = process_frame(frame,rife_model)

        # Show the interpolated frame
        cv2.imshow("Server - Interpolated Frame", interpolated_frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
