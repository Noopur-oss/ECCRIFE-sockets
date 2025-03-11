import cv2
import socket
import struct
import pickle
import numpy as np
import onnxruntime as ort

def receive_frame(client_socket):
    packed_msg_size = client_socket.recv(8)
    if not packed_msg_size:
        return None
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    data = b""
    while len(data) < msg_size:
        data += client_socket.recv(4096)

    frame = pickle.loads(data)
    return frame

def process_frame(frame, session):
    frame = np.array(frame).astype(np.float32) / 255.0
    frame = frame.transpose(2, 0, 1)[None, :]

    # Ensure 8 channels
    if frame.shape[1] < 8:
        padding = np.zeros((1, 8 - frame.shape[1], frame.shape[2], frame.shape[3]), dtype=np.float32)
        frame = np.concatenate([frame, padding], axis=1)

    n, c, h, w = frame.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = ((0, 0), (0, 0), (0, ph - h), (0, pw - w))
    frame = np.pad(frame, padding, mode='constant')

    inputs = {session.get_inputs()[0].name: frame.astype(np.float16)}
    middle_frame = session.run(None, inputs)[0]
    middle_frame = (middle_frame[0] * 255).astype(np.uint8).transpose(1, 2, 0)[:h, :w]

    return middle_frame

if __name__ == "__main__":
    serverSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSock.bind(('0.0.0.0', 5000))
    serverSock.listen()

    onnx_model_path = './RIFE.onnx'
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

        # Use a small wait time to allow the display to update continuously
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
