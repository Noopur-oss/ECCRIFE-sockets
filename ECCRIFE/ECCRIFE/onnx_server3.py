import socket
import cv2
import struct
import pickle
import numpy as np
import onnxruntime as ort  # ONNX Runtime for running the ONNX model
import threading
from queue import Queue

def receive_frame(client_socket, frame_queue):
    while True:
        # Unpack the message size
        packed_msg_size = client_socket.recv(8)
        if not packed_msg_size:
            print("No message size received, closing connection.")
            break
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Receive the frame data based on the message size
        data = b""
        while len(data) < msg_size:
            data += client_socket.recv(4096)

        frame = pickle.loads(data)
        # Put the received frame in the queue
        frame_queue.put(frame)

def send_frame(client_socket, frame):
    # Serialize the frame
    data = pickle.dumps(frame)
    # Pack the message length
    message_size = struct.pack("Q", len(data))
    # Send the frame size followed by the frame data
    client_socket.sendall(message_size + data)

def process_frame(frame_queue, session, processed_frame_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break  # Exit condition

        # Convert frame to numpy array and normalize to [0, 1]
        frame = np.array(frame).astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Ensure the frame has the shape (H, W, C) and convert to (1, 3, H, W)
        frame = frame.transpose(2, 0, 1)[None, :]  # Shape: (1, 3, H, W)

        # Ensure we have 8 channels; adjust based on your needs
        if frame.shape[1] < 8:  # If less than 8 channels
            # Pad with zeros to reach 8 channels
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

        # Put the processed frame in the queue for streaming
        processed_frame_queue.put(middle_frame)

        # Show the interpolated frame
        cv2.imshow("Server - Interpolated Frame", middle_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def stream_frames(client_socket, processed_frame_queue):
    while True:
        frame = processed_frame_queue.get()
        if frame is None:
            break  # Exit condition

        # Send the processed frame back to the client
        send_frame(client_socket, frame)

if __name__ == "__main__":
    # Set up sockets for input and output (modify the IP and ports as needed)
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(('0.0.0.0', 5000))
    server_sock.listen()

    # Load the ONNX model
    onnx_model_path = './RIFE.onnx'  # Path to your ONNX model
    session = ort.InferenceSession(onnx_model_path)

    c, addr = server_sock.accept()
    print(f"Got Connection from: {addr}")

    # Create queues for frames
    frame_queue = Queue(maxsize=10)
    processed_frame_queue = Queue(maxsize=10)  # Queue for processed frames

    # Start the receiving thread
    receiver_thread = threading.Thread(target=receive_frame, args=(c, frame_queue))
    receiver_thread.start()

    # Start the processing thread
    processor_thread = threading.Thread(target=process_frame, args=(frame_queue, session, processed_frame_queue))
    processor_thread.start()

    # Start the streaming thread
    streaming_thread = threading.Thread(target=stream_frames, args=(c, processed_frame_queue))
    streaming_thread.start()

    # Wait for threads to finish
    receiver_thread.join()
    frame_queue.put(None)  # Signal the processor thread to exit
    processed_frame_queue.put(None)  # Signal the streaming thread to exit
    processor_thread.join()
    streaming_thread.join()

    c.close()
    server_sock.close()
