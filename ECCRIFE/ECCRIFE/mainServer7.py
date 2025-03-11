import cv2
import socket
import struct
import pickle
import torch
import torch.nn.functional as F
import threading
from queue import Queue
from train_log.RIFE_HDv3 import Model

# Constants
MAX_QUEUE_SIZE = 10  # Maximum size of the frame queue
NUM_WORKER_THREADS = 4  # Number of threads to process frames

# Queue to hold frames
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)


def receive_frame(client_socket):
    try:
        packed_msg_size = client_socket.recv(8)
        if not packed_msg_size:
            return None
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Receive the frame data
        data = b""
        while len(data) < msg_size:
            packet = client_socket.recv(4096)
            if not packet:
                break
            data += packet

        frame = pickle.loads(data)
        return frame
    except Exception as e:
        print(f"Error receiving frame: {e}")
        return None


def process_frame(model, device):
    while True:
        if not frame_queue.empty():
            try:
                # Get two frames for interpolation
                frame1 = frame_queue.get()
                frame2 = frame_queue.get() if not frame_queue.empty() else frame1  # Get the next frame for interpolation

                # Convert frames to tensors and preprocess
                frame1_tensor = torch.tensor(frame1.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255.
                frame2_tensor = torch.tensor(frame2.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255.

                # Prepare dimensions for padding
                n, c, h, w = frame1_tensor.shape
                ph = ((h - 1) // 32 + 1) * 32
                pw = ((w - 1) // 32 + 1) * 32
                padding = (0, pw - w, 0, ph - h)

                frame1_tensor = F.pad(frame1_tensor, padding)
                frame2_tensor = F.pad(frame2_tensor, padding)

                # Perform interpolation
                middle_frame_tensor = model.inference(frame1_tensor, frame2_tensor)

                # Reverse preprocessing to convert tensor back to image format
                middle_frame = (middle_frame_tensor[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

                # Resize the interpolated frame to match the original frame size
                middle_frame = cv2.resize(middle_frame, (w, h))

                # Show the interpolated frame
                cv2.imshow("Server - Interpolated Frame", middle_frame)
                cv2.imshow("Server - Original Frame 1", frame1)
                cv2.imshow("Server - Original Frame 2", frame2)

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_queue.task_done()  # Mark the task as done
            except Exception as e:
                print(f"Error processing frames: {e}")


def receive_frames(client_socket):
    """Thread to continuously receive frames and add them to the queue."""
    while True:
        frame = receive_frame(client_socket)
        if frame is None:
            print("Error: No frame received or connection closed")
            break
        try:
            frame_queue.put(frame, timeout=1)  # Add frame to queue with a timeout
        except Exception as e:
            print(f"Queue is full, could not add frame: {e}")  # Handle full queue


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up socket for server
    serverSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSock.bind(('0.0.0.0', 8080))
    serverSock.listen()

    # Initialize the RIFE model
    rife_model = Model()
    rife_model.load_model('./train_log/', -1)
    rife_model.eval()
    rife_model.device()

    # Wait for client connection
    c, addr = serverSock.accept()
    print(f"Got Connection from: {addr}")

    # Start a thread to receive frames and add them to the queue
    receiver_thread = threading.Thread(target=receive_frames, args=(c,))
    receiver_thread.daemon = True
    receiver_thread.start()

    # Start worker threads for processing frames
    worker_threads = []
    for _ in range(NUM_WORKER_THREADS):
        worker_thread = threading.Thread(target=process_frame, args=(rife_model, device))
        worker_thread.daemon = True
        worker_thread.start()
        worker_threads.append(worker_thread)

    # Wait for the receiver thread to finish
    receiver_thread.join()

    # Cleanup
    c.close()
    serverSock.close()
    cv2.destroyAllWindows()
