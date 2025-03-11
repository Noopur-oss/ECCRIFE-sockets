import socket
import cv2
import struct
import pickle

def send_frame(client_socket, frame):
    # Serialize the frame
    data = pickle.dumps(frame)
    # Pack the message length
    message_size = struct.pack("Q", len(data))
    # Send the frame size followed by the frame data
    client_socket.sendall(message_size + data)

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

if __name__ == "__main__":
    # Setup socket connection to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 5000))  # Change to server IP and port

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Could not open webcam")
        exit()

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Send frame to the server
        send_frame(client_socket, frame)

        # Receive the interpolated frame from the server
        interpolated_frame = receive_frame(client_socket)
        if interpolated_frame is None:
            print("Error: No frame received from server")
            break
        else:
            print("Received interpolated frame")  # Debugging info

        # Show the captured frame and the interpolated frame
        cv2.imshow('Client - Captured Frame', frame)  # Original frame
        cv2.imshow('Client - Interpolated Frame', interpolated_frame)  # Interpolated frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    client_socket.close()
    cv2.destroyAllWindows()
