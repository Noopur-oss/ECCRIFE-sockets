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

        # Show the captured frame (for testing)
        #cv2.imshow('Client - Captured Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    client_socket.close()
    cv2.destroyAllWindows()
