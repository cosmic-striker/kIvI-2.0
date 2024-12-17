import socket
import os
import argparse
import threading

def send_file(connection, file_path):
    """ Sends a file over the established connection. """
    try:
        with open(file_path, 'rb') as file:
            while chunk := file.read(1024):
                connection.sendall(chunk)
        print(f"Sent: {file_path}")
    except Exception as e:
        print(f"Error sending file {file_path}: {e}")

def receive_file(connection, dest_folder):
    """ Receives a file and writes it to the destination folder. """
    try:
        file_name = connection.recv(1024).decode('utf-8')
        file_path = os.path.join(dest_folder, file_name)
        with open(file_path, 'wb') as file:
            while chunk := connection.recv(1024):
                file.write(chunk)
        print(f"Received: {file_path}")
    except Exception as e:
        print(f"Error receiving file: {e}")

def start_server(host, port, data_folder):
    """ Starts the server to share files. """
    print("Starting server...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"Server listening on {host}:{port}")

        while True:
            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")
            threading.Thread(target=handle_client_server, args=(conn, data_folder)).start()

def handle_client_server(conn, data_folder):
    """ Handles file sending logic for connected clients. """
    try:
        for file_name in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file_name)
            conn.sendall(file_name.encode('utf-8'))  # Send file name
            send_file(conn, file_path)
    finally:
        conn.close()

def start_client(host, port, dest_folder):
    """ Starts the client to request and download files. """
    print("Starting client...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((host, port))
        print(f"Connected to server {host}:{port}")
        receive_file(client_socket, dest_folder)

def main():
    """ Entry point for the data sharing script. """
    parser = argparse.ArgumentParser(description="Data sharing between server and client.")
    parser.add_argument('--role', choices=['server', 'client'], required=True, help="Role of the node: server or client")
    parser.add_argument('--host', type=str, default='0.0.0.0', help="Host IP")
    parser.add_argument('--port', type=int, default=5001, help="Port number")
    parser.add_argument('--data_folder', type=str, default='data/raw', help="Folder to share (server mode)")
    parser.add_argument('--dest_folder', type=str, default='data/received', help="Folder to save files (client mode)")

    args = parser.parse_args()

    if args.role == 'server':
        start_server(args.host, args.port, args.data_folder)
    elif args.role == 'client':
        start_client(args.host, args.port, args.dest_folder)

if __name__ == "__main__":
    main()
