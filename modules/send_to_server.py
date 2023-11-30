import cv2
import requests

def send_frame_to_server(frame):
    # Encode the frame as bytes
    _, frame_bytes = cv2.imencode('.png', frame)

    # Create a multipart form-data request
    files = {'image': ('frame.png', frame_bytes.tobytes(), 'image/png')}
    response = requests.post('http://localhost:8080/api/detects', files=files)

    # Check the response status
    if response.status_code == 200:
        print('Frame successfully sent to the server')
    else:
        print('Failed to send frame to the server')
