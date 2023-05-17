import face_recognition
import cv2
import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt


def load_known_face_encodings(directory):
    known_face_encodings = {}
    for filename in os.listdir(directory):
        try:
            image = face_recognition.load_image_file(os.path.join(directory, filename))
            face_encoding = face_recognition.face_encodings(image)[0]
            name, _ = os.path.splitext(filename)
            known_face_encodings[name] = face_encoding
        except Exception as e:
            print(f"Failed to load image {filename}: {e}")
    return known_face_encodings


def main_menu(known_face_encodings, camera_index):
    console = Console()
    menu_options = {
        '1': {'label': 'Run Face Recognition', 'action': lambda: run_face_recognition(known_face_encodings, camera_index)},
        '2': {'label': 'Exit', 'action': lambda: sys.exit()}
    }
    while True:
        console.clear()
        menu_title = Text('Face Recognition Menu', style="bold underline")
        menu_items = [f"{key}. {menu_options[key]['label']}" for key in menu_options]
        menu_panel = Panel('\n'.join(menu_items), title=menu_title, border_style='green')
        console.print(menu_panel)
        choice = Prompt.ask('Select an option', choices=[str(i) for i in range(1, len(menu_items) + 1)])
        menu_options[choice]['action']()


def run_face_recognition(known_face_encodings, camera_index):
    video_capture = cv2.VideoCapture(camera_index)
    if not video_capture.isOpened():
        print("Failed to open camera.")
        return

    face_data = {'locations': [], 'encodings': [], 'names': []}
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/2 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            face_data['locations'] = []
            face_data['encodings'] = []
            face_data['names'] = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                if known_face_encodings:
                    matches = face_recognition.compare_faces(list(known_face_encodings.values()), face_encoding)
                    name = "Unknown"

                    # If a match was found in known_face_encodings, use the first one.
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = list(known_face_encodings.keys())[first_match_index]

                    face_data['names'].append(name)

                face_data['encodings'].append(face_encoding)

            face_data['locations'] = face_locations

        process_this_frame = not process_this_frame

        if face_data['locations']:
            # Display the results
            for (top, right, bottom, left), name in zip(face_data['locations'], face_data['names']):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (70, 130, 180), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (70, 130, 180), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, os.path.splitext(name)[0], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard or 'ESC' in the window to quit!
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "Kimage"
    known_face_encodings = load_known_face_encodings(directory)
    main_menu(known_face_encodings, 0)