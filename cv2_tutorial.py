import cv2
import os
import numpy as np

## read and show one image
def read_and_show_image(fn='sample.jpg'):
    img = cv2.imread(fn)
    window_name = 'image'
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

## capture an image from the webcam
def capture_frame(cam_num=1, fn='capture.jpg'):
    # cam_num = 1 (webcam) or 0 (epoc cam)
    cap = cv2.VideoCapture(cam_num)
    while(True):
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        cv2.imshow('frame', rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            out = cv2.imwrite(fn, frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

## record a videom from the webcam
def record_video(cam_num=1, fn='output.avi', fps=30):
    # cam_num = 1 (webcam) or 0 (epoc cam)
    # doesn' work with epoc cam at the moment
    assert cam_num == 1, 'cam_num must be 1 (webcam) for recording'
    cap = cv2.VideoCapture(cam_num)

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # out = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
    out = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        if ret:
            # Write the frame into the file 'output.avi'
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

        # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    cv2.waitKey(1)

## read this video, annotate with text, and record the annotated video
def read_and_show_video(fn='output.avi'):
    cap = cv2.VideoCapture(fn)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            print('Reached the end of the video')
            break

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def annotate_video(input_fn='output.avi', output_fn='output_withtext.avi', annotation='OpenCV'):

    cap = cv2.VideoCapture(input_fn)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_fn, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            print('Reached the end of the video')
            break

        gray = frame

        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (500, 200)

        # fontScale
        fontScale = 1

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.putText() method
        gray = cv2.putText(gray, annotation, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        out.write(gray)

        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

