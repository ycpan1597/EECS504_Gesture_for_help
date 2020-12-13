from utils import *
from models import *
import os
import time
import torch
import torchvision.transforms as transforms

OUR_IMAGE_HEIGHT = 320; OUR_IMAGE_WIDTH = 192
valid_transform = transforms.Compose([
    transforms.ToTensor(),
    NewPad(), # requires images to be tensors; pad from the center out to make 608 x 352
    transforms.Resize([OUR_IMAGE_HEIGHT, OUR_IMAGE_WIDTH]), # make it a little smaller so that the network requires less memory
    transforms.Normalize(0.5, 0.5),
    ]
)
net = Net()

model_path = 'model.pth'
net.load_state_dict(torch.load(model_path))

# process video to make it runnable by the network
# make prediction on a frame-by-frame basis; annotate with text; save as a new video

# filename = 'PP_test.mov'
filename = 'CL_2-9.mp4'
export = False

cap = cv2.VideoCapture(filename)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

if export:
    out = cv2.VideoWriter('{}_labeled.avi'.format(filename.strip('.mp4')), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

options = ['1. Please get me some water',
           '2. Please help me use the bathroom',
           '3. I\'m not feeling well']
elapsed = []

# text parameters
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (300, 100)

# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

frame_index = 0
while True:

    has_frame, frame = cap.read()
    frame_index += 1
    if frame_index < 2: # ignore the first frame
        continue
    else:

        if not has_frame:
            print('Reached the end of the video')
            break

        if filename == 'PP_test.mov':
            cropped_frame = frame[60:560, 230:580, :]  # for PP_test.mp4
            thresh = 100
            gesture_present_thresh = 0.1

        elif filename == 'CL_1-9.mp4':
            cropped_frame = frame[90:690, 600:950, :] # for CL_3-9.mp4
            thresh = 88
            gesture_present_thresh = 0.3

        elif filename == 'CL_2-9.mp4':
            cropped_frame = frame[90:690, 550:900, :] # for CL_3-9.mp4
            thresh = 88
            gesture_present_thresh = 0.3

        elif filename == 'CL_3-9.mp4':
            cropped_frame = frame[90:690, 600:950, :] # for CL_3-9.mp4
            thresh = 88
            gesture_present_thresh = 0.3


        cropped_frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        cropped_frame_binary = (np.invert(cropped_frame_gray > thresh) * 255).astype(np.uint8) # for CL_3-9, thresh = 88

        if np.count_nonzero(cropped_frame_binary) / cropped_frame_binary.size > gesture_present_thresh:
            tformed_image = valid_transform(cropped_frame_binary)
            tformed_image = tformed_image.unsqueeze(0)  # convert from C x H x W to 1(B) x C x H x W

            start = time.time()
            output = net(tformed_image)
            elapsed.append(time.time() - start)
            _, predicted = torch.max(output, 1)
            # Using cv2.putText() method
            frame = cv2.putText(frame, options[predicted], org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
        if export:
            out.write(frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
if export:
    out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

# print mean prediction time and standard deviation
elapsed = np.array(elapsed)
print(elapsed.mean())
print(elapsed.std())