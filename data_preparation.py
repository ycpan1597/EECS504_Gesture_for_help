import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

filename = "PP_test.mov" # name of recorded file
cap = cv2.VideoCapture(filename)
i = 0
ret = True
frames = []

i = 0
# store all files from a video of gestures. Not working very well (some frames are very blurry)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:
        if i % 3 == 0:
            frames.append(np.mean(frame, axis = 2))
            # cv2.imwrite(os.path.join(root_folder, 'gesture_1', 'pp_img%03d.png'%(i)), frame)
        i += 1
frames = np.asarray(frames)

## Identify bounding box
bbox = {'upperleft': (230, 60), 'width': 350, 'height': 500}

fig, ax = plt.subplots(1)
ax.imshow(frames.mean(axis=0), 'gray')
rect = patches.Rectangle(bbox['upperleft'], bbox['width'], bbox['height'],linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)

## Crop to bounding box
wmin, hmin = bbox['upperleft']
hmax = hmin + bbox['height']
wmax = wmin + bbox['width']

cropped_frames = np.zeros((bbox['height'], bbox['width'], frames.shape[0]))
for i in range(frames.shape[0]):
    cropped_frames[:, :, i] = frames[i, hmin:hmax, wmin:wmax]

## Tune threshold
thresh = 130 # out of 255
for i in range(cropped_frames.shape[2]):
    if i % 5 == 0: # view every 5
        plt.figure()
        plt.imshow((cropped_frames[:, :, i] > thresh) * 225, 'gray')

## Save grayscale and binary
initial = filename[0:2]
class_dash_trials = filename.strip('.mov')[3:]
folder = os.path.join('gesture_{}'.format(class_dash_trials[0]))

for i in range(cropped_frames.shape[2]):
    # save colored files too? not sure if this would work well
    cv2.imwrite(os.path.join(folder, 'grayscale', '%s_%s_%03d.png' % (initial, class_dash_trials, i)), (cropped_frames[:, :, i]))
    # save binary
    cv2.imwrite(os.path.join(folder, 'binary', '%s_%s_%03d.png' % (initial, class_dash_trials, i)), (cropped_frames[:, :, i] > thresh) * 255)