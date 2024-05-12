import numpy as np
import cv2 
#Q12
def loadVid(path):
    video = cv2.VideoCapture(path)
    # Get the number of frames in the video
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    # Loop through all the frames and save them as separate images
    for _ in range(num_frames):
        # Read the next frame
        ret, frame = video.read()
        frames.append(frame)
    # Release the video file
    video.release()
    return np.stack(frames, axis=0)
