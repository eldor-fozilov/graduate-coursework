import numpy as np
import cv2
from loadVid import loadVid
from planarH import *
from yourHelperFunctions import *

# Write script for Q12
cv_cover = cv2.imread('../data/cv_cover.jpg')
panda = loadVid('../data/ar_source.mov')
book = loadVid('../data/book.mov')

# Read the first frame
book_first_frame = book[0]

H, W = book_first_frame.shape[:2]

# Create a VideoWriter object to write the output video
video_writer = cv2.VideoWriter("../result/ar.avi",
                               cv2.VideoWriter_fourcc(*'XVID'), 25, (W, H))

# Loop over the frames
num_frames = min(len(panda), len(book))

for i in range(num_frames):
    # Read the next frame
    panda_frame = panda[i]
    book_frame = book[i]

    # Compose the image
    composite_image = compose_image(cv_cover, panda_frame, book_frame)
    # Write the image to the video
    video_writer.write(composite_image)

video_writer.release()
cv2.destroyAllWindows()
