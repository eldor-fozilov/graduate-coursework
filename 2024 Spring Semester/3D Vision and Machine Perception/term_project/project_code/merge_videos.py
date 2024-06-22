from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, clips_array

# Load your two video clips
clip1 = VideoFileClip("video1.mp4")
clip2 = VideoFileClip("video2.mp4")

# Create titles for each video
txt_clip1 = TextClip("Original Video", fontsize=24, color='Black').set_position(("center", 10)).set_duration(clip1.duration)
txt_clip2 = TextClip("Edited Video", fontsize=24,  color='black').set_position(("center", 10)).set_duration(clip2.duration)

# Overlay the titles on the video clips
video1 = CompositeVideoClip([clip1, txt_clip1])
video2 = CompositeVideoClip([clip2, txt_clip2])

# Resize the clips to have the same height
height = min(video1.h, video2.h)
video1 = video1.resize(height=height)
video2 = video2.resize(height=height)

# Combine the two videos side by side
final_clip = clips_array([[video1, video2]])

# Write the result to a file
final_clip.write_videofile("output.mp4", codec='libx264')