import cv2
import numpy as np
import math
import os
import json


def generate_clock(hour, minute):

    img_size = (227, 227, 3)

    # choose a random background color
    background_color = list(np.random.randint(0, 256, 3))
    image = np.full(img_size, background_color, dtype=np.uint8)

    # draw circle with the above background color and following properties
    radius = 112
    center = (img_size[0] // 2, img_size[1] // 2)
    circle_color = tuple(np.random.randint(0, 256, 3).tolist())
    cv2.circle(image, center, radius, circle_color, -1)

    # add random noise (optional)
    add_random_noise = np.random.rand() > 0.5
    if add_random_noise:
        noise = np.random.normal(0, 0.5, img_size).astype(np.uint8)
        image = cv2.add(image, noise)

    # clock hands
    hour_hand_length = 45
    minute_hand_length = 90

    hour_angle = 360 * (hour % 12) / 12 + 30 * \
        (minute / 60) - 90  # -90 to start at 12 o'clock
    minute_angle = 360 * minute / 60 - 90  # -90 to start at 12 o'clock

    hour_hand_end = (
        round(center[0] + hour_hand_length *
              math.cos(hour_angle * math.pi / 180)),
        round(center[1] + hour_hand_length *
              math.sin(hour_angle * math.pi / 180))
    )

    minute_hand_end = (
        round(center[0] + minute_hand_length *
              math.cos(minute_angle * math.pi / 180)),
        round(center[1] + minute_hand_length *
              math.sin(minute_angle * math.pi / 180))
    )

    # draw clock hands
    cv2.line(image, center, hour_hand_end, (0, 0, 0),
             thickness=5)  # 0, 0, 0 is black
    cv2.line(image, center, minute_hand_end, (0, 0, 0),
             thickness=3)  # 0, 0, 0 is black

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    for i in range(1, 13):

        angle = math.radians(360 * i / 12 - 90)  # -90 to start at 12 o'clock

        if i in [10, 11, 12]:
            # adjust for 2-digit numbers
            digit_position = (
                int(center[0] + 0.9 * radius * math.cos(angle)) - 12,
                int(center[1] + 0.9 * radius * math.sin(angle)) + 12
            )
        else:
            digit_position = (
                # general adjustment to appropriatly center the digits
                int(center[0] + 0.9 * radius * math.cos(angle)) - 7,
                int(center[1] + 0.9 * radius * math.sin(angle)) + 7
            )
        cv2.putText(image, str(i), digit_position, font, font_scale,
                    (0, 0, 0), font_thickness, cv2.LINE_AA)

    return image


def generate_clock_image_data(number_of_images, save_folder_path="./dataset"):
    # save images to disk and hour / minute labels to json file

    os.makedirs(f"{save_folder_path}/clock_images", exist_ok=True)
    data_info_list = []

    for i in range(number_of_images):
        hour = np.random.randint(1, 13)
        minute = np.random.randint(1, 61)
        img = generate_clock(hour, minute)

        cv2.imwrite(f"{save_folder_path}/clock_images/{i}.png", img)

        data_info = {
            "image_path": f"clock_images/{i}.png",
            "hour": hour,
            "minute": minute
        }

        data_info_list.append(data_info)

    json_file_path = f"{save_folder_path}/data_info.json"

    # use json for convenience
    with open(json_file_path, "w") as output_file:
        json.dump(data_info_list, output_file, indent=2)


if __name__ == "__main__":

    # generate_clock_image_data(3000)  # generate 30000 images

    # example
    img = generate_clock(11, 60)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
