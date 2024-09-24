import os
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from keras.applications import mobilenet_v2, vgg16, vgg19, resnet_v2, xception
from keras.models import load_model
from segmentation import segmentation
from keras.utils import custom_object_scope
from keras.layers import DepthwiseConv2D, SeparableConv2D


#  Added because the code is throwing an error in macos
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


#  Added because the code is throwing an error in macos
class CustomSeparableConv2D(SeparableConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        kwargs.pop("kernel_initializer", None)
        kwargs.pop("kernel_regularizer", None)
        kwargs.pop("kernel_constraint", None)
        super().__init__(*args, **kwargs)


#  Added because the code is throwing an error in macos
def load_custom_model(file_path):
    with custom_object_scope(
        {
            "DepthwiseConv2D": CustomDepthwiseConv2D,
            "SeparableConv2D": CustomSeparableConv2D,
        }
    ):
        return load_model(file_path)


def imageContrast(image, alpha=1.1, beta=5):
    image = image.astype(np.float32)
    image_contrast = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_contrast)
    new_image = image
    filter = ImageEnhance.Contrast(new_image)
    new_image = filter.enhance(1.25)
    filter = ImageEnhance.Sharpness(new_image)
    new_image = filter.enhance(1.25)
    new_image = np.asarray(new_image)
    return new_image


def image_to_text(image, model="mobilenet"):
    root_dir = os.path.join(os.path.dirname(__file__))
    models_folder_path = f"{root_dir}/models"
    # images_folder_path = f"{root_dir}/images"
    google = json.load(open(f"{root_dir}/meetei_mayek.json", "r"))["google"]
    model_name = model
    for file in os.listdir(models_folder_path):
        if model_name in file:
            print(model_name, file)
            # model = load_model(f"{models_folder_path}/{file}")
            model = load_custom_model(f"{models_folder_path}/{file}")
            break

    # temp_path = f"{os.path.join(os.path.dirname(__file__))}/temp"
    # if not os.path.exists(temp_path):
    #     os.mkdir(temp_path)
    # for file in os.listdir(temp_path):
    #     file_path = f"{temp_path}/{file}"
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)

    # SEGMENTATION
    LINES_WORDS_LETTERS = segmentation(image)
    LINES_WORDS_LETTERS_32 = []
    LINES_WORDS_LETTERS_71 = []
    LINES_WORDS_LETTERS_96 = []

    for line_count, line in enumerate(LINES_WORDS_LETTERS):
        for word_count, word in enumerate(line):
            for letter_count, letter in enumerate(word):
                if letter.shape[0] < 15 or letter.shape[1] < 15:
                    letter = imageContrast(cv2.resize(letter, (15, 15)))
                letter = imageContrast(cv2.resize(letter, (32, 32)))
                LINES_WORDS_LETTERS[line_count][word_count][letter_count] = letter

                # try:
                #     image_name = f"{temp_path}/{line_count}_{word_count}_{letter_count}.jpg"
                #     cv2.imwrite(image_name, letter)
                # except:
                #     print("#:", line_count, word_count, letter_count)

                letter = imageContrast(cv2.resize(letter, (96, 96)))
                if model_name == "vgg16" or model_name == "vgg19":
                    LINES_WORDS_LETTERS_32.append(cv2.resize(letter, (32, 32)))
                elif model_name == "xception":
                    letter = imageContrast(cv2.resize(letter, (71, 71)))
                    LINES_WORDS_LETTERS_71.append(letter)
                elif model_name == "mobilenet" or model_name == "resnet":
                    letter = imageContrast(cv2.resize(letter, (96, 96)))
                    LINES_WORDS_LETTERS_96.append(letter)

    LINES_WORDS_LETTERS_32 = np.asarray(LINES_WORDS_LETTERS_32)
    LINES_WORDS_LETTERS_71 = np.asarray(LINES_WORDS_LETTERS_71)
    LINES_WORDS_LETTERS_96 = np.asarray(LINES_WORDS_LETTERS_96)

    # PREPROCESS_INPUT
    if model_name == "vgg16":
        model_input = vgg16.preprocess_input(LINES_WORDS_LETTERS_32)
    if model_name == "vgg19":
        model_input = vgg19.preprocess_input(LINES_WORDS_LETTERS_32)
    if model_name == "xception":
        model_input = xception.preprocess_input(LINES_WORDS_LETTERS_71)
    if model_name == "resnet":
        model_input = resnet_v2.preprocess_input(LINES_WORDS_LETTERS_96)
    if model_name == "mobilenet":
        model_input = mobilenet_v2.preprocess_input(LINES_WORDS_LETTERS_96)

    # PREDICTION
    predictions = model.predict(model_input)
    char_arr = [google[np.argmax(prediction)] for prediction in predictions]

    # PREDICTIONS TO TEXT
    text = ""
    i = 0
    for line_count, line in enumerate(LINES_WORDS_LETTERS):
        for word_count, word in enumerate(line):
            for letter_count, letter in enumerate(word):
                text += char_arr[i]
                i += 1
            text += " "
        text += "\n"

    # text_file = open('output.txt', "w", encoding="utf-8")
    # text_file.write(text)
    # text_file.close()
    return text
