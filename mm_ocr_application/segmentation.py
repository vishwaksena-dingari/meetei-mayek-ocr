import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance


def addPadding(image, all=None):
    """
    - ``Input``: an image
    - ``Output``:
        - add padding top: 10, bottom: 20 - ((height + top) % 10), left: 10, right: 20 - ((width + left) % 10)
        - so that the image can be divided into 10 equal parts vertically
    """
    image = np.asarray(image)
    height, width, channels = image.shape
    image = Image.fromarray(image)
    if all == None:
        top = 10
        bottom = 20 - ((height + top) % 10)
        left = 10
        right = 20 - ((width + left) % 10)
    else:
        top, right, bottom, left = all, all, all, all
    new_width = width + left + right
    new_height = height + top + bottom
    result = Image.new(image.mode, (new_width, new_height), (255, 255, 255))
    result.paste(image, (left, top))
    result = np.asarray(result)
    return result


def removePadding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(
        gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV
    )
    for i in range(4):
        while sum(binary_image[0]) == 0:
            image = image[1:]
            binary_image = binary_image[1:]
        image = np.rot90(image)
        binary_image = np.rot90(binary_image)
    return image


def tempDisplay(dict_):
    """
    - `Input`: dictionary
        - ``key``: name for the image to be displayed
        - ``value``: image to be displayed
    - `Output`: a matplotlib plot with the all the images in the dictionary in one image
    """
    rows = int(math.sqrt(len(dict_)))
    cols = math.ceil(len(dict_) / rows)
    fig = plt.figure(figsize=(10, 5))
    i = 1
    for key in dict_:
        fig.add_subplot(rows, cols, i)
        i = i + 1
        plt.imshow(dict_[key])
        plt.axis("off")
        plt.title(str(key))
    plt.show()


def cropImage(pixel_values, original_image, rotate=False):
    """
    - `Input`:
        - ``pixel_values``: vertical pixel values range [from, to] to be cropped
        - ``original_image``: image from which the cropped images are needed
        - ``rotate``: if the image is to be rotated or not before the cropping (used for word segmentation form the line images)
    - `Output`: cropped images from the ``original_image``
    """
    image = original_image
    if rotate:
        image = np.rot90(image, 3)
    cropped_images = []
    h, w = image.shape[:2]
    for i in range(len(pixel_values)):
        crop_image = image[pixel_values[i][0] : pixel_values[i][1], 0:w]
        if rotate:
            cropped_images.append(np.rot90(crop_image))
        else:
            cropped_images.append(crop_image)
    return cropped_images


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


def line_word_segmentation(
    image, line_segmentation=False, word_segmentation=False, letter_segmentation=False
):
    """
    - `Input`:
        - ``image``: image to be segmented
        - ``rotate``: ``bool`` if the image is to be rotated or not (used for word segmentation form the line images)
    - `Output`: vertical pixel values range [from, to] to be cropped
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ret, image = cv2.threshold(image, 128, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    if line_segmentation:
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        image = cv2.erode(image, rect_kernel_2, iterations=1)
        image = cv2.dilate(image, line_kernel, iterations=1)
    if word_segmentation:
        image = np.rot90(image, 3)
        image = cv2.dilate(image, rect_kernel_2, iterations=4)
    if letter_segmentation:
        image = np.rot90(image, 3)

    h, w = image.shape[:2]
    pixel_values = []
    temp = []
    for i in range(1, h):
        if len(temp) == 2:
            pixel_values.append(temp)
            temp = []
        if len(temp) == 0 and sum(image[i - 1]) == 0 and sum(image[i]) != 0:
            temp.append(i - 1)
        if len(temp) == 1 and sum(image[i - 1]) != 0 and sum(image[i]) == 0:
            temp.append(i + 1)
    return pixel_values


def word2Letters(image):
    org_img = image.copy()
    # img_copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, binary_img = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    thinned_img = binary_img

    contours, hierarchy = cv2.findContours(
        thinned_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
    cropped_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x1, y1, x2, y2 = x - 2, y - 2, x + w + 2, y + h + 2
        cropped = org_img[y1:y2, x1:x2]
        cropped_images.append(cropped)
    return cropped_images


def letterSegmentation(letter, avg_width_or_height, rotate=False):
    letter = removePadding(letter)
    AVG_WIDTH_OR_HEIGHT = avg_width_or_height
    sub_let = [letter]
    max_idx = 0

    while (
        sub_let[max_idx].shape[1] if rotate else sub_let[max_idx].shape[0]
    ) > AVG_WIDTH_OR_HEIGHT * 1.5:
        sub_let_img = sub_let[max_idx]
        gray = cv2.cvtColor(sub_let_img, cv2.COLOR_RGB2GRAY)
        gray = np.rot90(gray, 3) if rotate else gray
        _, binary_img = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)

        pixel_values = []
        sum_histogram = np.asarray([sum(row) for row in binary_img])

        if rotate:
            idx_arr = sum_histogram[
                AVG_WIDTH_OR_HEIGHT // 2 : len(sum_histogram) - AVG_WIDTH_OR_HEIGHT // 4
            ]
        else:
            idx_arr = sum_histogram[
                AVG_WIDTH_OR_HEIGHT // 4 : len(sum_histogram) - AVG_WIDTH_OR_HEIGHT // 4
            ]

        if idx_arr.size == 0:
            break

        if rotate:
            idx = (
                AVG_WIDTH_OR_HEIGHT // 2
                + np.where(idx_arr == np.min(idx_arr[np.nonzero(idx_arr)]))[0][0]
            )
        else:
            idx = (
                AVG_WIDTH_OR_HEIGHT // 4
                + np.where(idx_arr == np.min(idx_arr[np.nonzero(idx_arr)]))[0][0]
            )

        left, right = sub_let[:max_idx], sub_let[max_idx + 1 :]
        if rotate:
            pixel_values = [[0, idx], [idx, len(binary_img)]]
        else:
            pixel_values = [[0, idx + 1], [idx - 1, len(binary_img)]]

        if rotate:
            cropped_sub_let = cropImage(pixel_values, sub_let_img, rotate=True)
        else:
            cropped_sub_let = cropImage(pixel_values, sub_let_img)

        sub_let = [*left, *cropped_sub_let, *right]
        max_idx = 0
        for i, img in enumerate(sub_let):
            if rotate:
                if img.shape[1] > sub_let[max_idx].shape[1]:
                    max_idx = i
            else:
                if img.shape[0] > sub_let[max_idx].shape[0]:
                    max_idx = i
    return sub_let


def segmentation(image):
    ORIGINAL_IMAGE = addPadding(image)
    ORIGINAL_IMAGE = imageContrast(ORIGINAL_IMAGE)

    # Line Segmentation
    pixel_values = line_word_segmentation(ORIGINAL_IMAGE, line_segmentation=True)
    ARTICLE_LINES = cropImage(pixel_values, ORIGINAL_IMAGE)
    avg_height = 0
    for i, line in enumerate(ARTICLE_LINES):
        avg_height += line.shape[0]
    avg_height = avg_height / len(ARTICLE_LINES)
    NEW_ARTICLE_LINES = []
    for i, line in enumerate(ARTICLE_LINES):
        if line.shape[0] < avg_height // 2:
            NEW_ARTICLE_LINES[len(NEW_ARTICLE_LINES) - 1] = np.vstack(
                (NEW_ARTICLE_LINES[len(NEW_ARTICLE_LINES) - 1], line)
            )
        else:
            NEW_ARTICLE_LINES.append(line)
    ARTICLE_LINES = NEW_ARTICLE_LINES

    # Word Segmentation
    ARTICLE_WORDS = []
    for line in ARTICLE_LINES:
        pixel_values = line_word_segmentation(line, word_segmentation=True)
        words = cropImage(pixel_values, line, rotate=True)
        ARTICLE_WORDS.append(words)

    # Letter Segmentation
    ARTICLE_LETTERS = []
    for line in ARTICLE_WORDS:
        temp_line = []
        for word in line:
            word = addPadding(word, 2)
            letters = word2Letters(word)
            temp_line.append(letters)
        ARTICLE_LETTERS.append(temp_line)

    width_sum, height_sum, count = 0, 0, 0
    for line_count, line in enumerate(ARTICLE_LETTERS):
        for word_count, word in enumerate(line):
            for letter_count, letter in enumerate(word):
                temp_letter = removePadding(letter)
                width_sum += temp_letter.shape[1]
                height_sum += temp_letter.shape[0]
                count += 1

    AVG_HEIGHT = math.floor(height_sum / count)
    AVG_WIDTH = math.floor(width_sum / count)

    TEMP_ARTICLE_LETTERS = []
    for line_count, line in enumerate(ARTICLE_LETTERS):
        temp_line = []
        for word_count, word in enumerate(line):
            temp_word = []
            for letter_count, letter in enumerate(word):
                if letter.shape[1] > AVG_WIDTH * 1.25:
                    sub_let_horizontal = letterSegmentation(letter, AVG_WIDTH, True)
                    temp_word = [*temp_word, *sub_let_horizontal]
                else:
                    temp_word.append(letter)
            temp_line.append(temp_word)
        TEMP_ARTICLE_LETTERS.append(temp_line)
    ARTICLE_LETTERS = TEMP_ARTICLE_LETTERS

    TEMP_ARTICLE_LETTERS = []
    for line_count, line in enumerate(ARTICLE_LETTERS):
        temp_line = []
        for word_count, word in enumerate(line):
            temp_word = []
            for letter_count, letter in enumerate(word):
                if letter.shape[0] > AVG_HEIGHT * 1.25:
                    sub_let_vertical = letterSegmentation(letter, AVG_HEIGHT)
                    temp_word = [*temp_word, *sub_let_vertical]
                else:
                    temp_word.append(letter)
            temp_line.append(temp_word)
        TEMP_ARTICLE_LETTERS.append(temp_line)
    ARTICLE_LETTERS = TEMP_ARTICLE_LETTERS

    ARTICLE_LINES_WORDS_LETTERS = ARTICLE_LETTERS
    return ARTICLE_LINES_WORDS_LETTERS
