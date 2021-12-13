import string
import random
from captcha.audio import AudioCaptcha
from captcha.image import ImageCaptcha
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description="Gen data Configuration")
    parser.add_argument(
            "--number_words", dest="number_words", type=int
        )
    parser.add_argument(
            "--number_train", dest="number_train", type=int
        )
    parser.add_argument(
            "--number_test", dest="number_test", type=int
        )
    parser.add_argument(
            "--lower_case", dest="lower_case", type=int
        )
    parser.set_defaults(
        number_words = 4,  
        number_train = 11000,
        number_test = 5100,
        lower_case = 0
    )
    return parser.parse_args()

prog_args = arg_parse()

number_words = prog_args.number_words
number_train = prog_args.number_train
number_test = prog_args.number_test
lower_case = prog_args.lower_case 
image_width = 64
image_height = 24

captcha = ImageCaptcha(image_width, image_height)

if lower_case == 0:
    directory = "data/noLow/"
    for i in range(number_train):
        random_string = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k = number_words))
        captcha.write(random_string, directory + 'train/' + random_string +'.png')

    for i in range(number_test):
        random_string = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k = number_words))
        captcha.write(random_string, directory + 'test/' + random_string +'.png')
else:
    directory = "data/all/"
    for i in range(number_train):
        random_string = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase +
                                     string.digits, k = number_words))
        captcha.write(random_string, directory + 'train/' + random_string +'.png')

    for i in range(number_test):
        random_string = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase +
                                     string.digits, k = number_words))
        captcha.write(random_string, directory + 'test/' + random_string +'.png')