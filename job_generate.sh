#!/bin/bash

python captcha_gen.py --number_words 4 --number_train 200000 --number_test 10100 --lower_case 0
python captcha_gen.py --number_words 4 --number_train 500000 --number_test 10100 --lower_case 1
