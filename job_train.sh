#!/bin/bash

python captcha_train.py --model resnet18
python captcha_train.py --model resnet34
python captcha_train.py --model resnet50
python captcha_train.py --model wide_resnet
