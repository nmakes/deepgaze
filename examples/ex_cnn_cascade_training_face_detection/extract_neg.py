#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2017 Ishit Mehta
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Example usage:
# python extract_neg.py -i ./VOC2007/JPEGImages -o ./outdir

import argparse
import os
import cv2
from random import randint
import numpy as np

patch_size = 48
n_patches = 40

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', required=True)
    parser.add_argument('-o', '--outdir', required=True)

    args = vars(parser.parse_args())

    indir = args['indir']
    outdir = args['outdir']

    count = 1
    for file_name in os.listdir(indir):

        # Skip non-images
        if file_name.split('.')[-1] not in ['jpg', 'JPG', 'jpeg', 'png', 'PNG']:
            continue

        image_path = os.path.join(indir, file_name)
        im = cv2.imread(image_path)

        h, w, c = im.shape
        max_path_range_x = w - patch_size
        max_path_range_y = h - patch_size

        for i in range(n_patches):
            x1 = randint(0, max_path_range_x)
            y1 = randint(0, max_path_range_y)
            patch = np.copy(im[y1:y1 + patch_size, x1:x1 + patch_size])
            out_file = os.path.join(outdir, str(count) + ".jpg")
            cv2.imwrite(out_file, patch)
            count += 1


if __name__ == '__main__':
    main()
