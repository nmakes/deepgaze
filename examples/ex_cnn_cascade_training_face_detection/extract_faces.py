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
# python preprocess_WIDER.py -i WIDER_train/images -o ./outdir -g wider_face_train_bbx_gt.txt

import cv2
import numpy as np
import os
import argparse


def makeSquare(x, y, w, h):
    """ Convert a rectangle ROI to square.

    @param: left-most column
    @param: top-most row
    @param: width of region
    @param: height of region
    @return: x, y, w, h of the new ROI
    """

    c_x = x + w // 2
    c_y = y + h // 2

    sz = max(w, h)
    x = c_x - sz // 2
    y = c_y - sz // 2

    return int(x), int(y), int(sz), int(sz)


def createPatterns(line, count, im, combinations, out_dir):
    """" Create calibration patterns

    @param: current line which describes the bounding box
    @param: current count of faces
    @param: current iamge
    @param: list of calibration patterns
    @param: output directory where the face images are to be saved
    @return: updated count
    """

    face = np.array(line.split(' ')[:-1]).astype(np.int)
    blur = face[4]
    ill = face[6]
    ocl = face[8]

    # Ignore images with heavy blur, extreme illumination or severe occlusion
    if blur == 2 or ill == 1 or ocl == 2:
        return count

    for (a, b, c) in combinations:
        x = face[0]
        y = face[1]
        w = face[2]
        h = face[3]
        blur = face[4]
        ill = face[6]
        ocl = face[8]
        x = x - (b * w) / a
        y = y - (c * h) / a
        w = w / a
        h = h / a

        x, y, w, h = makeSquare(x, y, w, h)
        im_h, im_w, im_ch = im.shape
        if x >= 0 and x + w < im_w and y >= 0 and y + h <= im_h:
            face_im = im[y: y + h, x:x + w]
            folder = "."
            if(w >= 48):
                folder = os.path.join(out_dir, "48x48")
            elif(w >= 24):
                folder = os.path.join(out_dir, "24x24")
            elif(w >= 12):
                folder = os.path.join(out_dir, "12x12")
            else:
                folder = None

            if folder:
                # Save image in the format: ID_blur_ill_ocl_sn_xn_yn.jpg
                cv2.imwrite(os.path.join(folder, str(count) + "_" + str(blur) + "_" + str(ill) +
                                         "_" + str(ocl) + "_" + str(a) + "_" + str(b) + "_" + str(c) + ".jpg"), face_im)
                count += 1
    return count


def extractFaces(in_dir, out_dir, bb_file, patterns=False):
    """ Exxtract faces from images

    @param: Input directory containing images
    @param: Output directory where faces are to be saved. The output directory 
            should contain 12x12, 24x24 and 48x48 folders
    @param: Ground truth file with information of bounding boxes
    @param: Create calibration patterns for calibration nets
    """

    lines = open(bb_file).read().split('\n')

    # Calibration pattern combinations as per the paper. Refer sec. 3.2.2
    if patterns:
        sn = [0.83, 0.91, 1.0, 1.10, 1.21]
        xn = [-0.17, 0, 0.17]
        yn = [-0.17, 0, 0.17]
        combinations = [(a, b, c) for a in sn for b in xn for c in yn]
    else:
        combinations = [(1.0, 0, 0)]

    i = 0
    count = 0
    while i < len(lines):
        line = lines[i]
        if len(line.split('.')) > 0:
            im = cv2.imread(os.path.join(in_dir, line))
            im_h, im_w, im_ch = im.shape
            i += 1
            face_count = int(lines[i])
            for j in range(1, face_count + 1):
                count = createPatterns(
                    lines[i + j], count, im, combinations, out_dir)
                print("Face:", count)
            i = i + face_count
        i += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--ground_truth', required=True,
                        help='txt file contatining bounding box ground truth values')
    parser.add_argument('-i', '--indir', required=True,
                        help='Input directory containing images with faces')
    parser.add_argument('-o', '--outdir', required=True,
                        help='Output directory containing three folders: 1)12x12 2)24x24 3)48x48')
    args = vars(parser.parse_args())
    extractFaces(args['indir'], args['outdir'], args['ground_truth'], patterns=True)


if __name__ == "__main__":
    main()