#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2017 Ishit Mehta
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np

DEBUG = True


def NMS(data, thresh=0.8):
    """ Uses NMS to eliminate BBs with IOU values higher than the threshold

    @param: a 2D numpy matrix with each row as (x, y, w, h, c) where c is the confidence
    @param: Threshold for elimnation
    @return: 2D matrix after NMS
    """

    sorted_idxs = np.argsort(data[:, -1])

    if DEBUG:
        print("Boxes before NMS:", len(sorted_idxs))

    areas = (data[:, 2] * data[:, 3]).reshape((len(sorted_idxs), 1))
    data = np.append(data, areas, axis=1)

    resultant_data = np.zeros((1, 6))

    while len(sorted_idxs) > 1:
        curr_box_idx = sorted_idxs[-1]

        x1 = np.maximum(data[curr_box_idx, 0],
                        data[sorted_idxs[:curr_box_idx], 0])
        y1 = np.maximum(data[curr_box_idx, 1],
                        data[sorted_idxs[:curr_box_idx], 1])
        x2 = np.minimum(data[curr_box_idx, 2] + data[curr_box_idx, 0],
                        data[sorted_idxs[:curr_box_idx], 2] + data[sorted_idxs[:curr_box_idx], 0])
        y2 = np.minimum(data[curr_box_idx, 3] + data[curr_box_idx, 1],
                        data[sorted_idxs[:curr_box_idx], 3] + data[sorted_idxs[:curr_box_idx], 1])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        aoi = w * h
        aou = data[sorted_idxs[:curr_box_idx], -1] + \
            data[curr_box_idx, -1] - aoi
        iou = aoi / aou

        delete_idx = np.where(iou > thresh)[0]
        sorted_idxs = np.delete(sorted_idxs, delete_idx)
        sorted_idxs = sorted_idxs[:-1]

        resultant_data = np.append(
            resultant_data, np.asmatrix(data[curr_box_idx, :]), axis=0)

    if DEBUG:
        print("Boxes after NMS:", len(resultant_data[1:, 0]))

    return resultant_data[1:, :-1]


def main():
    a = np.array([
        (12, 84, 140, 212, 1),
        (24, 84, 152, 212, 1),
        (36, 84, 164, 212, 1),
        (12, 96, 140, 224, 1),
        (24, 96, 152, 224, 1),
        (24, 108, 152, 236, 1)]).astype(dtype=float)
    a[:, 2] = a[:, 2] - a[:, 0]
    a[:, 3] = a[:, 3] - a[:, 1]

    print(NMS(a))


if __name__ == '__main__':
    main()
