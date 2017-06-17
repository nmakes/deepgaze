#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2017 Ishit Mehta
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


def area(box):
    """ Calculates area of a box

    @param: (x, y, w, h) tuple of a box
    @return: area
    """

    (x, y, w, h) = box

    return w * h


def intersection(box1, box2):
    """ Calculate area of intersection of two boxes

    @param: (xmin1, ymin1, xmax1, ymax1) tuple of first box
    @param: (xmin2, ymin2, xmax2, ymax2) tuple of second box
    @return: Area of intersection of box1 and box2
    """

    (xmin1, ymin1, xmax1, ymax1) = box1
    (xmin2, ymin2, xmax2, ymax2) = box2

    left = max(xmin1, xmin2)
    right = min(xmax1, xmax2)
    bottom = min(ymax1, ymax2)
    top = max(ymin1, ymin2)

    if left < right and top < bottom:
        return area((left, top, right - left, bottom - top))
    else:
        return 0


def IOU(box1, box2):
    """ Finds IOU for two bounding boxes

    @param: (x1, y1, w1, h1) tuple of first box
    @param: (x2, y2, w2, h2) tuple of second box
    @return: IOU of box1 and box2
    """

    (x1, y1, w1, h1) = box1
    (x2, y2, w2, h2) = box2

    aoi = intersection((x1, y1, x1 + w1, y1 + h1), (x2, y2, x2 + w2, y2 + h2))
    aou = area(box1) + area(box2) - aoi

    return aoi / aou


def main():
    print(IOU((2, 3, 10, 20), (12, 35, 10, 20)))


if __name__ == '__main__':
    main()
