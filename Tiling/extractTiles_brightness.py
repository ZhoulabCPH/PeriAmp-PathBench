# Development of script by Narmin Ghaffari, Jakob Kather and James Dolezal

# Narmin Ghaffari, November 2020
# Jakob Nikolas Kather, November 2020
# James Dolezal, March 2019

###############################################################################

# Requires: Openslide (https://openslide.org/download/)

from multiprocessing.dummy import Pool as ThreadPool
from os.path import join, isfile, exists

import os
import threading
import progressbar
import numpy as np
import imageio
from PIL import Image
import argparse

###############################################################################
# Set OPENSLIDE_DLL_DIR when using an external OpenSlide installation on Windows.
openslide_dll_dir = os.environ.get("OPENSLIDE_DLL_DIR")
_openslide_dll_handle = (
    os.add_dll_directory(openslide_dll_dir)
    if os.name == "nt" and openslide_dll_dir
    else None
)
import openslide as ops
###############################################################################
import shapely.geometry as sg
import pandas as pd

# conda config --add channels conda-forge
# conda install shapely
# It is explained in https://conda-forge.org/

import cv2
import json
from math import sqrt

#  HOW TO RUN IT :
# run ----> Configuration per file  -------> command line option ------->
# --px 512 --um 256 -s inputPath  --num_threads 8 -o outputPath
# Pass --no_brightness_filter to disable brightness filtering.

###############################################################################

Image.MAX_IMAGE_PIXELS = 100000000000
NUM_THREADS = 8
DEFAULT_JPG_MPP = 0.2494  # Default microns per pixel for 40x JPEG inputs
JSON_ANNOTATION_SCALE = 10
DONE_MARKER = ".done"
VALID_SLIDE_EXTS = {'.svs', '.mrxs', '.ndpi', '.scn', '.tif', '.tiff', '.mdsx', '.jpg', '.jpeg'}
OPENSLIDE_EXTS = {'.svs', '.mrxs', '.ndpi', '.scn', '.tif', '.tiff', '.mdsx'}
JPG_EXTS = {'.jpg', '.jpeg'}

outputFile = None
report_lock = threading.Lock()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if value in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value, e.g. True/False")


def write_report(text):
    if outputFile is None:
        return
    with report_lock:
        outputFile.write(text)
        outputFile.flush()


def tile_file_stem(tiles_path, case_name, x, y):
    safe_name = case_name.replace('\\', '_').replace('/', '_')
    return join(tiles_path, safe_name + '_(' + str(x) + ',' + str(y) + ')')


def as_rgb(image):
    region = np.asarray(image)
    if region.ndim == 2:
        return np.stack([region, region, region], axis=-1)
    if region.shape[-1] >= 3:
        return region[:, :, :3]
    return region


def is_slide_filename(filename):
    name = filename.lower()
    ext = os.path.splitext(name)[1]
    return ext in VALID_SLIDE_EXTS


def case_name_from_path(path, root=None):
    path = os.path.normpath(path)
    if root and os.path.isdir(root):
        try:
            rel = os.path.relpath(path, os.path.normpath(root))
            if not rel.startswith('..'):
                return os.path.splitext(rel)[0]
        except ValueError:
            pass
    return os.path.splitext(os.path.basename(path))[0]


def collect_done_case_names(blocks_dir):
    done = set()
    if not os.path.isdir(blocks_dir):
        return done
    for root, dirs, files in os.walk(blocks_dir):
        if DONE_MARKER in files:
            rel = os.path.relpath(root, blocks_dir)
            if rel != '.':
                done.add(os.path.normpath(rel))
    return done


def mark_done(tiles_path):
    os.makedirs(tiles_path, exist_ok=True)
    with open(join(tiles_path, DONE_MARKER), 'w', encoding='utf-8') as handle:
        handle.write('ok\n')


###############################################################################

class AnnotationObject:

    def __init__(self, name):
        self.name = name
        self.coordinates = []

    def add_coord(self, coord):
        self.coordinates.append(coord)

    def scaled_area(self, scale):
        return np.multiply(self.coordinates, 1 / scale)

    def print_coord(self):
        for c in self.coordinates:
            print(c)

    def add_shape(self, shape):
        for point in shape:
            self.add_coord(point)


###############################################################################

class JPGSlide:
    def __init__(self, path, mpp):
        self.loaded_image = imageio.imread(path)
        self.dimensions = (
            self.loaded_image.shape[1], self.loaded_image.shape[0])
        self.properties = {ops.PROPERTY_NAME_MPP_X: mpp}
        self.level_dimensions = [self.dimensions]
        self.level_count = 1

    def get_thumbnail(self, dimensions):
        return cv2.resize(self.loaded_image, dsize=dimensions, interpolation=cv2.INTER_CUBIC)

    def read_region(self, topleft, level, window):
        return self.loaded_image[topleft[1]:topleft[1] + window[1],
               topleft[0]:topleft[0] + window[0], ]


###############################################################################

class SlideReader:
    def __init__(self, path, filetype, export_folder=None, pb=None, case_name=None):

        self.coord = []
        self.annotations = []
        self.export_folder = export_folder
        self.pb = pb
        self.p_id = None
        self.extract_px = None
        self.shape = None
        self.basename = os.path.splitext(path)[0]
        self.name = case_name if case_name else os.path.basename(self.basename)
        self.has_anno = True
        self.annPolys = []
        self.ignoredFiles = []
        self.noMPPFlag = 0
        self.NotAbleToLoad = False
        self.slide = None

        ext = '.' + filetype.lower().lstrip('.')
        if ext in OPENSLIDE_EXTS:
            try:
                self.slide = ops.OpenSlide(path)
            except Exception:
                write_report('Unable to read ' + filetype + ',' + path + '\n')
                self.NotAbleToLoad = True
                return
        elif ext in JPG_EXTS:
            self.slide = JPGSlide(path, mpp=DEFAULT_JPG_MPP)
        else:
            write_report('Unsupported file type ' + filetype + ',' + path + '\n')
            self.NotAbleToLoad = True
            return

        thumbs_path = join(export_folder, "thumbs")
        os.makedirs(thumbs_path, exist_ok=True)

        # Load ROIs if available
        roi_path_csv = self.basename + ".csv"
        roi_path_json = self.basename + ".json"

        if exists(roi_path_csv) and not os.path.getsize(roi_path_csv) == 0:
            self.load_csv_roi(roi_path_csv)
        elif exists(roi_path_json) and not os.path.getsize(roi_path_json) == 0:
            self.load_json_roi(roi_path_json)
        else:
            self.has_anno = False

        try:
            self.shape = self.slide.dimensions
            self.filter_dimensions = self.slide.level_dimensions[-1]
            self.filter_magnification = self.filter_dimensions[0] / self.shape[0]
        except Exception:
            write_report('Unable to read slide dimensions' + ',' + path + '\n')
            self.NotAbleToLoad = True
            return

        try:
            goal_thumb_area = 4096 * 4096
            y_x_ratio = self.shape[1] / self.shape[0]
            thumb_x = sqrt(goal_thumb_area / y_x_ratio)
            thumb_y = thumb_x * y_x_ratio
            self.thumb = self.slide.get_thumbnail((int(thumb_x), int(thumb_y)))
            self.thumb_file = join(thumbs_path, self.name + '_thumb.jpg')
            os.makedirs(os.path.dirname(self.thumb_file), exist_ok=True)
            imageio.imwrite(self.thumb_file, self.thumb)
        except Exception:
            write_report('Can not Load thumb File' + ',' + path + '\n')

        try:
            if ops.PROPERTY_NAME_MPP_X in self.slide.properties:
                self.MPP = float(self.slide.properties[ops.PROPERTY_NAME_MPP_X])
            elif 'tiff.XResolution' in self.slide.properties:
                self.MPP = 1 / float(self.slide.properties['tiff.XResolution']) * 10000
            else:
                self.noMPPFlag = 1
                write_report('No PROPERTY_NAME_MPP_X' + ',' + path + '\n')
                return
        except Exception:
            self.noMPPFlag = 1
            write_report('No PROPERTY_NAME_MPP_X' + ',' + path + '\n')
            return

    def close(self):
        slide = getattr(self, 'slide', None)
        if slide is not None and hasattr(slide, 'close'):
            try:
                slide.close()
            except Exception:
                pass
            self.slide = None

    def loaded_correctly(self):
        return bool(self.shape)

    def build_generator(self, size_px, size_um, stride_div, case_name, tiles_path, category, fileSize, export=False,
                        augment=False, brightness_filter=True):

        self.extract_px = int(size_um / self.MPP)
        stride = int(self.extract_px * stride_div)

        slide_x_size = self.shape[0] - self.extract_px
        slide_y_size = self.shape[1] - self.extract_px

        for y in range(0, (self.shape[1] + 1) - self.extract_px, stride):
            for x in range(0, (self.shape[0] + 1) - self.extract_px, stride):
                is_unique = ((y % self.extract_px == 0) and (x % self.extract_px == 0))
                self.coord.append([x, y, is_unique])

        self.annPolys = []
        for annotation in self.annotations:
            if len(annotation.coordinates) < 3:
                continue
            try:
                polygon = sg.Polygon(annotation.coordinates)
                if polygon.is_valid and not polygon.is_empty:
                    self.annPolys.append(polygon)
            except Exception:
                continue

        tile_mask = np.asarray([0 for i in range(len(self.coord))])
        self.tile_mask = None

        def generator():
            for ci in range(len(self.coord)):
                c = self.coord[ci]
                location = (int(c[0]), int(c[1]))

                # Check if the center of the current window lies within any annotation; if not, skip
                if bool(self.annPolys) and not any(
                        [annPoly.contains(sg.Point(int(c[0] + self.extract_px / 2), int(c[1] + self.extract_px / 2)))
                         for annPoly in self.annPolys]):
                    continue

                # Optional brightness filtering.
                # When enabled, skip tiles whose median RGB brightness sum exceeds 660.
                # When disabled, still apply the Canny edge filter below.
                if brightness_filter:
                    filter_px = int(self.extract_px * self.filter_magnification)
                    if filter_px == 0:
                        filter_px = 1
                    try:
                        filter_region = as_rgb(
                            self.slide.read_region(location, self.slide.level_count - 1, [filter_px, filter_px]))
                    except Exception:
                        continue
                    if filter_region.size == 0:
                        continue
                    channel_median = np.median(filter_region, axis=(0, 1))
                    if np.any(np.isnan(channel_median)):
                        continue
                    median_brightness = int(sum(channel_median))
                    if median_brightness > 660:
                        continue

                # Read the region and discard the alpha pixels
                try:
                    region = as_rgb(self.slide.read_region(location, 0, [self.extract_px, self.extract_px]))
                    region = cv2.resize(region, dsize=(size_px, size_px), interpolation=cv2.INTER_CUBIC)
                except Exception:
                    continue

                edge = cv2.Canny(region, 40, 100)
                edge_max = float(np.max(edge))
                if edge_max == 0:
                    continue
                edge = (np.sum(edge / edge_max) / (size_px * size_px)) * 100

                if (edge < 4) or np.isnan(edge):
                    continue

                tile_mask[ci] = 1
                coord_label = ci
                unique_tile = c[2]

                if stride_div == 1:
                    exportFlag = export and unique_tile
                else:
                    exportFlag = export

                if exportFlag:
                    stem = tile_file_stem(tiles_path, case_name, c[0], c[1])
                    imageio.imwrite(stem + '.jpg', region)
                    if augment:
                        imageio.imwrite(stem + '._aug1.jpg', np.rot90(region))
                        imageio.imwrite(stem + '._aug2.jpg', np.flipud(region))
                        imageio.imwrite(stem + '._aug3.jpg', np.flipud(np.rot90(region)))
                        imageio.imwrite(stem + '._aug4.jpg', np.fliplr(region))
                        imageio.imwrite(stem + '._aug5.jpg', np.fliplr(np.rot90(region)))
                        imageio.imwrite(stem + '._aug6.jpg', np.flipud(np.fliplr(region)))
                        imageio.imwrite(stem + '._aug7.jpg', np.flipud(np.fliplr(np.rot90(region))))
                yield region, coord_label, unique_tile

            if self.pb:
                if sum(tile_mask) < 4:
                    write_report('Number of Extracted Tiles < 4 ' + ',' + join(tiles_path, case_name) + '\n')

                print('Remained Slides: ' + str(fileSize))
                print('***************************************************************************')

            self.tile_mask = tile_mask

        return generator, slide_x_size, slide_y_size, stride

    def load_csv_roi(self, path):
        reader = pd.read_csv(path)
        headers = []
        for col in reader.columns:
            headers.append(col.strip())
        if 'X_base' in headers and 'Y_base' in headers:
            index_x = headers.index('X_base')
            index_y = headers.index('Y_base')
        else:
            raise IndexError('Unable to find "X_base" and "Y_base" columns in CSV file.')
        self.annotations.append(AnnotationObject("Object" + str(len(self.annotations))))

        for index, row in reader.iterrows():
            if (str(row[index_x]).strip() == 'X_base' or str(row[index_y]).strip() == 'Y_base'):
                self.annotations.append(AnnotationObject(f"Object{len(self.annotations)}"))
                continue

            x_coord = int(float(row[index_x]))
            y_coord = int(float(row[index_y]))
            self.annotations[-1].add_coord((x_coord, y_coord))

    def load_json_roi(self, path):
        with open(path, "r") as json_file:
            json_data = json.load(json_file)['shapes']
        for shape in json_data:
            area_reduced = np.multiply(shape['points'], JSON_ANNOTATION_SCALE)
            self.annotations.append(AnnotationObject("Object" + str(len(self.annotations))))
            self.annotations[-1].add_shape(area_reduced)


###############################################################################

class Convoluter:
    def __init__(self, size_px, size_um, stride_div, save_folder='', skipws=False, augment=False,
                 brightness_filter=True):

        self.SLIDES = {}
        self.SIZE_PX = size_px
        self.SIZE_UM = size_um
        self.SAVE_FOLDER = save_folder
        self.STRIDE_DIV = stride_div
        self.AUGMENT = augment
        self.skipws = skipws
        self.BRIGHTNESS_FILTER = brightness_filter
        self._iter_lock = threading.Lock()
        self._ignored_lock = threading.Lock()

    def load_slides(self, slides_array, directory="None", category="None"):
        self.fileSize = len(slides_array)
        self.iterator = 0
        print('TOTAL NUMBER OF SLIDES IN THIS FOLDER : ' + str(self.fileSize))

        root = directory if directory and os.path.isdir(directory) else None
        used_names = set()
        for slide in slides_array:
            path = os.path.normpath(slide)
            name = case_name_from_path(path, root)
            if name in used_names:
                parent = os.path.basename(os.path.dirname(path))
                name = parent + '_' + os.path.basename(name)
                suffix = 2
                while name in used_names:
                    name = parent + '_' + os.path.basename(case_name_from_path(path, root)) + '_' + str(suffix)
                    suffix += 1
            used_names.add(name)
            filetype = os.path.splitext(path)[1].lstrip('.').lower()

            self.SLIDES.update({name: {"name": name,
                                       "path": path,
                                       "type": filetype,
                                       "category": category}})

        return self.SLIDES

    def convolute_slides(self):

        '''Parent function to guide convolution across a whole-slide image and execute desired functions.
        '''
        ignoredFile_list = []
        os.makedirs(join(self.SAVE_FOLDER, "BLOCKS"), exist_ok=True)

        pb = progressbar.ProgressBar()
        pool = ThreadPool(NUM_THREADS)
        try:
            pool.map(lambda slide: self.export_tiles(self.SLIDES[slide], pb, ignoredFile_list), self.SLIDES)
        finally:
            pool.close()
            pool.join()
        return pb, ignoredFile_list

    def export_tiles(self, slide, pb, ignoredFile_list):
        case_name = slide['name']
        category = slide['category']
        path = slide['path']
        filetype = slide['type']
        with self._iter_lock:
            self.iterator = self.iterator + 1
            remaining = self.fileSize - self.iterator
        whole_slide = None
        try:
            whole_slide = SlideReader(path, filetype, self.SAVE_FOLDER, pb=pb, case_name=case_name)

            if not whole_slide.has_anno and self.skipws:
                self._record_ignored(ignoredFile_list, path)
                return

            if whole_slide.NotAbleToLoad:
                self._record_ignored(ignoredFile_list, path)
                return

            if whole_slide.noMPPFlag:
                self._record_ignored(ignoredFile_list, path)
                return

            tiles_path = join(whole_slide.export_folder, "BLOCKS", case_name)
            os.makedirs(tiles_path, exist_ok=True)

            if exists(join(tiles_path, DONE_MARKER)):
                print("Folder already filled")
                print('***************************************************************************')
                return

            gen_slice, _, _, _ = whole_slide.build_generator(self.SIZE_PX, self.SIZE_UM, self.STRIDE_DIV, case_name,
                                                             tiles_path, category,
                                                             fileSize=remaining, export=True,
                                                             augment=self.AUGMENT,
                                                             brightness_filter=self.BRIGHTNESS_FILTER)
            for tile, coord, unique in gen_slice():
                pass
            mark_done(tiles_path)
        except Exception as exc:
            self._record_ignored(ignoredFile_list, path)
            write_report('Unhandled error,' + path + ',' + str(exc) + '\n')
            print('Failed: ' + path + ' (' + str(exc) + ')')
        finally:
            if whole_slide is not None:
                whole_slide.close()

    def _record_ignored(self, ignoredFile_list, path):
        with self._ignored_lock:
            ignoredFile_list.append(path)


###############################################################################

def get_args():
    # Approximate MPP values are 0.25 at 40x, 0.5 at 20x, 1 at 10x, and 2 at 5x. At 20x, 512 pixels span about 256 microns.
    parser = argparse.ArgumentParser(
        description='The script to generate the tiles for Whole Slide Image (WSI).')
    parser.add_argument(
        '-s', '--slide', default=r'data/slides',
        help='Path to whole-slide image (SVS or JPG format) or folder of images (SVS or JPG) to analyze.')
    parser.add_argument('-o', '--out', default=r'outputs/tiles/brightness',
                        help='Path to directory in which exported images and data will be saved.')
    parser.add_argument('--skipws', type=str2bool, nargs='?', const=True, default=False,
                        help='Skip slides that have no ROI annotation (CSV/JSON). Accepts True/False.')
    parser.add_argument('--px', type=int, default=256,
                        help='Size of image patches to analyze, in pixels.')
    parser.add_argument('--ov', type=float, default=1.0,
                        help='Stride ratio relative to tile size. 1.0 means no overlap; 0.5 means 50 percent overlap.')
    parser.add_argument('--um', type=float, default=128,
                        help='Size of image patches to analyze, in microns.')
    parser.add_argument('--augment', action="store_true", default=False,
                        help='Augment extracted tiles with flipping/rotating.')

    # Brightness filtering is enabled by default.
    brightness_group = parser.add_mutually_exclusive_group()
    brightness_group.add_argument(
        '--brightness_filter',
        dest='brightness_filter',
        action='store_true',
        help='Enable brightness filtering (median RGB brightness sum > 660 will be skipped).')
    brightness_group.add_argument(
        '--no_brightness_filter',
        dest='brightness_filter',
        action='store_false',
        help='Disable brightness filtering; bright patches are not removed by the brightness rule.')
    parser.set_defaults(brightness_filter=True)

    parser.add_argument('--num_threads', type=int, default=16,
                        help='Number of threads to use when tessellating.')

    return parser.parse_args()


###############################################################################

if __name__ == '__main__':

    args = get_args()
    if not args.out:
        args.out = args.slide
    if args.num_threads:
        NUM_THREADS = args.num_threads

    os.makedirs(args.out, exist_ok=True)

    c = Convoluter(args.px, args.um, args.ov, args.out,
                   augment=args.augment,
                   skipws=args.skipws,
                   brightness_filter=args.brightness_filter)

    outputFile = open(os.path.join(args.out, 'report.txt'), 'a', encoding="utf-8")
    outputFile.write('The Features Selected For this Experiment: ' + '\n')
    outputFile.write('InputPath: ' + args.slide + '\n')
    outputFile.write('OutPutPath: ' + args.out + '\n')
    outputFile.write('Size of image patches to analyze, in pixels: ' + str(args.px) + '\n')
    outputFile.write('Size of image patches to analyze, in microns: ' + str(args.um) + '\n')
    outputFile.write('Size of overlapping: ' + str(args.ov) + '\n')
    outputFile.write('Did we skip WSI: ' + str(args.skipws) + '\n')
    outputFile.write('Brightness filtering enabled: ' + str(args.brightness_filter) + '\n')
    outputFile.write('#########################################################################' + '\n')
    outputFile.flush()

    if isfile(args.slide):
        slide_list = [os.path.normpath(args.slide)]
        c.load_slides(slide_list)
    else:
        slide_list = []
        for root, dirs, files in os.walk(args.slide):
            for file in files:
                if is_slide_filename(file):
                    slide_list.append(os.path.join(root, file))

        done_names = collect_done_case_names(join(args.out, "BLOCKS"))
        if done_names:
            remaining = []
            for s in slide_list:
                case_name = case_name_from_path(s, args.slide)
                if os.path.normpath(case_name) not in done_names:
                    remaining.append(s)
            slide_list = remaining
        c.load_slides(slide_list, args.slide)

    pb, ignoredFile_list = c.convolute_slides()
    if ignoredFile_list:
        outputFile.write('Ignored / failed slides:' + '\n')
        for ignored in ignoredFile_list:
            outputFile.write(ignored + '\n')
    outputFile.close()
