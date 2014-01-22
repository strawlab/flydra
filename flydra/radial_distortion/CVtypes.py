"""
This file started life as cvtypes.py with the following header (that I can't read)

Wrapper-Modul cvtypes.py zur Verwendung der OpenCV-Bibliothek beta5
unter Python, wobei der Zugriff ueber ctypes erfolgt.
Autor: Michael Otto
To do: noch fehlende Strukturen wrappen (z. B. CvKalman)
       noch fehlende Konstanten wrappen (z. B. CV_MAX_DIM_HEAP)
       noch fehlende Makros und Inlinefunktionen wrappen
       ausgiebig testen
Log:   2006/07/25 Dokumentationsstrings hinzugefuegt
       2006/07/10 Fehler in cvGEMM, cvMatMulAdd und cvMatMul beseitigt
       2006/06/28 Modul erzeugt

I hacked it both automatically and by hand to bring it up to date with OpenCV 1.0 and
to use prototype for the functions. I also added from_param methods to allow lists to many
functions that expect a C array.

I checked with Michael and he graciously agreed to let me give it away. This software is
free for any use. If you or your lawyer are stupid enough to believe that Micheal or I have
any liability for it, you should not use it, otherwise be our guest.

Gary Bishop February 2007

Updated 12 May 2007 to include modifications provided by Russell Warren
"""

# --- Importe ----------------------------------------------------------------

import ctypes, os, sys
from ctypes import Structure, Union, POINTER, SetPointerType, CFUNCTYPE, cdll, byref, sizeof
from ctypes import c_char_p, c_double, c_float, c_byte, c_ubyte, c_int, c_void_p, c_ulong
from ctypes import c_uint32, c_short, c_char, c_longlong
import math
# ----Load the DLLs ----------------------------------------------------------

if os.name == 'posix' and sys.platform.startswith('linux'):
    # FIXME: hacked .so names for OpenCV 2.3/2.4
    _cxDLL = cdll.LoadLibrary('libopencv_core.so')
    _cvDLL = cdll.LoadLibrary('libopencv_core.so')
    _hgDLL = cdll.LoadLibrary('libopencv_highgui.so')
    ALL_OPENCV_DLLs = [_cxDLL, _cvDLL, _hgDLL,
                       'libopencv_imgproc.so',
                       ]
elif os.name == 'posix' and sys.platform.startswith('darwin'):
    # FIXME: hacked .dylib names for OpenCV 2.3/2.4
    _cxDLL = cdll.LoadLibrary('libopencv_core.dylib')
    _cvDLL = cdll.LoadLibrary('libopencv_core.dylib')
    _hgDLL = cdll.LoadLibrary('libopencv_highgui.dylib')
    ALL_OPENCV_DLLs = [_cxDLL, _cvDLL, _hgDLL,
                       'libopencv_imgproc.dylib',
                       ]
elif os.name == 'nt':
    _cxDLL = cdll.cxcore100
    _cvDLL = cdll.cv100
    _hgDLL = cdll.highgui100
    ALL_OPENCV_DLLs = [_cxDLL, _cvDLL, _hgDLL]
else:
    raise NotImplemented


# --- CONSTANTS AND STUFF FROM CV.H ------------------------------------------------------

CV_BLUR_NO_SCALE = 0
CV_BLUR = 1
CV_GAUSSIAN = 2
CV_MEDIAN = 3
CV_BILATERAL = 4
CV_INPAINT_NS = 0
CV_INPAINT_TELEA = 1
CV_SCHARR = -1
CV_MAX_SOBEL_KSIZE = 7
CV_BGR2BGRA = 0
CV_RGB2RGBA = CV_BGR2BGRA
CV_BGRA2BGR = 1
CV_RGBA2RGB = CV_BGRA2BGR
CV_BGR2RGBA = 2
CV_RGB2BGRA = CV_BGR2RGBA
CV_RGBA2BGR = 3
CV_BGRA2RGB = CV_RGBA2BGR
CV_BGR2RGB = 4
CV_RGB2BGR = CV_BGR2RGB
CV_BGRA2RGBA = 5
CV_RGBA2BGRA = CV_BGRA2RGBA
CV_BGR2GRAY = 6
CV_RGB2GRAY = 7
CV_GRAY2BGR = 8
CV_GRAY2RGB = CV_GRAY2BGR
CV_GRAY2BGRA = 9
CV_GRAY2RGBA = CV_GRAY2BGRA
CV_BGRA2GRAY = 10
CV_RGBA2GRAY = 11
CV_BGR2BGR565 = 12
CV_RGB2BGR565 = 13
CV_BGR5652BGR = 14
CV_BGR5652RGB = 15
CV_BGRA2BGR565 = 16
CV_RGBA2BGR565 = 17
CV_BGR5652BGRA = 18
CV_BGR5652RGBA = 19
CV_GRAY2BGR565 = 20
CV_BGR5652GRAY = 21
CV_BGR2BGR555 = 22
CV_RGB2BGR555 = 23
CV_BGR5552BGR = 24
CV_BGR5552RGB = 25
CV_BGRA2BGR555 = 26
CV_RGBA2BGR555 = 27
CV_BGR5552BGRA = 28
CV_BGR5552RGBA = 29
CV_GRAY2BGR555 = 30
CV_BGR5552GRAY = 31
CV_BGR2XYZ = 32
CV_RGB2XYZ = 33
CV_XYZ2BGR = 34
CV_XYZ2RGB = 35
CV_BGR2YCrCb = 36
CV_RGB2YCrCb = 37
CV_YCrCb2BGR = 38
CV_YCrCb2RGB = 39
CV_BGR2HSV = 40
CV_RGB2HSV = 41
CV_BGR2Lab = 44
CV_RGB2Lab = 45
CV_BayerBG2BGR = 46
CV_BayerGB2BGR = 47
CV_BayerRG2BGR = 48
CV_BayerGR2BGR = 49
CV_BayerBG2RGB = CV_BayerRG2BGR
CV_BayerGB2RGB = CV_BayerGR2BGR
CV_BayerRG2RGB = CV_BayerBG2BGR
CV_BayerGR2RGB = CV_BayerGB2BGR
CV_BGR2Luv = 50
CV_RGB2Luv = 51
CV_BGR2HLS = 52
CV_RGB2HLS = 53
CV_HSV2BGR = 54
CV_HSV2RGB = 55
CV_Lab2BGR = 56
CV_Lab2RGB = 57
CV_Luv2BGR = 58
CV_Luv2RGB = 59
CV_HLS2BGR = 60
CV_HLS2RGB = 61
CV_COLORCVT_MAX = 100
CV_INTER_NN = 0
CV_INTER_LINEAR = 1
CV_INTER_CUBIC = 2
CV_INTER_AREA = 3
CV_WARP_FILL_OUTLIERS = 8
CV_WARP_INVERSE_MAP = 16
CV_SHAPE_RECT = 0
CV_SHAPE_CROSS = 1
CV_SHAPE_ELLIPSE = 2
CV_SHAPE_CUSTOM = 100
CV_MOP_OPEN = 2
CV_MOP_CLOSE = 3
CV_MOP_GRADIENT = 4
CV_MOP_TOPHAT = 5
CV_MOP_BLACKHAT = 6
CV_TM_SQDIFF = 0
CV_TM_SQDIFF_NORMED = 1
CV_TM_CCORR = 2
CV_TM_CCORR_NORMED = 3
CV_TM_CCOEFF = 4
CV_TM_CCOEFF_NORMED = 5
CV_LKFLOW_PYR_A_READY = 1
CV_LKFLOW_PYR_B_READY = 2
CV_LKFLOW_INITIAL_GUESSES = 4
CV_POLY_APPROX_DP = 0
CV_DOMINANT_IPAN = 1
CV_CONTOURS_MATCH_I1 = 1
CV_CONTOURS_MATCH_I2 = 2
CV_CONTOURS_MATCH_I3 = 3
CV_CONTOUR_TREES_MATCH_I1 = 1
CV_CLOCKWISE = 1
CV_COUNTER_CLOCKWISE = 2
CV_COMP_CORREL = 0
CV_COMP_CHISQR = 1
CV_COMP_INTERSECT = 2
CV_COMP_BHATTACHARYYA = 3
CV_VALUE = 1
CV_ARRAY = 2
CV_DIST_MASK_3 = 3
CV_DIST_MASK_5 = 5
CV_DIST_MASK_PRECISE = 0
CV_THRESH_BINARY = 0      # value = (value > threshold) ? max_value : 0
CV_THRESH_BINARY_INV = 1  # value = (value > threshold) ? 0 : max_value
CV_THRESH_TRUNC = 2       # value = (value > threshold) ? threshold : value
CV_THRESH_TOZERO = 3      # value = (value > threshold) ? value : 0
CV_THRESH_TOZERO_INV = 4  # value = (value > threshold) ? 0 : value
CV_THRESH_MASK = 7
CV_THRESH_OTSU = 8        # use Otsu algorithm to choose the optimal threshold value
CV_ADAPTIVE_THRESH_MEAN_C = 0
CV_ADAPTIVE_THRESH_GAUSSIAN_C = 1
CV_FLOODFILL_FIXED_RANGE = 1 << 16
CV_FLOODFILL_MASK_ONLY = 1 << 17
CV_CANNY_L2_GRADIENT = 1 << 31
CV_HOUGH_STANDARD = 0
CV_HOUGH_PROBABILISTIC = 1
CV_HOUGH_MULTI_SCALE = 2
CV_HOUGH_GRADIENT = 3
CV_HAAR_DO_CANNY_PRUNING = 1
CV_HAAR_SCALE_IMAGE = 2
CV_CALIB_USE_INTRINSIC_GUESS = 1
CV_CALIB_FIX_ASPECT_RATIO = 2
CV_CALIB_FIX_PRINCIPAL_POINT = 4
CV_CALIB_ZERO_TANGENT_DIST = 8
CV_CALIB_CB_ADAPTIVE_THRESH = 1
CV_CALIB_CB_NORMALIZE_IMAGE = 2
CV_CALIB_CB_FILTER_QUADS = 4
CV_FM_7POINT = 1
CV_FM_8POINT = 2
CV_FM_LMEDS_ONLY = 4
CV_FM_RANSAC_ONLY = 8
CV_FM_LMEDS = CV_FM_LMEDS_ONLY + CV_FM_8POINT
CV_FM_RANSAC = CV_FM_RANSAC_ONLY + CV_FM_8POINT

#Viji Periapoilan 4/16/2007 (start)
#Added constants for contour retrieval mode - Apr 19th
CV_RETR_EXTERNAL = 0
CV_RETR_LIST     = 1
CV_RETR_CCOMP    = 2
CV_RETR_TREE     = 3

#Added constants for contour approximation method  - Apr 19th
CV_CHAIN_CODE               = 0
CV_CHAIN_APPROX_NONE        = 1
CV_CHAIN_APPROX_SIMPLE      = 2
CV_CHAIN_APPROX_TC89_L1     = 3
CV_CHAIN_APPROX_TC89_KCOS   = 4
CV_LINK_RUNS                = 5
#Viji Periapoilan 4/16/2007(end)
# --- CONSTANTS AND STUFF FROM highgui.h ----
CV_WINDOW_AUTOSIZE = 1
CV_EVENT_MOUSEMOVE = 0
CV_EVENT_LBUTTONDOWN = 1
CV_EVENT_RBUTTONDOWN = 2
CV_EVENT_MBUTTONDOWN = 3
CV_EVENT_LBUTTONUP = 4
CV_EVENT_RBUTTONUP = 5
CV_EVENT_MBUTTONUP = 6
CV_EVENT_LBUTTONDBLCLK = 7
CV_EVENT_RBUTTONDBLCLK = 8
CV_EVENT_MBUTTONDBLCLK = 9
CV_EVENT_FLAG_LBUTTON = 1
CV_EVENT_FLAG_RBUTTON = 2
CV_EVENT_FLAG_MBUTTON = 4
CV_EVENT_FLAG_CTRLKEY = 8
CV_EVENT_FLAG_SHIFTKEY = 16
CV_EVENT_FLAG_ALTKEY = 32
CV_LOAD_IMAGE_UNCHANGED = -1 # 8 bit, color or gray - deprecated, use CV_LOAD_IMAGE_ANYCOLOR
CV_LOAD_IMAGE_GRAYSCALE =  0 # 8 bit, gray
CV_LOAD_IMAGE_COLOR     =  1 # 8 bit unless combined with CV_LOAD_IMAGE_ANYDEPTH, color
CV_LOAD_IMAGE_ANYDEPTH  =  2 # any depth, if specified on its own gray by itself
                             # equivalent to CV_LOAD_IMAGE_UNCHANGED but can be modified
                             # with CV_LOAD_IMAGE_ANYDEPTH
CV_LOAD_IMAGE_ANYCOLOR  =  4
CV_CVTIMG_FLIP = 1
CV_CVTIMG_SWAP_RB = 2
CV_CAP_ANY = 0     # autodetect
CV_CAP_MIL = 100     # MIL proprietary drivers
CV_CAP_VFW = 200     # platform native
CV_CAP_V4L = 200
CV_CAP_V4L2 = 200
CV_CAP_FIREWARE = 300     # IEEE 1394 drivers
CV_CAP_IEEE1394 = 300
CV_CAP_DC1394 = 300
CV_CAP_CMU1394 = 300
CV_CAP_STEREO = 400     # TYZX proprietary drivers
CV_CAP_TYZX = 400
CV_TYZX_LEFT = 400
CV_TYZX_RIGHT = 401
CV_TYZX_COLOR = 402
CV_TYZX_Z = 403
CV_CAP_QT = 500     # Quicktime
CV_CAP_PROP_POS_MSEC = 0
CV_CAP_PROP_POS_FRAMES = 1
CV_CAP_PROP_POS_AVI_RATIO = 2
CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4
CV_CAP_PROP_FPS = 5
CV_CAP_PROP_FOURCC = 6
CV_CAP_PROP_FRAME_COUNT = 7
CV_CAP_PROP_FORMAT = 8
CV_CAP_PROP_MODE = 9
CV_CAP_PROP_BRIGHTNESS = 10
CV_CAP_PROP_CONTRAST = 11
CV_CAP_PROP_SATURATION = 12
CV_CAP_PROP_HUE = 13
CV_CAP_PROP_GAIN = 14
CV_CAP_PROP_CONVERT_RGB = 15

# --- CONSTANTS AND STUFF FROM opencv_core.h ----
CV_AUTOSTEP = 0x7fffffff
CV_MAX_ARR = 10
CV_NO_DEPTH_CHECK = 1
CV_NO_CN_CHECK = 2
CV_NO_SIZE_CHECK = 4
CV_CMP_EQ = 0
CV_CMP_GT = 1
CV_CMP_GE = 2
CV_CMP_LT = 3
CV_CMP_LE = 4
CV_CMP_NE = 5
CV_CHECK_RANGE = 1
CV_CHECK_QUIET = 2
CV_RAND_UNI = 0
CV_RAND_NORMAL = 1
CV_GEMM_A_T = 1
CV_GEMM_B_T = 2
CV_GEMM_C_T = 4
CV_SVD_MODIFY_A = 1
CV_SVD_U_T = 2
CV_SVD_V_T = 4
CV_LU = 0
CV_SVD = 1
CV_SVD_SYM = 2
CV_COVAR_SCRAMBLED = 0
CV_COVAR_NORMAL = 1
CV_COVAR_USE_AVG = 2
CV_COVAR_SCALE = 4
CV_COVAR_ROWS = 8
CV_COVAR_COLS = 16
CV_PCA_DATA_AS_ROW = 0
CV_PCA_DATA_AS_COL = 1
CV_PCA_USE_AVG = 2
CV_C = 1
CV_L1 = 2
CV_L2 = 4
CV_NORM_MASK = 7
CV_RELATIVE = 8
CV_DIFF = 16
CV_MINMAX = 32
CV_DIFF_C = (CV_DIFF | CV_C)
CV_DIFF_L1 = (CV_DIFF | CV_L1)
CV_DIFF_L2 = (CV_DIFF | CV_L2)
CV_RELATIVE_C = (CV_RELATIVE | CV_C)
CV_RELATIVE_L1 = (CV_RELATIVE | CV_L1)
CV_RELATIVE_L2 = (CV_RELATIVE | CV_L2)
CV_REDUCE_SUM = 0
CV_REDUCE_AVG = 1
CV_REDUCE_MAX = 2
CV_REDUCE_MIN = 3
CV_DXT_FORWARD = 0
CV_DXT_INVERSE = 1
CV_DXT_SCALE = 2     # divide result by size of array
CV_DXT_INV_SCALE = CV_DXT_INVERSE + CV_DXT_SCALE
CV_DXT_INVERSE_SCALE = CV_DXT_INV_SCALE
CV_DXT_ROWS = 4     # transfor each row individually
CV_DXT_MUL_CONJ = 8     # conjugate the second argument of cvMulSpectrums
CV_FRONT = 1
CV_BACK = 0
CV_GRAPH_VERTEX = 1
CV_GRAPH_TREE_EDGE = 2
CV_GRAPH_BACK_EDGE = 4
CV_GRAPH_FORWARD_EDGE = 8
CV_GRAPH_CROSS_EDGE = 16
CV_GRAPH_ANY_EDGE = 30
CV_GRAPH_NEW_TREE = 32
CV_GRAPH_BACKTRACKING = 64
CV_GRAPH_OVER = -1
CV_GRAPH_ALL_ITEMS = -1
CV_GRAPH_ITEM_VISITED_FLAG = 1 << 30
CV_GRAPH_SEARCH_TREE_NODE_FLAG = 1 << 29
CV_GRAPH_FORWARD_EDGE_FLAG = 1 << 28
CV_FILLED = -1
CV_AA = 16
CV_FONT_HERSHEY_SIMPLEX = 0
CV_FONT_HERSHEY_PLAIN = 1
CV_FONT_HERSHEY_DUPLEX = 2
CV_FONT_HERSHEY_COMPLEX = 3
CV_FONT_HERSHEY_TRIPLEX = 4
CV_FONT_HERSHEY_COMPLEX_SMALL = 5
CV_FONT_HERSHEY_SCRIPT_SIMPLEX = 6
CV_FONT_HERSHEY_SCRIPT_COMPLEX = 7
CV_FONT_ITALIC = 16
CV_FONT_VECTOR0 = CV_FONT_HERSHEY_SIMPLEX
CV_ErrModeLeaf = 0     # print error and exit program
CV_ErrModeParent = 1     # print error and continue
CV_ErrModeSilent = 2     # don't print and continue

#------

class DeferredFail:
    def __init__(self,name):
        self.name = name
    def __call__(self,*args,**kwargs):
        raise RuntimeError('CVtypes failure: the function "%s" was not found '
                           'at load time, but this error was deferred until '
                           'you tried to use the function')

# make function prototypes a bit easier to declare
def cfunc(name, _, result, *args):
    '''build and apply a ctypes prototype complete with parameter flags
    e.g.
cvMinMaxLoc = cfunc('cvMinMaxLoc', _cxDLL, None,
                    ('image', POINTER(IplImage), 1),
                    ('min_val', POINTER(double), 2),
                    ('max_val', POINTER(double), 2),
                    ('min_loc', POINTER(CvPoint), 2),
                    ('max_loc', POINTER(CvPoint), 2),
                    ('mask', POINTER(IplImage), 1, None))
means locate cvMinMaxLoc in dll _cxDLL, it returns nothing.
The first argument is an input image. The next 4 arguments are output, and the last argument is
input with an optional value. A typical call might look like:

min_val,max_val,min_loc,max_loc = cvMinMaxLoc(img)
    '''
    atypes = []
    aflags = []
    for arg in args:
        atypes.append(arg[1])
        aflags.append((arg[2], arg[0]) + arg[3:])
    result = None
    for test_dll in ALL_OPENCV_DLLs:
        try:
            result = CFUNCTYPE(result, *atypes)((name, test_dll), tuple(aflags))
        except AttributeError:
            continue
        else:
            break
    if result is None:
        result = DeferredFail(name)
    return result

class ListPOINTER(object):
    '''Just like a POINTER but accept a list of ctype as an argument'''
    def __init__(self, etype):
        self.etype = etype

    def from_param(self, param):
        if isinstance(param, (list,tuple)):
            return (self.etype * len(param))(*param)

class ListPOINTER2(object):
    '''Just like POINTER(POINTER(ctype)) but accept a list of lists of ctype'''
    def __init__(self, etype):
        self.etype = etype

    def from_param(self, param):
        if isinstance(param, (list,tuple)):
            val = (POINTER(self.etype) * len(param))()
            for i,v in enumerate(param):
                if isinstance(v, (list,tuple)):
                    val[i] = (self.etype * len(v))(*v)
                else:
                    raise TypeError, 'nested list or tuple required at %d' % i
            return val
        else:
            raise TypeError, 'list or tuple required'

class ByRefArg(object):
    '''Just like a POINTER but accept an argument and pass it byref'''
    def __init__(self, atype):
        self.atype = atype

    def from_param(self, param):
        return byref(param)

class CallableToFunc(object):
    '''Make the callable argument into a C callback'''
    def __init__(self, cbacktype):
        self.cbacktype = cbacktype

    def from_param(self, param):
        return self.cbacktype(param)

# --- Globale Variablen und Ausnahmen ----------------------------------------

CV_TM_SQDIFF        =0
CV_TM_SQDIFF_NORMED =1
CV_TM_CCORR         =2
CV_TM_CCORR_NORMED  =3
CV_TM_CCOEFF        =4
CV_TM_CCOEFF_NORMED =5

# Image type (IplImage)
IPL_DEPTH_SIGN = 0x80000000 # this trick doesn't work in 2.5 or later

IPL_DEPTH_1U =  1
IPL_DEPTH_8U =  8
IPL_DEPTH_16U = 16
IPL_DEPTH_32F = 32

IPL_DEPTH_8S = -0x7ffffff8  #(IPL_DEPTH_SIGN | 8)
IPL_DEPTH_16S = -0x7ffffff0 #(IPL_DEPTH_SIGN | 16)
IPL_DEPTH_32S = -0x7fffffe0 #(IPL_DEPTH_SIGN | 32)

IPL_DATA_ORDER_PIXEL = 0
IPL_DATA_ORDER_PLANE = 1

IPL_ORIGIN_TL = 0
IPL_ORIGIN_BL = 1

IPL_ALIGN_4BYTES = 4
IPL_ALIGN_8BYTES = 8
IPL_ALIGN_16BYTES = 16
IPL_ALIGN_32BYTES = 32

IPL_ALIGN_DWORD = IPL_ALIGN_4BYTES
IPL_ALIGN_QWORD = IPL_ALIGN_8BYTES

IPL_BORDER_CONSTANT = 0
IPL_BORDER_REPLICATE = 1
IPL_BORDER_REFLECT = 2
IPL_BORDER_WRAP = 3

IPL_IMAGE_HEADER = 1
IPL_IMAGE_DATA = 2
IPL_IMAGE_ROI = 4

CV_TYPE_NAME_IMAGE = "opencv-image"

IPL_DEPTH_64F = 64

# Matrix type (CvMat)
CV_CN_MAX = 4
CV_CN_SHIFT = 3
CV_DEPTH_MAX = (1 << CV_CN_SHIFT)

CV_8U = 0
CV_8S = 1
CV_16U = 2
CV_16S = 3
CV_32S = 4
CV_32F = 5
CV_64F = 6
CV_USRTYPE1 = 7

#
# Sim Harbert 4/9/2007 (start)
# Added following defines:
#
CV_THRESH_BINARY      = 0
CV_THRESH_BINARY_INV  = 1
CV_THRESH_TRUNC       = 2
CV_THRESH_TOZERO      = 3
CV_THRESH_TOZERO_INV  = 4

CV_C           = 1
CV_L1          = 2
CV_L2          = 4

CV_PI = math.pi

# Sim Harbert 4/9/2007 (end)

#Viji Periapoilan 4/16/2007 (start)
# Added the following constants to work with facedetect sample
CV_INTER_NN     = 0 #nearest-neigbor interpolation,
CV_INTER_LINEAR = 1 #bilinear interpolation (used by default)
CV_INTER_CUBIC  = 2 # bicubic interpolation.
CV_INTER_AREA = 3 #resampling using pixel area relation. It is preferred method for image decimation that gives moire-free results. In case of zooming it is similar to CV_INTER_NN method.

#Added constants for contour retrieval mode - Apr 19th
CV_RETR_EXTERNAL = 0
CV_RETR_LIST     = 1
CV_RETR_CCOMP    = 2
CV_RETR_TREE     = 3

#Added constants for contour approximation method  - Apr 19th
CV_CHAIN_CODE               = 0
CV_CHAIN_APPROX_NONE        = 1
CV_CHAIN_APPROX_SIMPLE      = 2
CV_CHAIN_APPROX_TC89_L1     = 3
CV_CHAIN_APPROX_TC89_KCOS   = 4
CV_LINK_RUNS                = 5
#Viji Periapoilan 4/16/2007(end)

#Viji Periapoilan 5/23/2007(start)
CV_WHOLE_SEQ_END_INDEX = 0x3fffffff
#CV_WHOLE_SEQ = CvSlice(0, CV_WHOLE_SEQ_END_INDEX)

#Viji Periapoilan 5/23/2007(end)

def CV_MAKETYPE(depth,cn):
    return ((depth) + (((cn)-1) << CV_CN_SHIFT))
CV_MAKE_TYPE = CV_MAKETYPE

CV_8UC1 = CV_MAKETYPE(CV_8U,1)
CV_8UC2 = CV_MAKETYPE(CV_8U,2)
CV_8UC3 = CV_MAKETYPE(CV_8U,3)
CV_8UC4 = CV_MAKETYPE(CV_8U,4)

CV_8SC1 = CV_MAKETYPE(CV_8S,1)
CV_8SC2 = CV_MAKETYPE(CV_8S,2)
CV_8SC3 = CV_MAKETYPE(CV_8S,3)
CV_8SC4 = CV_MAKETYPE(CV_8S,4)

CV_16UC1 = CV_MAKETYPE(CV_16U,1)
CV_16UC2 = CV_MAKETYPE(CV_16U,2)
CV_16UC3 = CV_MAKETYPE(CV_16U,3)
CV_16UC4 = CV_MAKETYPE(CV_16U,4)

CV_16SC1 = CV_MAKETYPE(CV_16S,1)
CV_16SC2 = CV_MAKETYPE(CV_16S,2)
CV_16SC3 = CV_MAKETYPE(CV_16S,3)
CV_16SC4 = CV_MAKETYPE(CV_16S,4)

CV_32SC1 = CV_MAKETYPE(CV_32S,1)
CV_32SC2 = CV_MAKETYPE(CV_32S,2)
CV_32SC3 = CV_MAKETYPE(CV_32S,3)
CV_32SC4 = CV_MAKETYPE(CV_32S,4)

CV_32FC1 = CV_MAKETYPE(CV_32F,1)
CV_32FC2 = CV_MAKETYPE(CV_32F,2)
CV_32FC3 = CV_MAKETYPE(CV_32F,3)
CV_32FC4 = CV_MAKETYPE(CV_32F,4)

CV_64FC1 = CV_MAKETYPE(CV_64F,1)
CV_64FC2 = CV_MAKETYPE(CV_64F,2)
CV_64FC3 = CV_MAKETYPE(CV_64F,3)
CV_64FC4 = CV_MAKETYPE(CV_64F,4)

CV_AUTO_STEP = 0x7fffffff

CV_MAT_CN_MASK = ((CV_CN_MAX - 1) << CV_CN_SHIFT)
def CV_MAT_CN(flags):
    return ((((flags) & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)
CV_MAT_DEPTH_MASK = (CV_DEPTH_MAX - 1)
def CV_MAT_DEPTH(flags):
    return ((flags) & CV_MAT_DEPTH_MASK)
CV_MAT_TYPE_MASK = (CV_DEPTH_MAX*CV_CN_MAX - 1)
def CV_MAT_TYPE(flags):
    ((flags) & CV_MAT_TYPE_MASK)
CV_MAT_CONT_FLAG_SHIFT = 9
CV_MAT_CONT_FLAG = (1 << CV_MAT_CONT_FLAG_SHIFT)
def CV_IS_MAT_CONT(flags):
    return ((flags) & CV_MAT_CONT_FLAG)
CV_IS_CONT_MAT = CV_IS_MAT_CONT
CV_MAT_TEMP_FLAG_SHIFT = 10
CV_MAT_TEMP_FLAG = (1 << CV_MAT_TEMP_FLAG_SHIFT)
def CV_IS_TEMP_MAT(flags):
    return ((flags) & CV_MAT_TEMP_FLAG)

CV_MAGIC_MASK = 0xFFFF0000
CV_MAT_MAGIC_VAL = 0x42420000
CV_TYPE_NAME_MAT = "opencv-matrix"

# Termination criteria for iterative algorithms
CV_TERMCRIT_ITER = 1
CV_TERMCRIT_NUMBER = CV_TERMCRIT_ITER
CV_TERMCRIT_EPS = 2

# Data structures for persistence (a.k.a serialization) functionality
CV_STORAGE_READ = 0
CV_STORAGE_WRITE = 1
CV_STORAGE_WRITE_TEXT = CV_STORAGE_WRITE
CV_STORAGE_WRITE_BINARY = CV_STORAGE_WRITE
CV_STORAGE_APPEND = 2

CV_MAX_DIM = 32

CV_FILLED = -1
CV_AA = 16

CV_VERSION = "1.0.0"
CV_MAJOR_VERSION = 1
CV_MINOR_VERSION = 0
CV_SUBMINOR_VERSION = 0
CV_WINDOW_AUTOSIZE = 1

CV_CAP_PROP_POS_MSEC      = 0
CV_CAP_PROP_POS_FRAMES    = 1
CV_CAP_PROP_POS_AVI_RATIO = 2
CV_CAP_PROP_FRAME_WIDTH   = 3
CV_CAP_PROP_FRAME_HEIGHT  = 4
CV_CAP_PROP_FPS           = 5
CV_CAP_PROP_FOURCC        = 6
CV_CAP_PROP_FRAME_COUNT   = 7
CV_CAP_PROP_FORMAT        = 8
CV_CAP_PROP_MODE          = 9
CV_CAP_PROP_BRIGHTNESS    =10
CV_CAP_PROP_CONTRAST      =11
CV_CAP_PROP_SATURATION    =12
CV_CAP_PROP_HUE           =13
CV_CAP_PROP_GAIN          =14
CV_CAP_PROP_CONVERT_RGB   =15

def CV_FOURCC(c1,c2,c3,c4):
    return (((ord(c1))&255) + (((ord(c2))&255)<<8) + (((ord(c3))&255)<<16) + (((ord(c4))&255)<<24))

#Viji Periapoilan 5/21/2007(start)
#/****************************************************************************************\
#*                                    Sequence types                                      *
#\****************************************************************************************/

CV_SEQ_MAGIC_VAL            = 0x42990000

#define CV_IS_SEQ(seq) \
#    ((seq) != NULL && (((CvSeq*)(seq))->flags & CV_MAGIC_MASK) == CV_SEQ_MAGIC_VAL)

CV_SET_MAGIC_VAL           = 0x42980000
#define CV_IS_SET(set) \
#    ((set) != NULL && (((CvSeq*)(set))->flags & CV_MAGIC_MASK) == CV_SET_MAGIC_VAL)

CV_SEQ_ELTYPE_BITS         = 9
CV_SEQ_ELTYPE_MASK         =  ((1 << CV_SEQ_ELTYPE_BITS) - 1)
CV_SEQ_ELTYPE_POINT        =  CV_32SC2  #/* (x,y) */
CV_SEQ_ELTYPE_CODE         = CV_8UC1   #/* freeman code: 0..7 */
CV_SEQ_ELTYPE_GENERIC      =  0
CV_SEQ_ELTYPE_PTR          =  CV_USRTYPE1
CV_SEQ_ELTYPE_PPOINT       =  CV_SEQ_ELTYPE_PTR  #/* &(x,y) */
CV_SEQ_ELTYPE_INDEX        =  CV_32SC1  #/* #(x,y) */
CV_SEQ_ELTYPE_GRAPH_EDGE   =  0  #/* &next_o, &next_d, &vtx_o, &vtx_d */
CV_SEQ_ELTYPE_GRAPH_VERTEX =  0  #/* first_edge, &(x,y) */
CV_SEQ_ELTYPE_TRIAN_ATR    =  0  #/* vertex of the binary tree   */
CV_SEQ_ELTYPE_CONNECTED_COMP= 0  #/* connected component  */
CV_SEQ_ELTYPE_POINT3D      =  CV_32FC3  #/* (x,y,z)  */

CV_SEQ_KIND_BITS           = 3
CV_SEQ_KIND_MASK           = (((1 << CV_SEQ_KIND_BITS) - 1)<<CV_SEQ_ELTYPE_BITS)


#/* types of sequences */
CV_SEQ_KIND_GENERIC        = (0 << CV_SEQ_ELTYPE_BITS)
CV_SEQ_KIND_CURVE          = (1 << CV_SEQ_ELTYPE_BITS)
CV_SEQ_KIND_BIN_TREE       = (2 << CV_SEQ_ELTYPE_BITS)

#Viji Periapoilan 5/21/2007(end)


# hack the ctypes.Structure class to include printing the fields
class _Structure(Structure):
    def __repr__(self):
        '''Print the fields'''
        res = []
        for field in self._fields_:
            res.append('%s=%s' % (field[0], repr(getattr(self, field[0]))))
        return self.__class__.__name__ + '(' + ','.join(res) + ')'
    @classmethod
    def from_param(cls, obj):
        '''Magically construct from a tuple'''
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, tuple):
            return cls(*obj)
        raise TypeError

# --- Klassen- und Funktionsdefinitionen -------------------------------------

# 2D point with integer coordinates
class CvPoint(_Structure):
    _fields_ = [("x", c_int),
                ("y", c_int)]

# 2D point with floating-point coordinates
class CvPoint2D32f(_Structure):
    _fields_ = [("x", c_float),
                ("y", c_float)]

# 3D point with floating-point coordinates
class CvPoint3D32f(_Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("z", c_float)]

# 2D point with double precision floating-point coordinates
class CvPoint2D64f(_Structure):
    _fields_ = [("x", c_double),
                ("y", c_double)]
CvPoint2D64d = CvPoint2D64f

# 3D point with double precision floating-point coordinates
class CvPoint3D64f(_Structure):
    _fields_ = [("x", c_double),
                ("y", c_double),
                ("z", c_double)]
CvPoint3D64d = CvPoint3D64f

# pixel-accurate size of a rectangle
class CvSize(_Structure):
    _fields_ = [("width", c_int),
                ("height", c_int)]

# sub-pixel accurate size of a rectangle
class CvSize2D32f(_Structure):
    _fields_ = [("width", c_float),
                ("height", c_float)]

# offset and size of a rectangle
class CvRect(_Structure):
    _fields_ = [("x", c_int),
                ("y", c_int),
                ("width", c_int),
                ("height", c_int)]
    def bloat(self, s):
        return CvRect(self.x-s, self.y-s, self.width+2*s, self.height+2*s)

# A container for 1-,2-,3- or 4-tuples of numbers
class CvScalar(_Structure):
    _fields_ = [("val", c_double * 4)]
    def __init__(self, *vals):
        '''Enable initialization with multiple parameters instead of just a tuple'''
        if len(vals) == 1:
            super(CvScalar, self).__init__(vals[0])
        else:
            super(CvScalar, self).__init__(vals)

# Termination criteria for iterative algorithms
class CvTermCriteria(_Structure):
    _fields_ = [("type", c_int),
                ("max_iter", c_int),
                ("epsilon", c_double)]

# Multi-channel matrix
class CvMat(Structure):
    _fields_ = [("type", c_int),
                ("step", c_int),
                ("refcount", c_void_p),
                ("hdr_refcount", c_int),
                ("data", c_void_p),
                ("rows", c_int),
                ("cols", c_int)]

# Multi-dimensional dense multi-channel matrix
class CvMatNDdata(Union):
    _fields_ = [("ptr", POINTER(c_ubyte)),
                ("s", POINTER(c_short)),
                ("i", POINTER(c_int)),
                ("fl", POINTER(c_float)),
                ("db", POINTER(c_double))]
class CvMatNDdim(Structure):
    _fields_ = [("size", c_int),
                ("step", c_int)]
class CvMatND(Structure):
    _fields_ = [("type", c_int),
                ("dims", c_int),
                ("refcount", c_void_p),
                ("data", CvMatNDdata),
                ("dim", CvMatNDdim*CV_MAX_DIM)]

# IPL image header
class IplImage(Structure):
    _fields_ = [("nSize", c_int),
                ("ID", c_int),
                ("nChannels", c_int),
                ("alphaChannel", c_int),
                ("depth", c_int),
                ("colorModel", c_char * 4),
                ("channelSeq", c_char * 4),
                ("dataOrder", c_int),
                ("origin", c_int),
                ("align", c_int),
                ("width", c_int),
                ("height", c_int),
                ("roi", c_void_p),
                ("maskROI", c_void_p),
                ("imageID", c_void_p),
                ("tileInfo", c_void_p),
                ("imageSize", c_int),
                ("imageData", c_void_p),
                ("widthStep", c_int),
                ("BorderMode", c_int * 4),
                ("BorderConst", c_int * 4),
                ("imageDataOrigin", c_char_p)]

    def __repr__(self):
        '''Print the fields'''
        res = []
        for field in self._fields_:
            if field[0] in ['imageData', 'imageDataOrigin']: continue
            res.append('%s=%s' % (field[0], repr(getattr(self, field[0]))))
        return self.__class__.__name__ + '(' + ','.join(res) + ')'

# List of attributes
class CvAttrList(Structure):
    _fields_ = [("attr", c_void_p),
                ("next", c_void_p)]

# Memory storage
_lpCvMemStorage = POINTER("CvMemStorage")
class CvMemStorage(Structure):
    _fields_ = [("signature", c_int),
                ("bottom", c_void_p),
                ("top", c_void_p),
                ("parent", _lpCvMemStorage),
                ("block_size", c_int),
                ("free_space", c_int)]
SetPointerType(_lpCvMemStorage, CvMemStorage)

class CvMemStoragePos(Structure):
    _fields_ = []

# Sequence
class CvSeq(Structure):
    _fields_ = [("flags", c_int),
                ("header_size", c_int),
                ("h_prev", c_void_p),
                ("h_next", c_void_p),
                ("v_prev", c_void_p),
                ("v_next", c_void_p),
                ("total", c_int),
                ("elem_size", c_int),
                ("block_max", c_void_p),
                ("ptr", c_void_p),
                ("delta_elems", c_int),
                ("storage", POINTER(CvMemStorage)),
                ("free_blocks", c_void_p),
                ("first", c_void_p)]

    def hrange(self):
        """
        generator function iterating along h_next
        """
    	s = self
    	t = type(self)
    	while s:
    	    yield s
    	    s = ctypes.cast(s.h_next , POINTER(CvSeq))

class CvSeqBlock(Structure):
    _fields_ = []

class CvSeqWriter(Structure):
    _fields_ = []

class CvSeqReader(Structure):
    _fields_ = []

# File storage
class CvFileStorage(Structure):
    _fields_ = []

# not implemented yet
class CvSparseMat(Structure):
    _fields_ = []

class CvContourScanner(Structure):
    _fields_ = []

class CvHistogram(Structure):
    _fields_ = [('type', c_int),
                ('bins', c_void_p)]

class CvString(Structure):
    _fields_ = []

class CvSlice(Structure):
    _fields_ = [('start_index', c_int),
                ('end_index', c_int)]

class CvSET(Structure):
    _fields_ = []

class CvGraph(Structure):
    _fields_ = []

class CvGraphEdge(Structure):
    _fields_ = []

class CvGraphScanner(Structure):
    _fields_ = []

class CvFileNode(Structure):
    _fields_ = []

class CvStringHashNode(Structure):
    _fields_ = []

class CvTypeInfo(Structure):
    _fields_ = []

class CvContourTree(Structure):
    _fields_ = []

class CvBox2D(_Structure):
    _fields_ = [('center', CvPoint2D32f),
                ('size', CvSize2D32f),
                ('angle', c_float)]

class CvSubdiv2DPoint(Structure):
    _fields_ = []

class CvSubdiv2DPointLocation(Structure):
    _fields_ = []

class CvKalman(Structure):
    _fields_ = []

class CvConDensation(Structure):
    _fields_ = []

class CvHaarClassifierCascade(Structure):
    _fields_ = []

class CvPOSITObject(Structure):
    _fields_ = []

class CvMatr32f(Structure):
    _fields_ = []

class CvVect32f(Structure):
    _fields_ = []

class CvCapture(Structure):
    _fields_ = []

class CvVideoWriter(Structure):
    _fields_ = []

class CvSetElem(Structure):
    _fields_ = []

class CvGraphVtx(Structure):
    _fields_ = []

class CvTreeNodeIterator(Structure):
    _fields_ = []

class CvFont(Structure):
    _fields_ = []

class CvLineIterator(Structure):
    _fields_ = []

class CvModuleInfo(Structure):
    _fields_ = []

class IplConvKernel(Structure):
    _fields_ = []

class CvConnectedComp(_Structure):
    _fields_ = [('area', c_double),
                ('value', CvScalar),
                ('rect', CvRect),
                ('contour', POINTER(CvSeq))]

class CvMOMENTS(Structure):
    _fields_ = []

class CvHuMoments(Structure):
    _fields_ = []

class CvChain(Structure):
    _fields_ = []

class CvChainPtReader(Structure):
    _fields_ = []

#Added the fields for contour - Start
class CvContour(Structure):
    _fields_ = [("flags", c_int),
                ("header_size", c_int),
                ("h_prev", c_void_p),
                ("h_next", POINTER(CvSeq)),
                ("v_prev", c_void_p),
                ("v_next", c_void_p),
                ("total", c_int),
                ("elem_size", c_int),
                ("block_max", c_void_p),
                ("ptr", c_void_p),
                ("delta_elems", c_int),
                ("storage", POINTER(CvMemStorage)),
                ("free_blocks", c_void_p),
                ("first", c_void_p),
                ('rect', CvRect),
                ("color", c_int),
                ("reserved", c_int * 3)]
#Added the fields for contour - End - Viji - Apr 19, 2007

class CvCmpFunc(Structure):
    _fields_ = []

class CvDistanceFunction(Structure):
    _fields_ = []

class CvSubdiv2D(Structure):
    _fields_ = []

class CvSubdiv2DEdge(Structure):
    _fields_ = []


# --- 1 Operations on Arrays -------------------------------------------------

# --- 1.1 Initialization -----------------------------------------------------

# Creates header and allocates data
cvCreateImage = cfunc('cvCreateImage', _cxDLL, POINTER(IplImage),
    ('size', CvSize, 1), # CvSize size
    ('depth', c_int, 1), # int depth
    ('channels', c_int, 1), # int channels
)

# Allocates, initializes, and returns structure IplImage
cvCreateImageHeader = cfunc('cvCreateImageHeader', _cxDLL, POINTER(IplImage),
    ('size', CvSize, 1), # CvSize size
    ('depth', c_int, 1), # int depth
    ('channels', c_int, 1), # int channels
)

# Releases header
cvReleaseImageHeader = cfunc('cvReleaseImageHeader', _cxDLL, None,
    ('image', ByRefArg(POINTER(IplImage)), 1), # IplImage** image
)

# Releases header and image data
cvReleaseImage = cfunc('cvReleaseImage', _cxDLL, None,
    ('image', ByRefArg(POINTER(IplImage)), 1), # IplImage** image
)

# Initializes allocated by user image header
cvInitImageHeader = cfunc('cvInitImageHeader', _cxDLL, POINTER(IplImage),
    ('image', POINTER(IplImage), 1), # IplImage* image
    ('size', CvSize, 1), # CvSize size
    ('depth', c_int, 1), # int depth
    ('channels', c_int, 1), # int channels
    ('origin', c_int, 1, 0), # int origin
    ('align', c_int, 1, 4), # int align
)

# Makes a full copy of image
cvCloneImage = cfunc('cvCloneImage', _cxDLL, POINTER(IplImage),
    ('image', POINTER(IplImage), 1), # const IplImage* image
)

# Sets channel of interest to given value
cvSetImageCOI = cfunc('cvSetImageCOI', _cxDLL, None,
    ('image', POINTER(IplImage), 1), # IplImage* image
    ('coi', c_int, 1), # int coi
)

# Returns index of channel of interest
cvGetImageCOI = cfunc('cvGetImageCOI', _cxDLL, c_int,
    ('image', POINTER(IplImage), 1), # const IplImage* image
)

# Sets image ROI to given rectangle
cvSetImageROI = cfunc('cvSetImageROI', _cxDLL, None,
    ('image', POINTER(IplImage), 1), # IplImage* image
    ('rect', CvRect, 1), # CvRect rect
)

# Releases image ROI
cvResetImageROI = cfunc('cvResetImageROI', _cxDLL, None,
    ('image', POINTER(IplImage), 1), # IplImage* image
)

# Returns image ROI coordinates
cvGetImageROI = cfunc('cvGetImageROI', _cxDLL, CvRect,
    ('image', POINTER(IplImage), 1), # const IplImage* image
)

# Creates new matrix
cvCreateMat = cfunc('cvCreateMat', _cxDLL, POINTER(CvMat),
    ('rows', c_int, 1), # int rows
    ('cols', c_int, 1), # int cols
    ('type', c_int, 1), # int type
)

# Creates new matrix header
cvCreateMatHeader = cfunc('cvCreateMatHeader', _cxDLL, POINTER(CvMat),
    ('rows', c_int, 1), # int rows
    ('cols', c_int, 1), # int cols
    ('type', c_int, 1), # int type
)

# Deallocates matrix
cvReleaseMat = cfunc('cvReleaseMat', _cxDLL, None,
    ('mat', ByRefArg(POINTER(CvMat)), 1), # CvMat** mat
)

# Initializes matrix header
cvInitMatHeader = cfunc('cvInitMatHeader', _cxDLL, POINTER(CvMat),
    ('mat', POINTER(CvMat), 1), # CvMat* mat
    ('rows', c_int, 1), # int rows
    ('cols', c_int, 1), # int cols
    ('type', c_int, 1), # int type
    ('data', c_void_p, 1, None), # void* data
    ('step', c_int, 1), # int step
)

# Creates matrix copy
cvCloneMat = cfunc('cvCloneMat', _cxDLL, POINTER(CvMat),
    ('mat', POINTER(CvMat), 1), # const CvMat* mat
)

# Creates multi-dimensional dense array
cvCreateMatND = cfunc('cvCreateMatND', _cxDLL, POINTER(CvMatND),
    ('dims', c_int, 1), # int dims
    ('sizes', POINTER(c_int), 1), # const int* sizes
    ('type', c_int, 1), # int type
)

# Creates new matrix header
cvCreateMatNDHeader = cfunc('cvCreateMatNDHeader', _cxDLL, POINTER(CvMatND),
    ('dims', c_int, 1), # int dims
    ('sizes', POINTER(c_int), 1), # const int* sizes
    ('type', c_int, 1), # int type
)

# Initializes multi-dimensional array header
cvInitMatNDHeader = cfunc('cvInitMatNDHeader', _cxDLL, POINTER(CvMatND),
    ('mat', POINTER(CvMatND), 1), # CvMatND* mat
    ('dims', c_int, 1), # int dims
    ('sizes', POINTER(c_int), 1), # const int* sizes
    ('type', c_int, 1), # int type
    ('data', c_void_p, 1, None), # void* data
)

# Creates full copy of multi-dimensional array
cvCloneMatND = cfunc('cvCloneMatND', _cxDLL, POINTER(CvMatND),
    ('mat', POINTER(CvMatND), 1), # const CvMatND* mat
)

# Allocates array data
cvCreateData = cfunc('cvCreateData', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
)

# Releases array data
cvReleaseData = cfunc('cvReleaseData', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
)

# Assigns user data to the array header
cvSetData = cfunc('cvSetData', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
    ('data', c_void_p, 1), # void* data
    ('step', c_int, 1), # int step
)

# Retrieves low-level information about the array
cvGetRawData = cfunc('cvGetRawData', _cxDLL, None,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('data', POINTER(POINTER(c_byte)), 1), # uchar** data
    ('step', POINTER(c_int), 1, None), # int* step
    ('roi_size', POINTER(CvSize), 1, None), # CvSize* roi_size
)

# Returns matrix header for arbitrary array
cvGetMat = cfunc('cvGetMat', _cxDLL, POINTER(CvMat),
    ('arr', c_void_p, 1), # const CvArr* arr
    ('header', POINTER(CvMat), 1), # CvMat* header
    ('coi', POINTER(c_int), 1, None), # int* coi
    ('allowND', c_int, 1, 0), # int allowND
)

# Returns image header for arbitrary array
cvGetImage = cfunc('cvGetImage', _cxDLL, POINTER(IplImage),
    ('arr', c_void_p, 1), # const CvArr* arr
    ('image_header', POINTER(IplImage), 1), # IplImage* image_header
)

# Creates sparse array
cvCreateSparseMat = cfunc('cvCreateSparseMat', _cxDLL, POINTER(CvSparseMat),
    ('dims', c_int, 1), # int dims
    ('sizes', POINTER(c_int), 1), # const int* sizes
    ('type', c_int, 1), # int type
)

# Deallocates sparse array
cvReleaseSparseMat = cfunc('cvReleaseSparseMat', _cxDLL, None,
    ('mat', ByRefArg(POINTER(CvSparseMat)), 1), # CvSparseMat** mat
)

# Creates full copy of sparse array
cvCloneSparseMat = cfunc('cvCloneSparseMat', _cxDLL, POINTER(CvSparseMat),
    ('mat', POINTER(CvSparseMat), 1), # const CvSparseMat* mat
)

# --- 1.2 Accessing Elements and sub-Arrays ----------------------------------

# Returns matrix header corresponding to the rectangular sub-array of input image or matrix
cvGetSubRect = cfunc('cvGetSubRect', _cxDLL, POINTER(CvMat),
    ('arr', c_void_p, 1), # const CvArr* arr
    ('submat', POINTER(CvMat), 2), # CvMat* submat
    ('rect', CvRect, 1), # CvRect rect
)

# Returns array row or row span
cvGetRows = cfunc('cvGetRows', _cxDLL, POINTER(CvMat),
    ('arr', c_void_p, 1), # const CvArr* arr
    ('submat', POINTER(CvMat), 1), # CvMat* submat
    ('start_row', c_int, 1), # int start_row
    ('end_row', c_int, 1), # int end_row
    ('delta_row', c_int, 1, 1), # int delta_row
)

# Returns array column or column span
cvGetCols = cfunc('cvGetCols', _cxDLL, POINTER(CvMat),
    ('arr', c_void_p, 1), # const CvArr* arr
    ('submat', POINTER(CvMat), 1), # CvMat* submat
    ('start_col', c_int, 1), # int start_col
    ('end_col', c_int, 1), # int end_col
)

# Returns one of array diagonals
cvGetDiag = cfunc('cvGetDiag', _cxDLL, POINTER(CvMat),
    ('arr', c_void_p, 1), # const CvArr* arr
    ('submat', POINTER(CvMat), 1), # CvMat* submat
    ('diag', c_int, 1, 0), # int diag
)

# Returns size of matrix or image ROI
cvGetSize = cfunc('cvGetSize', _cxDLL, CvSize,
    ('arr', c_void_p, 1), # const CvArr* arr
)

### Initializes sparse array elements iterator
##cvInitSparseMatIterator = _cxDLL.cvInitSparseMatIterator
##cvInitSparseMatIterator.restype = POINTER(CvSparseNode) # CvSparseNode*
##cvInitSparseMatIterator.argtypes = [
##    c_void_p, # const CvSparseMat* mat
##    c_void_p # CvSparseMatIterator* mat_iterator
##    ]
##
### Initializes sparse array elements iterator
##cvGetNextSparseNode = _cxDLL.cvGetNextSparseNode
##cvGetNextSparseNode.restype = POINTER(CvSparseNode) # CvSparseNode*
##cvGetNextSparseNode.argtypes = [
##    c_void_p # CvSparseMatIterator* mat_iterator
##    ]

# Returns type of array elements
cvGetElemType = cfunc('cvGetElemType', _cxDLL, c_int,
    ('arr', c_void_p, 1), # const CvArr* arr
)

# Return number of array dimensions and their sizes or the size of particular dimension
cvGetDims = cfunc('cvGetDims', _cxDLL, c_int,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('sizes', POINTER(c_int), 1, None), # int* sizes
)

cvGetDimSize = cfunc('cvGetDimSize', _cxDLL, c_int,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('index', c_int, 1), # int index
)

# Return pointer to the particular array element
cvPtr1D = cfunc('cvPtr1D', _cxDLL, c_void_p,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('idx0', c_int, 1), # int idx0
    ('type', POINTER(c_int), 1, None), # int* type
)

cvPtr2D = cfunc('cvPtr2D', _cxDLL, c_void_p,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('idx0', c_int, 1), # int idx0
    ('idx1', c_int, 1), # int idx1
    ('type', POINTER(c_int), 1, None), # int* type
)

cvPtr3D = cfunc('cvPtr3D', _cxDLL, c_void_p,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('idx0', c_int, 1), # int idx0
    ('idx1', c_int, 1), # int idx1
    ('idx2', c_int, 1), # int idx2
    ('type', POINTER(c_int), 1, None), # int* type
)

cvPtrND = cfunc('cvPtrND', _cxDLL, c_void_p,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('idx', POINTER(c_int), 1), # int* idx
    ('type', POINTER(c_int), 1, None), # int* type
    ('create_node', c_int, 1, 1), # int create_node
    ('precalc_hashval', POINTER(c_uint32), 1, None), # unsigned* precalc_hashval
)

# Return the particular array element
cvGet1D = cfunc('cvGet1D', _cxDLL, CvScalar,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('idx0', c_int, 1), # int idx0
)

cvGet2D = cfunc('cvGet2D', _cxDLL, CvScalar,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('idx0', c_int, 1), # int idx0
    ('idx1', c_int, 1), # int idx1
)

cvGet3D = cfunc('cvGet3D', _cxDLL, CvScalar,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('idx0', c_int, 1), # int idx0
    ('idx1', c_int, 1), # int idx1
    ('idx2', c_int, 1), # int idx2
)

cvGetND = cfunc('cvGetND', _cxDLL, CvScalar,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('idx', POINTER(c_int), 1), # int* idx
)

# Return the particular element of single-channel array
cvGetReal1D = cfunc('cvGetReal1D', _cxDLL, c_double,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('idx0', c_int, 1), # int idx0
)

cvGetReal2D = cfunc('cvGetReal2D', _cxDLL, c_double,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('idx0', c_int, 1), # int idx0
    ('idx1', c_int, 1), # int idx1
)

cvGetReal3D = cfunc('cvGetReal3D', _cxDLL, c_double,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('idx0', c_int, 1), # int idx0
    ('idx1', c_int, 1), # int idx1
    ('idx2', c_int, 1), # int idx2
)

cvGetRealND = cfunc('cvGetRealND', _cxDLL, c_double,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('idx', POINTER(c_int), 1), # int* idx
)

# Change the particular array element
cvSet1D = cfunc('cvSet1D', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
    ('idx0', c_int, 1), # int idx0
    ('value', CvScalar, 1), # CvScalar value
)

cvSet2D = cfunc('cvSet2D', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
    ('idx0', c_int, 1), # int idx0
    ('idx1', c_int, 1), # int idx1
    ('value', CvScalar, 1), # CvScalar value
)

cvSet3D = cfunc('cvSet3D', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
    ('idx0', c_int, 1), # int idx0
    ('idx1', c_int, 1), # int idx1
    ('idx2', c_int, 1), # int idx2
    ('value', CvScalar, 1), # CvScalar value
)

cvSetND = cfunc('cvSetND', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
    ('idx', POINTER(c_int), 1), # int* idx
    ('value', CvScalar, 1), # CvScalar value
)

# Change the particular array element
cvSetReal1D = cfunc('cvSetReal1D', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
    ('idx0', c_int, 1), # int idx0
    ('value', c_double, 1), # double value
)

cvSetReal2D = cfunc('cvSetReal2D', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
    ('idx0', c_int, 1), # int idx0
    ('idx1', c_int, 1), # int idx1
    ('value', c_double, 1), # double value
)

cvSetReal3D = cfunc('cvSetReal3D', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
    ('idx0', c_int, 1), # int idx0
    ('idx1', c_int, 1), # int idx1
    ('idx2', c_int, 1), # int idx2
    ('value', c_double, 1), # double value
)

cvSetRealND = cfunc('cvSetRealND', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
    ('idx', POINTER(c_int), 1), # int* idx
    ('value', c_double, 1), # double value
)

# Clears the particular array element
cvClearND = cfunc('cvClearND', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
    ('idx', POINTER(c_int), 1), # int* idx
)

# --- 1.3 Copying and Filling ------------------------------------------------

# Copies one array to another
cvCopy = cfunc('cvCopy', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Sets every element of array to given value
cvSet = cfunc('cvSet', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
    ('value', CvScalar, 1), # CvScalar value
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Clears the array
cvSetZero = cfunc('cvSetZero', _cxDLL, None,
    ('arr', c_void_p, 1), # CvArr* arr
)

cvZero = cvSetZero

# --- 1.4 Transforms and Permutations ----------------------------------------

# Changes shape of matrix/image without copying data
cvReshape = cfunc('cvReshape', _cxDLL, POINTER(CvMat),
    ('arr', c_void_p, 1), # const CvArr* arr
    ('header', POINTER(CvMat), 1), # CvMat* header
    ('new_cn', c_int, 1), # int new_cn
    ('new_rows', c_int, 1, 0), # int new_rows
)

# Changes shape of multi-dimensional array w/o copying data
cvReshapeMatND = cfunc('cvReshapeMatND', _cxDLL, c_void_p,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('sizeof_header', c_int, 1), # int sizeof_header
    ('header', c_void_p, 1), # CvArr* header
    ('new_cn', c_int, 1), # int new_cn
    ('new_dims', c_int, 1), # int new_dims
    ('new_sizes', POINTER(c_int), 1), # int* new_sizes
)

# Fill destination array with tiled source array
cvRepeat = cfunc('cvRepeat', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
)

# Flip a 2D array around vertical, horizontall or both axises
cvFlip = cfunc('cvFlip', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1, None), # CvArr* dst
    ('flip_mode', c_int, 1, 0), # int flip_mode
)

# Divides multi-channel array into several single-channel arrays or extracts a single channel from the array
cvSplit = cfunc('cvSplit', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst0', c_void_p, 1, None), # CvArr* dst0
    ('dst1', c_void_p, 1, None), # CvArr* dst1
    ('dst2', c_void_p, 1, None), # CvArr* dst2
    ('dst3', c_void_p, 1, None), # CvArr* dst3
)

# Composes multi-channel array from several single-channel arrays or inserts a single channel into the array
cvMerge = cfunc('cvMerge', _cxDLL, None,
    ('src0', c_void_p, 1), # const CvArr* src0
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('src3', c_void_p, 1), # const CvArr* src3
    ('dst', c_void_p, 1), # CvArr* dst
)

# --- 1.5 Arithmetic, Logic and Comparison -----------------------------------

# Performs look-up table transform of array
cvLUT = cfunc('cvLUT', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('lut', c_void_p, 1), # const CvArr* lut
)

# Converts one array to another with optional linear transformation
cvConvertScale = cfunc('cvConvertScale', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('scale', c_double, 1, 1), # double scale
    ('shift', c_double, 1, 0), # double shift
)

cvCvtScale = cvConvertScale

cvScale = cvConvertScale

def cvConvert(src, dst):
    cvConvertScale(src, dst, 1, 0)

# Converts input array elements to 8-bit unsigned integer another with optional linear transformation
cvConvertScaleAbs = cfunc('cvConvertScaleAbs', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('scale', c_double, 1, 1), # double scale
    ('shift', c_double, 1, 0), # double shift
)

cvCvtScaleAbs = cvConvertScaleAbs

# Computes per-element sum of two arrays
cvAdd = cfunc('cvAdd', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Computes sum of array and scalar
cvAddS = cfunc('cvAddS', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('value', CvScalar, 1), # CvScalar value
    ('dst', c_void_p, 1), # CvArr* dst
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Computes weighted sum of two arrays
cvAddWeighted = cfunc('cvAddWeighted', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('alpha', c_double, 1), # double alpha
    ('src2', c_void_p, 1), # const CvArr* src2
    ('beta', c_double, 1), # double beta
    ('gamma', c_double, 1), # double gamma
    ('dst', c_void_p, 1), # CvArr* dst
)

# Computes per-element difference between two arrays
cvSub = cfunc('cvSub', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Computes difference between scalar and array
cvSubRS = cfunc('cvSubRS', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('value', CvScalar, 1), # CvScalar value
    ('dst', c_void_p, 1), # CvArr* dst
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Calculates per-element product of two arrays
cvMul = cfunc('cvMul', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
    ('scale', c_double, 1, 1), # double scale
)

# Performs per-element division of two arrays
cvDiv = cfunc('cvDiv', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
    ('scale', c_double, 1, 1), # double scale
)

# Calculates per-element bit-wise conjunction of two arrays
cvAnd = cfunc('cvAnd', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Calculates per-element bit-wise conjunction of array and scalar
cvAndS = cfunc('cvAndS', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('value', CvScalar, 1), # CvScalar value
    ('dst', c_void_p, 1), # CvArr* dst
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Calculates per-element bit-wise disjunction of two arrays
cvOr = cfunc('cvOr', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Calculates per-element bit-wise disjunction of array and scalar
cvOrS = cfunc('cvOrS', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('value', CvScalar, 1), # CvScalar value
    ('dst', c_void_p, 1), # CvArr* dst
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Performs per-element bit-wise "exclusive or" operation on two arrays
cvXor = cfunc('cvXor', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Performs per-element bit-wise "exclusive or" operation on array and scalar
cvXorS = cfunc('cvXorS', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('value', CvScalar, 1), # CvScalar value
    ('dst', c_void_p, 1), # CvArr* dst
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Performs per-element bit-wise inversion of array elements
cvNot = cfunc('cvNot', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
)

# Performs per-element comparison of two arrays
cvCmp = cfunc('cvCmp', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
    ('cmp_op', c_int, 1), # int cmp_op
)

# Performs per-element comparison of array and scalar
cvCmpS = cfunc('cvCmpS', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('value', c_double, 1), # double value
    ('dst', c_void_p, 1), # CvArr* dst
    ('cmp_op', c_int, 1), # int cmp_op
)

# Checks that array elements lie between elements of two other arrays
cvInRange = cfunc('cvInRange', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('lower', c_void_p, 1), # const CvArr* lower
    ('upper', c_void_p, 1), # const CvArr* upper
    ('dst', c_void_p, 1), # CvArr* dst
)

# Checks that array elements lie between two scalars
cvInRangeS = cfunc('cvInRangeS', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('lower', CvScalar, 1), # CvScalar lower
    ('upper', CvScalar, 1), # CvScalar upper
    ('dst', c_void_p, 1), # CvArr* dst
)

# Finds per-element maximum of two arrays
cvMax = cfunc('cvMax', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
)

# Finds per-element maximum of array and scalar
cvMaxS = cfunc('cvMaxS', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('value', c_double, 1), # double value
    ('dst', c_void_p, 1), # CvArr* dst
)

# Finds per-element minimum of two arrays
cvMin = cfunc('cvMin', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
)

# Finds per-element minimum of array and scalar
cvMinS = cfunc('cvMinS', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('value', c_double, 1), # double value
    ('dst', c_void_p, 1), # CvArr* dst
)

# Calculates absolute difference between two arrays
cvAbsDiff = cfunc('cvAbsDiff', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
)

# Calculates absolute difference between array and scalar
cvAbsDiffS = cfunc('cvAbsDiffS', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('value', CvScalar, 1), # CvScalar value
)

def cvAbs(src, dst):
    value = CvScalar()
    value.val[0] = 0.0
    value.val[1] = 0.0
    value.val[2] = 0.0
    value.val[3] = 0.0
    cvAbsDiffS(src, dst, value)

# --- 1.6 Statistics ---------------------------------------------------------

# Counts non-zero array elements
cvCountNonZero = cfunc('cvCountNonZero', _cxDLL, c_int,
    ('arr', c_void_p, 1), # const CvArr* arr
)

# Summarizes array elements
cvSum = cfunc('cvSum', _cxDLL, CvScalar,
    ('arr', c_void_p, 1), # const CvArr* arr
)

# Calculates average (mean) of array elements
cvAvg = cfunc('cvAvg', _cxDLL, CvScalar,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Calculates average (mean) of array elements
cvAvgSdv = cfunc('cvAvgSdv', _cxDLL, None,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('mean', POINTER(CvScalar), 1), # CvScalar* mean
    ('std_dev', POINTER(CvScalar), 1), # CvScalar* std_dev
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Finds global minimum and maximum in array or subarray
## cvMinMaxLoc = _cxDLL.cvMinMaxLoc
## cvMinMaxLoc.restype = None # void
## cvMinMaxLoc.argtypes = [
##     c_void_p, # const CvArr* arr
##     c_void_p, # double* min_val
##     c_void_p, # double* max_val
##     c_void_p, # CvPoint* min_loc=NULL
##     c_void_p, # CvPoint* max_loc=NULL
##     c_void_p # const CvArr* mask=NULL
##     ]

cvMinMaxLoc = cfunc('cvMinMaxLoc', _cxDLL, None,
                    ('image', POINTER(IplImage), 1),
                    ('min_val', POINTER(c_double), 2),
                    ('max_val', POINTER(c_double), 2),
                    ('min_loc', POINTER(CvPoint), 2),
                    ('max_loc', POINTER(CvPoint), 2),
                    ('mask', c_void_p, 1, None))


# Calculates absolute array norm, absolute difference norm or relative difference norm
cvNorm = cfunc('cvNorm', _cxDLL, c_double,
    ('arr1', c_void_p, 1), # const CvArr* arr1
    ('arr2', c_void_p, 1, None), # const CvArr* arr2
    ('norm_type', c_int, 1), # int norm_type
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# --- 1.7 Linear Algebra -----------------------------------------------------

# Initializes scaled identity matrix
cvSetIdentity = cfunc('cvSetIdentity', _cxDLL, None,
    ('mat', c_void_p, 1), # CvArr* mat
    ('value', CvScalar, 1), # CvScalar value
)

# Calculates dot product of two arrays in Euclidian metrics
cvDotProduct = cfunc('cvDotProduct', _cxDLL, c_double,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
)

# Calculates cross product of two 3D vectors
cvCrossProduct = cfunc('cvCrossProduct', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
)

# Calculates sum of scaled array and another array
cvScaleAdd = cfunc('cvScaleAdd', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('scale', CvScalar, 1), # CvScalar scale
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
)

# Performs generalized matrix multiplication
cvGEMM = cfunc('cvGEMM', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('alpha', c_double, 1), # double alpha
    ('src3', c_void_p, 1), # const CvArr* src3
    ('beta', c_double, 1), # double beta
    ('dst', c_void_p, 1), # CvArr* dst
    ('tABC', c_int, 1, 0), # int tABC
)

def cvMatMulAdd(src1, src2, src3, dst):
    cvGEMM(src1, src2, 1, src3, 1, dst, 0)

def cvMatMul(src1, src2, dst):
    cvMatMulAdd(src1, src2, 0, dst)

# Performs matrix transform of every array element
cvTransform = cfunc('cvTransform', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('transmat', POINTER(CvMat), 1), # const CvMat* transmat
    ('shiftvec', POINTER(CvMat), 1, None), # const CvMat* shiftvec
)

# Performs perspective matrix transform of vector array
cvPerspectiveTransform = cfunc('cvPerspectiveTransform', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('mat', POINTER(CvMat), 1), # const CvMat* mat
)

# Calculates product of array and transposed array
cvMulTransposed = cfunc('cvMulTransposed', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('order', c_int, 1), # int order
    ('delta', c_void_p, 1, None), # const CvArr* delta
)

# Returns trace of matrix
cvTrace = cfunc('cvTrace', _cxDLL, CvScalar,
    ('mat', c_void_p, 1), # const CvArr* mat
)

# Transposes matrix
cvTranspose = cfunc('cvTranspose', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
)

# Returns determinant of matrix
cvDet = cfunc('cvDet', _cxDLL, c_double,
    ('mat', c_void_p, 1), # const CvArr* mat
)

# Finds inverse or pseudo-inverse of matrix
cvInvert = cfunc('cvInvert', _cxDLL, c_double,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('method', c_int, 1), # int method
)

# Solves linear system or least-squares problem
cvSolve = cfunc('cvSolve', _cxDLL, c_int,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
    ('method', c_int, 1), # int method
)

# Performs singular value decomposition of real floating-point matrix
cvSVD = cfunc('cvSVD', _cxDLL, None,
    ('A', c_void_p, 1), # CvArr* A
    ('W', c_void_p, 1), # CvArr* W
    ('U', c_void_p, 1, None), # CvArr* U
    ('V', c_void_p, 1, None), # CvArr* V
    ('flags', c_int, 1, 0), # int flags
)

# Performs singular value back substitution
cvSVBkSb = cfunc('cvSVBkSb', _cxDLL, None,
    ('W', c_void_p, 1), # const CvArr* W
    ('U', c_void_p, 1), # const CvArr* U
    ('V', c_void_p, 1), # const CvArr* V
    ('B', c_void_p, 1), # const CvArr* B
    ('X', c_void_p, 1), # CvArr* X
    ('flags', c_int, 1), # int flags
)

# Computes eigenvalues and eigenvectors of symmetric matrix
cvEigenVV = cfunc('cvEigenVV', _cxDLL, None,
    ('mat', c_void_p, 1), # CvArr* mat
    ('evects', c_void_p, 1), # CvArr* evects
    ('evals', c_void_p, 1), # CvArr* evals
    ('eps', c_double, 1, 0), # double eps
)

# Calculates covariation matrix of the set of vectors
cvCalcCovarMatrix = cfunc('cvCalcCovarMatrix', _cxDLL, None,
    ('vects', POINTER(c_void_p), 1), # const CvArr** vects
    ('count', c_int, 1), # int count
    ('cov_mat', c_void_p, 1), # CvArr* cov_mat
    ('avg', c_void_p, 1), # CvArr* avg
    ('flags', c_int, 1), # int flags
)

# Calculates Mahalonobis distance between two vectors
cvMahalanobis = cfunc('cvMahalanobis', _cxDLL, c_double,
    ('vec1', c_void_p, 1), # const CvArr* vec1
    ('vec2', c_void_p, 1), # const CvArr* vec2
    ('mat', c_void_p, 1), # CvArr* mat
)

# --- 1.8 Math Functions -----------------------------------------------------

# Round to nearest integer
def cvRound(val):
    return int(val + 0.5)

# Calculates cubic root
cvCbrt = cfunc('cvCbrt', _cxDLL, c_float,
    ('value', c_float, 1), # float value
)

# Calculates angle of 2D vector
cvFastArctan = cfunc('cvFastArctan', _cxDLL, c_float,
    ('y', c_float, 1), # float y
    ('x', c_float, 1), # float x
)

# Calculates magnitude and/or angle of 2d vectors
cvCartToPolar = cfunc('cvCartToPolar', _cxDLL, None,
    ('x', c_void_p, 1), # const CvArr* x
    ('y', c_void_p, 1), # const CvArr* y
    ('magnitude', c_void_p, 1), # CvArr* magnitude
    ('angle', c_void_p, 1, None), # CvArr* angle
    ('angle_in_degrees', c_int, 1, 0), # int angle_in_degrees
)

# Calculates cartesian coordinates of 2d vectors represented in polar form
cvPolarToCart = cfunc('cvPolarToCart', _cxDLL, None,
    ('magnitude', c_void_p, 1), # const CvArr* magnitude
    ('angle', c_void_p, 1), # const CvArr* angle
    ('x', c_void_p, 1), # CvArr* x
    ('y', c_void_p, 1), # CvArr* y
    ('angle_in_degrees', c_int, 1, 0), # int angle_in_degrees
)

# Raises every array element to power
cvPow = cfunc('cvPow', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('power', c_double, 1), # double power
)

# Calculates exponent of every array element
cvExp = cfunc('cvExp', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
)

# Calculates natural logarithm of every array element absolute value
cvLog = cfunc('cvLog', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
)

# Finds real roots of a cubic equation
cvSolveCubic = cfunc('cvSolveCubic', _cxDLL, None,
    ('coeffs', c_void_p, 1), # const CvArr* coeffs
    ('roots', c_void_p, 1), # CvArr* roots
)

# --- 1.9 Random Number Generation -------------------------------------------

# Fills array with random numbers and updates the RNG state
cvRandArr = cfunc('cvRandArr', _cxDLL, None,
    ('rng', c_void_p, 1), # CvRNG* rng
    ('arr', c_void_p, 1), # CvArr* arr
    ('dist_type', c_int, 1), # int dist_type
    ('param1', CvScalar, 1), # CvScalar param1
    ('param2', CvScalar, 1), # CvScalar param2
)

# --- 1.10 Discrete Transforms -----------------------------------------------

# Performs forward or inverse Discrete Fourier transform of 1D or 2D floating-point array
CV_DXT_FORWARD = 0
CV_DXT_INVERSE = 1
CV_DXT_SCALE = 2
CV_DXT_ROWS = 4
CV_DXT_INV_SCALE = CV_DXT_SCALE | CV_DXT_INVERSE
CV_DXT_INVERSE_SCALE = CV_DXT_INV_SCALE

cvDFT = cfunc('cvDFT', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('flags', c_int, 1), # int flags
    ('nonzero_rows', c_int, 1, 0), # int nonzero_rows
)

# Returns optimal DFT size for given vector size
cvGetOptimalDFTSize = cfunc('cvGetOptimalDFTSize', _cxDLL, c_int,
    ('size0', c_int, 1), # int size0
)

# Performs per-element multiplication of two Fourier spectrums
cvMulSpectrums = cfunc('cvMulSpectrums', _cxDLL, None,
    ('src1', c_void_p, 1), # const CvArr* src1
    ('src2', c_void_p, 1), # const CvArr* src2
    ('dst', c_void_p, 1), # CvArr* dst
    ('flags', c_int, 1), # int flags
)

# Performs forward or inverse Discrete Cosine transform of 1D or 2D floating-point array
CV_DXT_FORWARD = 0
CV_DXT_INVERSE = 1
CV_DXT_ROWS = 4

cvDCT = cfunc('cvDCT', _cxDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('flags', c_int, 1), # int flags
)

# --- 2 Dynamic Structures ---------------------------------------------------

# --- 2.1 Memory Storages ----------------------------------------------------

# Creates memory storage
cvCreateMemStorage = cfunc('cvCreateMemStorage', _cxDLL, POINTER(CvMemStorage),
    ('block_size', c_int, 1, 0), # int block_size
)

# Creates child memory storage
cvCreateChildMemStorage = cfunc('cvCreateChildMemStorage', _cxDLL, POINTER(CvMemStorage),
    ('parent', POINTER(CvMemStorage), 1), # CvMemStorage* parent
)

# Releases memory storage
cvReleaseMemStorage = cfunc('cvReleaseMemStorage', _cxDLL, None,
    ('storage', POINTER(POINTER(CvMemStorage)), 1), # CvMemStorage** storage
)

# Clears memory storage
cvClearMemStorage = cfunc('cvClearMemStorage', _cxDLL, None,
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
)

# Allocates memory buffer in the storage
cvMemStorageAlloc = cfunc('cvMemStorageAlloc', _cxDLL, c_void_p,
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('size', c_ulong, 1), # size_t size
)

# Allocates text string in the storage
cvMemStorageAllocString = cfunc('cvMemStorageAllocString', _cxDLL, CvString,
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('ptr', c_char_p, 1), # const char* ptr
    ('len', c_int, 1), # int len
)

# Saves memory storage position
cvSaveMemStoragePos = cfunc('cvSaveMemStoragePos', _cxDLL, None,
    ('storage', POINTER(CvMemStorage), 1), # const CvMemStorage* storage
    ('pos', POINTER(CvMemStoragePos), 1), # CvMemStoragePos* pos
)

# Restores memory storage position
cvRestoreMemStoragePos = cfunc('cvRestoreMemStoragePos', _cxDLL, None,
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('pos', POINTER(CvMemStoragePos), 1), # CvMemStoragePos* pos
)

# --- 2.2 Sequences ----------------------------------------------------------

# Creates sequence
cvCreateSeq = cfunc('cvCreateSeq', _cxDLL, POINTER(CvSeq),
    ('seq_flags', c_int, 1), # int seq_flags
    ('header_size', c_int, 1), # int header_size
    ('elem_size', c_int, 1), # int elem_size
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
)

# Sets up sequence block size
cvSetSeqBlockSize = cfunc('cvSetSeqBlockSize', _cxDLL, None,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('delta_elems', c_int, 1), # int delta_elems
)

# Adds element to sequence end
cvSeqPush = cfunc('cvSeqPush', _cxDLL, c_void_p,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('element', c_void_p, 1, None), # void* element
)

# Removes element from sequence end
cvSeqPop = cfunc('cvSeqPop', _cxDLL, None,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('element', c_void_p, 1, None), # void* element
)

# Adds element to sequence beginning
cvSeqPushFront = cfunc('cvSeqPushFront', _cxDLL, c_void_p,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('element', c_void_p, 1, None), # void* element
)

# Removes element from sequence beginning
cvSeqPopFront = cfunc('cvSeqPopFront', _cxDLL, None,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('element', c_void_p, 1, None), # void* element
)

# Pushes several elements to the either end of sequence
cvSeqPushMulti = cfunc('cvSeqPushMulti', _cxDLL, None,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('elements', c_void_p, 1), # void* elements
    ('count', c_int, 1), # int count
    ('in_front', c_int, 1, 0), # int in_front
)

# Removes several elements from the either end of sequence
cvSeqPopMulti = cfunc('cvSeqPopMulti', _cxDLL, None,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('elements', c_void_p, 1), # void* elements
    ('count', c_int, 1), # int count
    ('in_front', c_int, 1, 0), # int in_front
)

# Inserts element in sequence middle
cvSeqInsert = cfunc('cvSeqInsert', _cxDLL, c_void_p,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('before_index', c_int, 1), # int before_index
    ('element', c_void_p, 1, None), # void* element
)

# Removes element from sequence middle
cvSeqRemove = cfunc('cvSeqRemove', _cxDLL, None,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('index', c_int, 1), # int index
)

# Clears sequence
cvClearSeq = cfunc('cvClearSeq', _cxDLL, None,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
)

# Returns pointer to sequence element by its index
cvGetSeqElem = cfunc('cvGetSeqElem', _cxDLL, c_void_p,
    ('seq', POINTER(CvSeq), 1), # const CvSeq* seq
    ('index', c_int, 1), # int index
)

def CV_GET_SEQ_ELEM(TYPE, seq, index):
    result = cvGetSeqElem(seq)
    return cast(result, POINTER(TYPE))

# Returns index of concrete sequence element
cvSeqElemIdx = cfunc('cvSeqElemIdx', _cxDLL, c_int,
    ('seq', POINTER(CvSeq), 1), # const CvSeq* seq
    ('element', c_void_p, 1), # const void* element
    ('block', POINTER(POINTER(CvSeqBlock)), 1, None), # CvSeqBlock** block
)

# Copies sequence to one continuous block of memory
cvCvtSeqToArray = cfunc('cvCvtSeqToArray', _cxDLL, c_void_p,
    ('seq', POINTER(CvSeq), 1), # const CvSeq* seq
    ('elements', c_void_p, 1), # void* elements
    ('slice', CvSlice, 1), # CvSlice slice
)

# Constructs sequence from array
cvMakeSeqHeaderForArray = cfunc('cvMakeSeqHeaderForArray', _cxDLL, POINTER(CvSeq),
    ('seq_type', c_int, 1), # int seq_type
    ('header_size', c_int, 1), # int header_size
    ('elem_size', c_int, 1), # int elem_size
    ('elements', c_void_p, 1), # void* elements
    ('total', c_int, 1), # int total
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('block', POINTER(CvSeqBlock), 1), # CvSeqBlock* block
)

# Makes separate header for the sequence slice
cvSeqSlice = cfunc('cvSeqSlice', _cxDLL, POINTER(CvSeq),
    ('seq', POINTER(CvSeq), 1), # const CvSeq* seq
    ('slice', CvSlice, 1), # CvSlice slice
    ('storage', POINTER(CvMemStorage), 1, None), # CvMemStorage* storage
    ('copy_data', c_int, 1, 0), # int copy_data
)

# Removes sequence slice
cvSeqRemoveSlice = cfunc('cvSeqRemoveSlice', _cxDLL, None,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('slice', CvSlice, 1), # CvSlice slice
)

# Inserts array in the middle of sequence
cvSeqInsertSlice = cfunc('cvSeqInsertSlice', _cxDLL, None,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('before_index', c_int, 1), # int before_index
    ('from_arr', c_void_p, 1), # const CvArr* from_arr
)

# Reverses the order of sequence elements
cvSeqInvert = cfunc('cvSeqInvert', _cxDLL, None,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
)

# a < b ? -1 : a > b ? 1 : 0
CvCmpFunc = CFUNCTYPE(c_int, # int
    c_void_p, # const void* a
    c_void_p, # const void* b
    c_void_p) # void* userdata

# Sorts sequence element using the specified comparison function
cvSeqSort = cfunc('cvSeqSort', _cxDLL, None,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('func', CvCmpFunc, 1), # CvCmpFunc func
    ('userdata', c_void_p, 1, None), # void* userdata
)

# Searches element in sequence
cvSeqSearch = cfunc('cvSeqSearch', _cxDLL, c_void_p,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('elem', c_void_p, 1), # const void* elem
    ('func', CvCmpFunc, 1), # CvCmpFunc func
    ('is_sorted', c_int, 1), # int is_sorted
    ('elem_idx', POINTER(c_int), 1), # int* elem_idx
    ('userdata', c_void_p, 1, None), # void* userdata
)

# Initializes process of writing data to sequence
cvStartAppendToSeq = cfunc('cvStartAppendToSeq', _cxDLL, None,
    ('seq', POINTER(CvSeq), 1), # CvSeq* seq
    ('writer', POINTER(CvSeqWriter), 1), # CvSeqWriter* writer
)

# Creates new sequence and initializes writer for it
cvStartWriteSeq = cfunc('cvStartWriteSeq', _cxDLL, None,
    ('seq_flags', c_int, 1), # int seq_flags
    ('header_size', c_int, 1), # int header_size
    ('elem_size', c_int, 1), # int elem_size
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('writer', POINTER(CvSeqWriter), 1), # CvSeqWriter* writer
)

# Finishes process of writing sequence
cvEndWriteSeq = cfunc('cvEndWriteSeq', _cxDLL, POINTER(CvSeq),
    ('writer', POINTER(CvSeqWriter), 1), # CvSeqWriter* writer
)

# Updates sequence headers from the writer state
cvFlushSeqWriter = cfunc('cvFlushSeqWriter', _cxDLL, None,
    ('writer', POINTER(CvSeqWriter), 1), # CvSeqWriter* writer
)

# Initializes process of sequential reading from sequence
cvStartReadSeq = cfunc('cvStartReadSeq', _cxDLL, None,
    ('seq', POINTER(CvSeq), 1), # const CvSeq* seq
    ('reader', POINTER(CvSeqReader), 1), # CvSeqReader* reader
    ('reverse', c_int, 1, 0), # int reverse
)

# Returns the current reader position
cvGetSeqReaderPos = cfunc('cvGetSeqReaderPos', _cxDLL, c_int,
    ('reader', POINTER(CvSeqReader), 1), # CvSeqReader* reader
)

# Moves the reader to specified position
cvSetSeqReaderPos = cfunc('cvSetSeqReaderPos', _cxDLL, None,
    ('reader', POINTER(CvSeqReader), 1), # CvSeqReader* reader
    ('index', c_int, 1), # int index
    ('is_relative', c_int, 1, 0), # int is_relative
)

# --- 2.3 Sets ---------------------------------------------------------------

# Creates empty set
cvCreateSet = cfunc('cvCreateSet', _cxDLL, POINTER(CvSET),
    ('set_flags', c_int, 1), # int set_flags
    ('header_size', c_int, 1), # int header_size
    ('elem_size', c_int, 1), # int elem_size
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
)

# Occupies a node in the set
cvSetAdd = cfunc('cvSetAdd', _cxDLL, c_int,
    ('set_header', POINTER(CvSET), 1), # CvSet* set_header
    ('elem', POINTER(CvSetElem), 1, None), # CvSetElem* elem
    ('inserted_elem', POINTER(POINTER(CvSetElem)), 1, None), # CvSetElem** inserted_elem
)

# Removes element from set
cvSetRemove = cfunc('cvSetRemove', _cxDLL, None,
    ('set_header', POINTER(CvSET), 1), # CvSet* set_header
    ('index', c_int, 1), # int index
)

# Clears set
cvClearSet = cfunc('cvClearSet', _cxDLL, None,
    ('set_header', POINTER(CvSET), 1), # CvSet* set_header
)

# --- 2.4 Graphs -------------------------------------------------------------

# Creates empty graph
cvCreateGraph = cfunc('cvCreateGraph', _cxDLL, POINTER(CvGraph),
    ('graph_flags', c_int, 1), # int graph_flags
    ('header_size', c_int, 1), # int header_size
    ('vtx_size', c_int, 1), # int vtx_size
    ('edge_size', c_int, 1), # int edge_size
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
)

# Adds vertex to graph
cvGraphAddVtx = cfunc('cvGraphAddVtx', _cxDLL, c_int,
    ('graph', POINTER(CvGraph), 1), # CvGraph* graph
    ('vtx', POINTER(CvGraphVtx), 1, None), # const CvGraphVtx* vtx
    ('inserted_vtx', POINTER(POINTER(CvGraphVtx)), 1, None), # CvGraphVtx** inserted_vtx
)

# Removes vertex from graph
cvGraphRemoveVtx = cfunc('cvGraphRemoveVtx', _cxDLL, c_int,
    ('graph', POINTER(CvGraph), 1), # CvGraph* graph
    ('index', c_int, 1), # int index
)

# Removes vertex from graph
cvGraphRemoveVtxByPtr = cfunc('cvGraphRemoveVtxByPtr', _cxDLL, c_int,
    ('graph', POINTER(CvGraph), 1), # CvGraph* graph
    ('vtx', POINTER(CvGraphVtx), 1), # CvGraphVtx* vtx
)

# Adds edge to graph
cvGraphAddEdge = cfunc('cvGraphAddEdge', _cxDLL, c_int,
    ('graph', POINTER(CvGraph), 1), # CvGraph* graph
    ('start_idx', c_int, 1), # int start_idx
    ('end_idx', c_int, 1), # int end_idx
    ('edge', POINTER(CvGraphEdge), 1, None), # const CvGraphEdge* edge
    ('inserted_edge', POINTER(POINTER(CvGraphEdge)), 1, None), # CvGraphEdge** inserted_edge
)

# Adds edge to graph
cvGraphAddEdgeByPtr = cfunc('cvGraphAddEdgeByPtr', _cxDLL, c_int,
    ('graph', POINTER(CvGraph), 1), # CvGraph* graph
    ('start_vtx', POINTER(CvGraphVtx), 1), # CvGraphVtx* start_vtx
    ('end_vtx', POINTER(CvGraphVtx), 1), # CvGraphVtx* end_vtx
    ('edge', POINTER(CvGraphEdge), 1, None), # const CvGraphEdge* edge
    ('inserted_edge', POINTER(POINTER(CvGraphEdge)), 1, None), # CvGraphEdge** inserted_edge
)

# Removes edge from graph
cvGraphRemoveEdge = cfunc('cvGraphRemoveEdge', _cxDLL, None,
    ('graph', POINTER(CvGraph), 1), # CvGraph* graph
    ('start_idx', c_int, 1), # int start_idx
    ('end_idx', c_int, 1), # int end_idx
)

# Removes edge from graph
cvGraphRemoveEdgeByPtr = cfunc('cvGraphRemoveEdgeByPtr', _cxDLL, None,
    ('graph', POINTER(CvGraph), 1), # CvGraph* graph
    ('start_vtx', POINTER(CvGraphVtx), 1), # CvGraphVtx* start_vtx
    ('end_vtx', POINTER(CvGraphVtx), 1), # CvGraphVtx* end_vtx
)

# Finds edge in graph
cvFindGraphEdge = cfunc('cvFindGraphEdge', _cxDLL, POINTER(CvGraphEdge),
    ('graph', POINTER(CvGraph), 1), # const CvGraph* graph
    ('start_idx', c_int, 1), # int start_idx
    ('end_idx', c_int, 1), # int end_idx
)

# Finds edge in graph
cvFindGraphEdgeByPtr = cfunc('cvFindGraphEdgeByPtr', _cxDLL, POINTER(CvGraphEdge),
    ('graph', POINTER(CvGraph), 1), # const CvGraph* graph
    ('start_vtx', POINTER(CvGraphVtx), 1), # const CvGraphVtx* start_vtx
    ('end_vtx', POINTER(CvGraphVtx), 1), # const CvGraphVtx* end_vtx
)

# Counts edges indicent to the vertex
cvGraphVtxDegree = cfunc('cvGraphVtxDegree', _cxDLL, c_int,
    ('graph', POINTER(CvGraph), 1), # const CvGraph* graph
    ('vtx_idx', c_int, 1), # int vtx_idx
)

# Finds edge in graph
cvGraphVtxDegreeByPtr = cfunc('cvGraphVtxDegreeByPtr', _cxDLL, c_int,
    ('graph', POINTER(CvGraph), 1), # const CvGraph* graph
    ('vtx', POINTER(CvGraphVtx), 1), # const CvGraphVtx* vtx
)

# Clears graph
cvClearGraph = cfunc('cvClearGraph', _cxDLL, None,
    ('graph', POINTER(CvGraph), 1), # CvGraph* graph
)

# Clone graph
cvCloneGraph = cfunc('cvCloneGraph', _cxDLL, POINTER(CvGraph),
    ('graph', POINTER(CvGraph), 1), # const CvGraph* graph
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
)

# Creates structure for depth-first graph traversal
cvCreateGraphScanner = cfunc('cvCreateGraphScanner', _cxDLL, POINTER(CvGraphScanner),
    ('graph', POINTER(CvGraph), 1), # CvGraph* graph
    ('vtx', POINTER(CvGraphVtx), 1, None), # CvGraphVtx* vtx
    ('mask', c_int, 1), # int mask
)

# Makes one or more steps of the graph traversal procedure
cvNextGraphItem = cfunc('cvNextGraphItem', _cxDLL, c_int,
    ('scanner', POINTER(CvGraphScanner), 1), # CvGraphScanner* scanner
)

# Finishes graph traversal procedure
cvReleaseGraphScanner = cfunc('cvReleaseGraphScanner', _cxDLL, None,
    ('scanner', POINTER(POINTER(CvGraphScanner)), 1), # CvGraphScanner** scanner
)

# --- 2.5 Trees --------------------------------------------------------------

# Initializes tree node iterator
cvInitTreeNodeIterator = cfunc('cvInitTreeNodeIterator', _cxDLL, None,
    ('tree_iterator', POINTER(CvTreeNodeIterator), 1), # CvTreeNodeIterator* tree_iterator
    ('first', c_void_p, 1), # const void* first
    ('max_level', c_int, 1), # int max_level
)

# Returns the currently observed node and moves iterator toward the next node
cvNextTreeNode = cfunc('cvNextTreeNode', _cxDLL, c_void_p,
    ('tree_iterator', POINTER(CvTreeNodeIterator), 1), # CvTreeNodeIterator* tree_iterator
)

# Returns the currently observed node and moves iterator toward the previous node
cvPrevTreeNode = cfunc('cvPrevTreeNode', _cxDLL, c_void_p,
    ('tree_iterator', POINTER(CvTreeNodeIterator), 1), # CvTreeNodeIterator* tree_iterator
)

# Gathers all node pointers to the single sequence
cvTreeToNodeSeq = cfunc('cvTreeToNodeSeq', _cxDLL, POINTER(CvSeq),
    ('first', c_void_p, 1), # const void* first
    ('header_size', c_int, 1), # int header_size
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
)

# Adds new node to the tree
cvInsertNodeIntoTree = cfunc('cvInsertNodeIntoTree', _cxDLL, None,
    ('node', c_void_p, 1), # void* node
    ('parent', c_void_p, 1), # void* parent
    ('frame', c_void_p, 1), # void* frame
)

# Removes node from tree
cvRemoveNodeFromTree = cfunc('cvRemoveNodeFromTree', _cxDLL, None,
    ('node', c_void_p, 1), # void* node
    ('frame', c_void_p, 1), # void* frame
)

# --- 3 Drawing Functions ----------------------------------------------------

# --- 3.1 Curves and Shapes --------------------------------------------------

# Constructs a color value
def CV_RGB(r, g, b):
    result = CvScalar()
    result.val[0] = b
    result.val[1] = g
    result.val[2] = r
    return result

# Draws a line segment connecting two points
cvLine = cfunc('cvLine', _cxDLL, None,
    ('img', c_void_p, 1), # CvArr* img
    ('pt1', CvPoint, 1), # CvPoint pt1
    ('pt2', CvPoint, 1), # CvPoint pt2
    ('color', CvScalar, 1), # CvScalar color
    ('thickness', c_int, 1, 1), # int thickness
    ('line_type', c_int, 1, 8), # int line_type
    ('shift', c_int, 1, 0), # int shift
)

# Draws simple, thick or filled rectangle
cvRectangle = cfunc('cvRectangle', _cxDLL, None,
    ('img', c_void_p, 1), # CvArr* img
    ('pt1', CvPoint, 1), # CvPoint pt1
    ('pt2', CvPoint, 1), # CvPoint pt2
    ('color', CvScalar, 1), # CvScalar color
    ('thickness', c_int, 1, 1), # int thickness
    ('line_type', c_int, 1, 8), # int line_type
    ('shift', c_int, 1, 0), # int shift
)

# Draws a circle
cvCircle = cfunc('cvCircle', _cxDLL, None,
    ('img', c_void_p, 1), # CvArr* img
    ('center', CvPoint, 1), # CvPoint center
    ('radius', c_int, 1), # int radius
    ('color', CvScalar, 1), # CvScalar color
    ('thickness', c_int, 1, 1), # int thickness
    ('line_type', c_int, 1, 8), # int line_type
    ('shift', c_int, 1, 0), # int shift
)

# Draws simple or thick elliptic arc or fills ellipse sector
cvEllipse = cfunc('cvEllipse', _cxDLL, None,
    ('img', c_void_p, 1), # CvArr* img
    ('center', CvPoint, 1), # CvPoint center
    ('axes', CvSize, 1), # CvSize axes
    ('angle', c_double, 1), # double angle
    ('start_angle', c_double, 1), # double start_angle
    ('end_angle', c_double, 1), # double end_angle
    ('color', CvScalar, 1), # CvScalar color
    ('thickness', c_int, 1, 1), # int thickness
    ('line_type', c_int, 1, 8), # int line_type
    ('shift', c_int, 1, 0), # int shift
)

def cvEllipseBox(img, box, color, thickness=1, line_type=8, shift=0):
    '''Draws simple or thick elliptic arc or fills ellipse sector'''
    cvEllipse(img, CvPoint(int(box.center.x), int(box.center.y)),
              CvSize(int(box.size.height*0.5),int(box.size.width*0.5)),
              box.angle, 0, 360, color, thickness, line_type, shift)


# Fills polygons interior
cvFillPoly = cfunc('cvFillPoly', _cxDLL, None,
    ('img', c_void_p, 1), # CvArr* img
    ('pts', POINTER(POINTER(CvPoint)), 1), # CvPoint** pts
    ('npts', POINTER(c_int), 1), # int* npts
    ('contours', c_int, 1), # int contours
    ('color', CvScalar, 1), # CvScalar color
    ('line_type', c_int, 1, 8), # int line_type
    ('shift', c_int, 1, 0), # int shift
)

# Fills convex polygon
cvFillConvexPoly = cfunc('cvFillConvexPoly', _cxDLL, None,
    ('img', c_void_p, 1), # CvArr* img
    ('pts', POINTER(CvPoint), 1), # CvPoint* pts
    ('npts', c_int, 1), # int npts
    ('color', CvScalar, 1), # CvScalar color
    ('line_type', c_int, 1, 8), # int line_type
    ('shift', c_int, 1, 0), # int shift
)

# Draws simple or thick polygons
cvPolyLine = cfunc('cvPolyLine', _cxDLL, None,
    ('img', c_void_p, 1), # CvArr* img
    ('pts', POINTER(POINTER(CvPoint)), 1), # CvPoint** pts
    ('npts', POINTER(c_int), 1), # int* npts
    ('contours', c_int, 1), # int contours
    ('is_closed', c_int, 1), # int is_closed
    ('color', CvScalar, 1), # CvScalar color
    ('thickness', c_int, 1, 1), # int thickness
    ('line_type', c_int, 1, 8), # int line_type
    ('shift', c_int, 1, 0), # int shift
)

# --- 3.2 Text ---------------------------------------------------------------

# Initializes font structure
cvInitFont = cfunc('cvInitFont', _cxDLL, None,
    ('font', POINTER(CvFont), 1), # CvFont* font
    ('font_face', c_int, 1), # int font_face
    ('hscale', c_double, 1), # double hscale
    ('vscale', c_double, 1), # double vscale
    ('shear', c_double, 1, 0), # double shear
    ('thickness', c_int, 1, 1), # int thickness
    ('line_type', c_int, 1, 8), # int line_type
)

# Draws text string
cvPutText = cfunc('cvPutText', _cxDLL, None,
    ('img', c_void_p, 1), # CvArr* img
    ('text', c_char_p, 1), # const char* text
    ('org', CvPoint, 1), # CvPoint org
    ('font', POINTER(CvFont), 1), # const CvFont* font
    ('color', CvScalar, 1), # CvScalar color
)

# Retrieves width and height of text string
cvGetTextSize = cfunc('cvGetTextSize', _cxDLL, None,
    ('text_string', c_char_p, 1), # const char* text_string
    ('font', POINTER(CvFont), 1), # const CvFont* font
    ('text_size', POINTER(CvSize), 1), # CvSize* text_size
    ('baseline', POINTER(c_int), 1), # int* baseline
)

# --- 3.3 Point Sets and Contours --------------------------------------------

# Draws contour outlines or interiors in the image
cvDrawContours = cfunc('cvDrawContours', _cxDLL, None,
    ('img', c_void_p, 1), # CvArr* img
    ('contour', c_void_p, 1), # CvSeq* contour
    ('external_color', CvScalar, 1), # CvScalar external_color
    ('hole_color', CvScalar, 1), # CvScalar hole_color
    ('max_level', c_int, 1), # int max_level
    ('thickness', c_int, 1, 1), # int thickness
    ('line_type', c_int, 1, 8), # int line_type
    ('offset', CvPoint, 1), # CvPoint offset
)

# Initializes line iterator
cvInitLineIterator = cfunc('cvInitLineIterator', _cxDLL, c_int,
    ('image', c_void_p, 1), # const CvArr* image
    ('pt1', CvPoint, 1), # CvPoint pt1
    ('pt2', CvPoint, 1), # CvPoint pt2
    ('line_iterator', POINTER(CvLineIterator), 1), # CvLineIterator* line_iterator
    ('connectivity', c_int, 1, 8), # int connectivity
    ('left_to_right', c_int, 1, 0), # int left_to_right
)

# Clips the line against the image rectangle
cvClipLine = cfunc('cvClipLine', _cxDLL, c_int,
    ('img_size', CvSize, 1), # CvSize img_size
    ('pt1', POINTER(CvPoint), 1), # CvPoint* pt1
    ('pt2', POINTER(CvPoint), 1), # CvPoint* pt2
)

# Approximates elliptic arc with polyline
cvEllipse2Poly = cfunc('cvEllipse2Poly', _cxDLL, c_int,
    ('center', CvPoint, 1), # CvPoint center
    ('axes', CvSize, 1), # CvSize axes
    ('angle', c_int, 1), # int angle
    ('arc_start', c_int, 1), # int arc_start
    ('arc_end', c_int, 1), # int arc_end
    ('pts', POINTER(CvPoint), 1), # CvPoint* pts
    ('delta', c_int, 1), # int delta
)

# --- 4 Data Persistence and RTTI --------------------------------------------

# --- 4.1 File Storage -------------------------------------------------------

# Opens file storage for reading or writing data
cvOpenFileStorage = cfunc('cvOpenFileStorage', _cxDLL, POINTER(CvFileStorage),
    ('filename', c_char_p, 1), # const char* filename
    ('memstorage', POINTER(CvMemStorage), 1), # CvMemStorage* memstorage
    ('flags', c_int, 1), # int flags
)

# Releases file storage
cvReleaseFileStorage = cfunc('cvReleaseFileStorage', _cxDLL, None,
    ('fs', POINTER(POINTER(CvFileStorage)), 1), # CvFileStorage** fs
)

# --- 4.2 Writing Data -------------------------------------------------------

# Starts writing a new structure
cvStartWriteStruct = cfunc('cvStartWriteStruct', _cxDLL, None,
    ('fs', POINTER(CvFileStorage), 1), # CvFileStorage* fs
    ('name', c_char_p, 1), # const char* name
    ('struct_flags', c_int, 1), # int struct_flags
    ('type_name', c_char_p, 1, None), # const char* type_name
    ('attributes', CvAttrList, 1), # CvAttrList attributes
)

# Ends writing a structure
cvEndWriteStruct = cfunc('cvEndWriteStruct', _cxDLL, None,
    ('fs', POINTER(CvFileStorage), 1), # CvFileStorage* fs
)

# Writes an integer value
cvWriteInt = cfunc('cvWriteInt', _cxDLL, None,
    ('fs', POINTER(CvFileStorage), 1), # CvFileStorage* fs
    ('name', c_char_p, 1), # const char* name
    ('value', c_int, 1), # int value
)

# Writes a floating-point value
cvWriteReal = cfunc('cvWriteReal', _cxDLL, None,
    ('fs', POINTER(CvFileStorage), 1), # CvFileStorage* fs
    ('name', c_char_p, 1), # const char* name
    ('value', c_double, 1), # double value
)

# Writes a text string
cvWriteString = cfunc('cvWriteString', _cxDLL, None,
    ('fs', POINTER(CvFileStorage), 1), # CvFileStorage* fs
    ('name', c_char_p, 1), # const char* name
    ('str', c_char_p, 1), # const char* str
    ('quote', c_int, 1, 0), # int quote
)

# Writes comment
cvWriteComment = cfunc('cvWriteComment', _cxDLL, None,
    ('fs', POINTER(CvFileStorage), 1), # CvFileStorage* fs
    ('comment', c_char_p, 1), # const char* comment
    ('eol_comment', c_int, 1), # int eol_comment
)

# Starts the next stream
cvStartNextStream = cfunc('cvStartNextStream', _cxDLL, None,
    ('fs', POINTER(CvFileStorage), 1), # CvFileStorage* fs
)

# Writes user object
cvWrite = cfunc('cvWrite', _cxDLL, None,
    ('fs', POINTER(CvFileStorage), 1), # CvFileStorage* fs
    ('name', c_char_p, 1), # const char* name
    ('ptr', c_void_p, 1), # const void* ptr
    ('attributes', CvAttrList, 1), # CvAttrList attributes
)

# Writes multiple numbers
cvWriteRawData = cfunc('cvWriteRawData', _cxDLL, None,
    ('fs', POINTER(CvFileStorage), 1), # CvFileStorage* fs
    ('src', c_void_p, 1), # const void* src
    ('len', c_int, 1), # int len
    ('dt', c_char_p, 1), # const char* dt
)

# Writes file node to another file storage
cvWriteFileNode = cfunc('cvWriteFileNode', _cxDLL, None,
    ('fs', POINTER(CvFileStorage), 1), # CvFileStorage* fs
    ('new_node_name', c_char_p, 1), # const char* new_node_name
    ('node', POINTER(CvFileNode), 1), # const CvFileNode* node
    ('embed', c_int, 1), # int embed
)

# --- 4.3 Reading Data -------------------------------------------------------

# Retrieves one of top-level nodes of the file storage
cvGetRootFileNode = cfunc('cvGetRootFileNode', _cxDLL, POINTER(CvFileNode),
    ('fs', POINTER(CvFileStorage), 1), # const CvFileStorage* fs
    ('stream_index', c_int, 1, 0), # int stream_index
)

# Finds node in the map or file storage
cvGetFileNodeByName = cfunc('cvGetFileNodeByName', _cxDLL, POINTER(CvFileNode),
    ('fs', POINTER(CvFileStorage), 1), # const CvFileStorage* fs
    ('map', POINTER(CvFileNode), 1), # const CvFileNode* map
    ('name', c_char_p, 1), # const char* name
)

# Returns a unique pointer for given name
cvGetHashedKey = cfunc('cvGetHashedKey', _cxDLL, POINTER(CvStringHashNode),
    ('fs', POINTER(CvFileStorage), 1), # CvFileStorage* fs
    ('name', c_char_p, 1), # const char* name
    ('len', c_int, 1), # int len
    ('create_missing', c_int, 1, 0), # int create_missing
)

# Finds node in the map or file storage
cvGetFileNode = cfunc('cvGetFileNode', _cxDLL, POINTER(CvFileNode),
    ('fs', POINTER(CvFileStorage), 1), # CvFileStorage* fs
    ('map', POINTER(CvFileNode), 1), # CvFileNode* map
    ('key', POINTER(CvStringHashNode), 1), # const CvStringHashNode* key
    ('create_missing', c_int, 1, 0), # int create_missing
)

# Returns name of file node
cvGetFileNodeName = cfunc('cvGetFileNodeName', _cxDLL, c_char_p,
    ('node', POINTER(CvFileNode), 1), # const CvFileNode* node
)

# Decodes object and returns pointer to it
cvRead = cfunc('cvRead', _cxDLL, c_void_p,
    ('fs', POINTER(CvFileStorage), 1), # CvFileStorage* fs
    ('node', POINTER(CvFileNode), 1), # CvFileNode* node
    ('attributes', POINTER(CvAttrList), 1, None), # CvAttrList* attributes
)

# Reads multiple numbers
cvReadRawData = cfunc('cvReadRawData', _cxDLL, None,
    ('fs', POINTER(CvFileStorage), 1), # const CvFileStorage* fs
    ('src', POINTER(CvFileNode), 1), # const CvFileNode* src
    ('dst', c_void_p, 1), # void* dst
    ('dt', c_char_p, 1), # const char* dt
)

# Initializes file node sequence reader
cvStartReadRawData = cfunc('cvStartReadRawData', _cxDLL, None,
    ('fs', POINTER(CvFileStorage), 1), # const CvFileStorage* fs
    ('src', POINTER(CvFileNode), 1), # const CvFileNode* src
    ('reader', POINTER(CvSeqReader), 1), # CvSeqReader* reader
)

# Initializes file node sequence reader
cvReadRawDataSlice = cfunc('cvReadRawDataSlice', _cxDLL, None,
    ('fs', POINTER(CvFileStorage), 1), # const CvFileStorage* fs
    ('reader', POINTER(CvSeqReader), 1), # CvSeqReader* reader
    ('count', c_int, 1), # int count
    ('dst', c_void_p, 1), # void* dst
    ('dt', c_char_p, 1), # const char* dt
)

# --- 4.4 RTTI and Generic Functions -----------------------------------------

# Registers new type
cvRegisterType = cfunc('cvRegisterType', _cxDLL, None,
    ('info', POINTER(CvTypeInfo), 1), # const CvTypeInfo* info
)

# Unregisters the type
cvUnregisterType = cfunc('cvUnregisterType', _cxDLL, None,
    ('type_name', c_char_p, 1), # const char* type_name
)

# Returns the beginning of type list
cvFirstType = cfunc('cvFirstType', _cxDLL, POINTER(CvTypeInfo),
)

# Finds type by its name
cvFindType = cfunc('cvFindType', _cxDLL, POINTER(CvTypeInfo),
    ('type_name', c_char_p, 1), # const char* type_name
)

# Returns type of the object
cvTypeOf = cfunc('cvTypeOf', _cxDLL, POINTER(CvTypeInfo),
    ('struct_ptr', c_void_p, 1), # const void* struct_ptr
)

# Releases the object
cvRelease = cfunc('cvRelease', _cxDLL, None,
    ('struct_ptr', POINTER(c_void_p), 1), # void** struct_ptr
)

# Makes a clone of the object
cvClone = cfunc('cvClone', _cxDLL, c_void_p,
    ('struct_ptr', c_void_p, 1), # const void* struct_ptr
)

# Saves object to file
cvSave = cfunc('cvSave', _cxDLL, None,
    ('filename', c_char_p, 1), # const char* filename
    ('struct_ptr', c_void_p, 1), # const void* struct_ptr
    ('name', c_char_p, 1, None), # const char* name
    ('comment', c_char_p, 1, None), # const char* comment
    ('attributes', CvAttrList, 1), # CvAttrList attributes
)

# Loads object from file
cvLoad = cfunc('cvLoad', _cxDLL, c_void_p,
    ('filename', c_char_p, 1), # const char* filename
    ('memstorage', POINTER(CvMemStorage), 1, None), # CvMemStorage* memstorage
    ('name', c_char_p, 1, None), # const char* name
    ('real_name', POINTER(c_char_p), 1, None), # const char** real_name
)
# Load and cast to given type
def cvLoadCast(filename, ctype):
    '''Use cvLoad and then cast the result to ctype'''
    return ctypes.cast(cvLoad(filename), ctypes.POINTER(ctype))


# --- 5 Miscellaneous Functions ----------------------------------------------

# Checks every element of input array for invalid values
cvCheckArr = cfunc('cvCheckArr', _cxDLL, c_int,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('flags', c_int, 1, 0), # int flags
    ('min_val', c_double, 1, 0), # double min_val
    ('max_val', c_double, 1, 0), # double max_val
)

cvCheckArray = cvCheckArr

# Splits set of vectors by given number of clusters
cvKMeans2 = cfunc('cvKMeans2', _cxDLL, None,
    ('samples', c_void_p, 1), # const CvArr* samples
    ('cluster_count', c_int, 1), # int cluster_count
    ('labels', c_void_p, 1), # CvArr* labels
    ('termcrit', CvTermCriteria, 1), # CvTermCriteria termcrit
)

# Splits sequence into equivalency classes
cvSeqPartition = cfunc('cvSeqPartition', _cxDLL, c_int,
    ('seq', POINTER(CvSeq), 1), # const CvSeq* seq
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('labels', POINTER(POINTER(CvSeq)), 1), # CvSeq** labels
    ('is_equal', CvCmpFunc, 1), # CvCmpFunc is_equal
    ('userdata', c_void_p, 1), # void* userdata
)

# --- 6 Error Handling and System Functions ----------------------------------

# --- 6.1 Error Handling -----------------------------------------------------

# Returns the current error status
cvGetErrStatus = cfunc('cvGetErrStatus', _cxDLL, c_int,
)

# Sets the error status
cvSetErrStatus = cfunc('cvSetErrStatus', _cxDLL, None,
    ('status', c_int, 1), # int status
)

# Returns the current error mode
cvGetErrMode = cfunc('cvGetErrMode', _cxDLL, c_int,
)

# Sets the error mode
CV_ErrModeLeaf = 0
CV_ErrModeParent = 1
CV_ErrModeSilent = 2

cvSetErrMode = cfunc('cvSetErrMode', _cxDLL, c_int,
    ('mode', c_int, 1), # int mode
)

# Raises an error
cvError = cfunc('cvError', _cxDLL, c_int,
    ('status', c_int, 1), # int status
    ('func_name', c_char_p, 1), # const char* func_name
    ('err_msg', c_char_p, 1), # const char* err_msg
    ('file_name', c_char_p, 1), # const char* file_name
    ('line', c_int, 1), # int line
)

# Returns textual description of error status code
cvErrorStr = cfunc('cvErrorStr', _cxDLL, c_char_p,
    ('status', c_int, 1), # int status
)

# Sets a new error handler
CvErrorCallback = CFUNCTYPE(c_int, # int
    c_int, # int status
    c_char_p, # const char* func_name
    c_char_p, # const char* err_msg
    c_char_p, # const char* file_name
    c_int) # int line

cvRedirectError = cfunc('cvRedirectError', _cxDLL, CvErrorCallback,
    ('error_handler', CvErrorCallback, 1), # CvErrorCallback error_handler
    ('userdata', c_void_p, 1, None), # void* userdata
    ('prev_userdata', POINTER(c_void_p), 1, None), # void** prev_userdata
)

# Provide standard error handling
cvNulDevReport = cfunc('cvNulDevReport', _cxDLL, c_int,
    ('status', c_int, 1), # int status
    ('func_name', c_char_p, 1), # const char* func_name
    ('err_msg', c_char_p, 1), # const char* err_msg
    ('file_name', c_char_p, 1), # const char* file_name
    ('line', c_int, 1), # int line
    ('userdata', c_void_p, 1), # void* userdata
)

cvStdErrReport = cfunc('cvStdErrReport', _cxDLL, c_int,
    ('status', c_int, 1), # int status
    ('func_name', c_char_p, 1), # const char* func_name
    ('err_msg', c_char_p, 1), # const char* err_msg
    ('file_name', c_char_p, 1), # const char* file_name
    ('line', c_int, 1), # int line
    ('userdata', c_void_p, 1), # void* userdata
)

cvGuiBoxReport = cfunc('cvGuiBoxReport', _cxDLL, c_int,
    ('status', c_int, 1), # int status
    ('func_name', c_char_p, 1), # const char* func_name
    ('err_msg', c_char_p, 1), # const char* err_msg
    ('file_name', c_char_p, 1), # const char* file_name
    ('line', c_int, 1), # int line
    ('userdata', c_void_p, 1), # void* userdata
)

# --- 6.2 System and Utility Functions ---------------------------------------

# Allocates memory buffer
cvAlloc = cfunc('cvAlloc', _cxDLL, c_void_p,
    ('size', c_ulong, 1), # size_t size
)

# Deallocates memory buffer
#cvFree = _cxDLL.cvFree
#cvFree.restype = None # void
#cvFree.argtypes = [
#    c_void_p # void** ptr
#    ]

# Returns number of tics
cvGetTickCount = cfunc('cvGetTickCount', _cxDLL, c_longlong,
)

# Returns number of tics per microsecond
cvGetTickFrequency = cfunc('cvGetTickFrequency', _cxDLL, c_double,
)

# Registers another module
cvRegisterModule = cfunc('cvRegisterModule', _cxDLL, c_int,
    ('module_info', POINTER(CvModuleInfo), 1), # const CvModuleInfo* module_info
)

# Retrieves information about the registered module(s) and plugins
cvGetModuleInfo = cfunc('cvGetModuleInfo', _cxDLL, None,
    ('module_name', c_char_p, 1), # const char* module_name
    ('version', POINTER(c_char_p), 1), # const char** version
    ('loaded_addon_plugins', POINTER(c_char_p), 1), # const char** loaded_addon_plugins
)

# Switches between optimized/non-optimized modes
cvUseOptimized = cfunc('cvUseOptimized', _cxDLL, c_int,
    ('on_off', c_int, 1), # int on_off
)

# Assings custom/default memory managing functions
#CvAllocFunc = CFUNCTYPE(c_void_p, # void*
#    c_ulong, # size_t size
#    c_void_p) # void* userdata
#
#CvFreeFunc = CFUNCTYPE(c_int, # int
#    c_void_p, # void* pptr
#    c_void_p) # void* userdata

#cvSetMemoryManager = _cxDLL.cvSetMemoryManager
#cvSetMemoryManager.restype = None # void
#cvSetMemoryManager.argtypes = [
#    CvAllocFunc, # CvAllocFunc alloc_func=NULL
#    CvFreeFunc, # CvFreeFunc free_func=NULL
#    c_void_p # void* userdata=NULL
#    ]

# --- 1 Image Processing -----------------------------------------------------

# --- 1.1 Gradients, Edges and Corners ---------------------------------------

# Calculates first, second, third or mixed image derivatives using extended Sobel operator
cvSobel = cfunc('cvSobel', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('xorder', c_int, 1), # int xorder
    ('yorder', c_int, 1), # int yorder
    ('aperture_size', c_int, 1, 3), # int aperture_size
)

# Calculates Laplacian of the image
cvLaplace = cfunc('cvLaplace', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('aperture_size', c_int, 1, 3), # int aperture_size
)

# Implements Canny algorithm for edge detection
cvCanny = cfunc('cvCanny', _cvDLL, None,
    ('image', c_void_p, 1), # const CvArr* image
    ('edges', c_void_p, 1), # CvArr* edges
    ('threshold1', c_double, 1), # double threshold1
    ('threshold2', c_double, 1), # double threshold2
    ('aperture_size', c_int, 1, 3), # int aperture_size
)

# Calculates feature map for corner detection
cvPreCornerDetect = cfunc('cvPreCornerDetect', _cvDLL, None,
    ('image', c_void_p, 1), # const CvArr* image
    ('corners', c_void_p, 1), # CvArr* corners
    ('aperture_size', c_int, 1, 3), # int aperture_size
)

# Calculates eigenvalues and eigenvectors of image blocks for corner detection
cvCornerEigenValsAndVecs = cfunc('cvCornerEigenValsAndVecs', _cvDLL, None,
    ('image', c_void_p, 1), # const CvArr* image
    ('eigenvv', c_void_p, 1), # CvArr* eigenvv
    ('block_size', c_int, 1), # int block_size
    ('aperture_size', c_int, 1, 3), # int aperture_size
)

# Calculates minimal eigenvalue of gradient matrices for corner detection
cvCornerMinEigenVal = cfunc('cvCornerMinEigenVal', _cvDLL, None,
    ('image', c_void_p, 1), # const CvArr* image
    ('eigenval', c_void_p, 1), # CvArr* eigenval
    ('block_size', c_int, 1), # int block_size
    ('aperture_size', c_int, 1, 3), # int aperture_size
)

# Harris edge detector
cvCornerHarris = cfunc('cvCornerHarris', _cvDLL, None,
    ('image', c_void_p, 1), # const CvArr* image
    ('harris_responce', c_void_p, 1), # CvArr* harris_responce
    ('block_size', c_int, 1), # int block_size
    ('aperture_size', c_int, 1, 3), # int aperture_size
    ('k', c_double, 1, 0), # double k
)

# Refines corner locations
cvFindCornerSubPix = cfunc('cvFindCornerSubPix', _cvDLL, None,
    ('image', c_void_p, 1), # const CvArr* image
    ('corners', POINTER(CvPoint2D32f), 1), # CvPoint2D32f* corners
    ('count', c_int, 1), # int count
    ('win', CvSize, 1), # CvSize win
    ('zero_zone', CvSize, 1), # CvSize zero_zone
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria
)

# Determines strong corners on image
cvGoodFeaturesToTrack = cfunc('cvGoodFeaturesToTrack', _cvDLL, None,
    ('image', c_void_p, 1), # const CvArr* image
    ('eig_image', c_void_p, 1), # CvArr* eig_image
    ('temp_image', c_void_p, 1), # CvArr* temp_image
    ('corners', POINTER(CvPoint2D32f), 1), # CvPoint2D32f* corners
    ('corner_count', POINTER(c_int), 1), # int* corner_count
    ('quality_level', c_double, 1), # double quality_level
    ('min_distance', c_double, 1), # double min_distance
    ('mask', c_void_p, 1, None), # const CvArr* mask
    ('block_size', c_int, 1, 3), # int block_size
    ('use_harris', c_int, 1, 0), # int use_harris
    ('k', c_double, 1, 0), # double k
)

# --- 1.2 Sampling, Interpolation and Geometrical Transforms -----------------

# Reads raster line to buffer
cvSampleLine = cfunc('cvSampleLine', _cvDLL, c_int,
    ('image', c_void_p, 1), # const CvArr* image
    ('pt1', CvPoint, 1), # CvPoint pt1
    ('pt2', CvPoint, 1), # CvPoint pt2
    ('buffer', c_void_p, 1), # void* buffer
    ('connectivity', c_int, 1, 8), # int connectivity
)

# Retrieves pixel rectangle from image with sub-pixel accuracy
cvGetRectSubPix = cfunc('cvGetRectSubPix', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('center', CvPoint2D32f, 1), # CvPoint2D32f center
)

# Retrieves pixel quadrangle from image with sub-pixel accuracy
cvGetQuadrangleSubPix = cfunc('cvGetQuadrangleSubPix', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('map_matrix', POINTER(CvMat), 1), # const CvMat* map_matrix
)

# Resizes image
cvResize = cfunc('cvResize', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('interpolation', c_int, 1), # int interpolation
)

# Applies affine transformation to the image
cvWarpAffine = cfunc('cvWarpAffine', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('map_matrix', POINTER(CvMat), 1), # const CvMat* map_matrix
    ('flags', c_int, 1), # int flags
    ('fillval', CvScalar, 1), # CvScalar fillval
)

# Calculates affine transform from 3 corresponding points
cvGetAffineTransform = cfunc('cvGetAffineTransform', _cvDLL, POINTER(CvMat),
    ('src', POINTER(CvPoint2D32f), 1), # const CvPoint2D32f* src
    ('dst', POINTER(CvPoint2D32f), 1), # const CvPoint2D32f* dst
    ('map_matrix', POINTER(CvMat), 1), # CvMat* map_matrix
)

# Calculates affine matrix of 2d rotation
cv2DRotationMatrix = cfunc('cv2DRotationMatrix', _cvDLL, POINTER(CvMat),
    ('center', CvPoint2D32f, 1), # CvPoint2D32f center
    ('angle', c_double, 1), # double angle
    ('scale', c_double, 1), # double scale
    ('map_matrix', POINTER(CvMat), 1), # CvMat* map_matrix
)

# Applies perspective transformation to the image
cvWarpPerspective = cfunc('cvWarpPerspective', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('map_matrix', POINTER(CvMat), 1), # const CvMat* map_matrix
    ('flags', c_int, 1), # int flags
    ('fillval', CvScalar, 1), # CvScalar fillval
)

# Calculates perspective transform from 4 corresponding points
cvGetPerspectiveTransform = cfunc('cvGetPerspectiveTransform', _cvDLL, POINTER(CvMat),
    ('src', POINTER(CvPoint2D32f), 1), # const CvPoint2D32f* src
    ('dst', POINTER(CvPoint2D32f), 1), # const CvPoint2D32f* dst
    ('map_matrix', POINTER(CvMat), 1), # CvMat* map_matrix
)

# Applies generic geometrical transformation to the image
cvRemap = cfunc('cvRemap', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('mapx', c_void_p, 1), # const CvArr* mapx
    ('mapy', c_void_p, 1), # const CvArr* mapy
    ('flags', c_int, 1), # int flags
    ('fillval', CvScalar, 1), # CvScalar fillval
)

# Remaps image to log-polar space
cvLogPolar = cfunc('cvLogPolar', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('center', CvPoint2D32f, 1), # CvPoint2D32f center
    ('M', c_double, 1), # double M
    ('flags', c_int, 1), # int flags
)

# --- 1.3 Morphological Operations -------------------------------------------

# Creates structuring element
cvCreateStructuringElementEx = cfunc('cvCreateStructuringElementEx', _cvDLL, c_void_p,
    ('cols', c_int, 1), # int cols
    ('rows', c_int, 1), # int rows
    ('anchor_x', c_int, 1), # int anchor_x
    ('anchor_y', c_int, 1), # int anchor_y
    ('shape', c_int, 1), # int shape
    ('values', POINTER(c_int), 1, None), # int* values
)

# Deletes structuring element
cvReleaseStructuringElement = cfunc('cvReleaseStructuringElement', _cvDLL, None,
    ('element', POINTER(POINTER(IplConvKernel)), 1), # IplConvKernel** element
)

# Erodes image by using arbitrary structuring element
cvErode = cfunc('cvErode', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('element', POINTER(IplConvKernel), 1, None), # IplConvKernel* element
    ('iterations', c_int, 1, 1), # int iterations
)

# Dilates image by using arbitrary structuring element
cvDilate = cfunc('cvDilate', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('element', POINTER(IplConvKernel), 1, None), # IplConvKernel* element
    ('iterations', c_int, 1, 1), # int iterations
)

# Performs advanced morphological transformations
cvMorphologyEx = cfunc('cvMorphologyEx', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('temp', c_void_p, 1), # CvArr* temp
    ('element', POINTER(IplConvKernel), 1), # IplConvKernel* element
    ('operation', c_int, 1), # int operation
    ('iterations', c_int, 1, 1), # int iterations
)

# --- 1.4 Filters and Color Conversion ---------------------------------------

# Smooths the image in one of several ways
cvSmooth = cfunc('cvSmooth', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('smoothtype', c_int, 1), # int smoothtype
    ('param1', c_int, 1, 3), # int param1
    ('param2', c_int, 1, 0), # int param2
    ('param3', c_double, 1, 0), # double param3
)

# Convolves image with the kernel
cvFilter2D = cfunc('cvFilter2D', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('kernel', POINTER(CvMat), 1), # const CvMat* kernel
    ('anchor', CvPoint, 1), # CvPoint anchor
)

# Copies image and makes border around it
cvCopyMakeBorder = cfunc('cvCopyMakeBorder', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('offset', CvPoint, 1), # CvPoint offset
    ('bordertype', c_int, 1), # int bordertype
    ('value', CvScalar, 1), # CvScalar value
)

# Calculates integral images
cvIntegral = cfunc('cvIntegral', _cvDLL, None,
    ('image', c_void_p, 1), # const CvArr* image
    ('sum', c_void_p, 1), # CvArr* sum
    ('sqsum', c_void_p, 1, None), # CvArr* sqsum
    ('tilted_sum', c_void_p, 1, None), # CvArr* tilted_sum
)


CV_BGR2BGRA =   0
CV_RGB2RGBA =   CV_BGR2BGRA

CV_BGRA2BGR =   1
CV_RGBA2RGB =   CV_BGRA2BGR

CV_BGR2RGBA =   2
CV_RGB2BGRA =   CV_BGR2RGBA

CV_RGBA2BGR =   3
CV_BGRA2RGB =   CV_RGBA2BGR

CV_BGR2RGB  =   4
CV_RGB2BGR  =   CV_BGR2RGB

CV_BGRA2RGBA =  5
CV_RGBA2BGRA =  CV_BGRA2RGBA

CV_BGR2GRAY =   6
CV_RGB2GRAY =   7
CV_GRAY2BGR =   8
CV_GRAY2RGB =   CV_GRAY2BGR
CV_GRAY2BGRA =  9
CV_GRAY2RGBA =  CV_GRAY2BGRA
CV_BGRA2GRAY =  10
CV_RGBA2GRAY =  11

CV_BGR2BGR565 = 12
CV_RGB2BGR565 = 13
CV_BGR5652BGR = 14
CV_BGR5652RGB = 15
CV_BGRA2BGR565 = 16
CV_RGBA2BGR565 = 17
CV_BGR5652BGRA = 18
CV_BGR5652RGBA = 19

CV_GRAY2BGR565 = 20
CV_BGR5652GRAY = 21

CV_BGR2BGR555  = 22
CV_RGB2BGR555  = 23
CV_BGR5552BGR  = 24
CV_BGR5552RGB  = 25
CV_BGRA2BGR555 = 26
CV_RGBA2BGR555 = 27
CV_BGR5552BGRA = 28
CV_BGR5552RGBA = 29

CV_GRAY2BGR555 = 30
CV_BGR5552GRAY = 31

CV_BGR2XYZ =    32
CV_RGB2XYZ =    33
CV_XYZ2BGR =    34
CV_XYZ2RGB =    35

CV_BGR2YCrCb =  36
CV_RGB2YCrCb =  37
CV_YCrCb2BGR =  38
CV_YCrCb2RGB =  39

CV_BGR2HSV =    40
CV_RGB2HSV =    41

CV_BGR2Lab =    44
CV_RGB2Lab =    45

CV_BayerBG2BGR = 46
CV_BayerGB2BGR = 47
CV_BayerRG2BGR = 48
CV_BayerGR2BGR = 49

CV_BayerBG2RGB = CV_BayerRG2BGR
CV_BayerGB2RGB = CV_BayerGR2BGR
CV_BayerRG2RGB = CV_BayerBG2BGR
CV_BayerGR2RGB = CV_BayerGB2BGR

CV_BGR2Luv =    50
CV_RGB2Luv =    51
CV_BGR2HLS =    52
CV_RGB2HLS =    53

CV_HSV2BGR =    54
CV_HSV2RGB =    55

CV_Lab2BGR =    56
CV_Lab2RGB =    57
CV_Luv2BGR =    58
CV_Luv2RGB =    59
CV_HLS2BGR =    60
CV_HLS2RGB =    61


# Converts image from one color space to another
cvCvtColor = cfunc('cvCvtColor', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('code', c_int, 1), # int code
)

# Applies fixed-level threshold to array elements
cvThreshold = cfunc('cvThreshold', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('threshold', c_double, 1), # double threshold
    ('max_value', c_double, 1), # double max_value
    ('threshold_type', c_int, 1), # int threshold_type
)

# Applies adaptive threshold to array
cvAdaptiveThreshold = cfunc('cvAdaptiveThreshold', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('max_value', c_double, 1), # double max_value
    ('adaptive_method', c_int, 1), # int adaptive_method
    ('threshold_type', c_int, 1), # int threshold_type
    ('block_size', c_int, 1, 3), # int block_size
    ('param1', c_double, 1, 5), # double param1
)

# --- 1.5 Pyramids and the Applications --------------------------------------

# Downsamples image
cvPyrDown = cfunc('cvPyrDown', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('filter', c_int, 1), # int filter
)

# Upsamples image
cvPyrUp = cfunc('cvPyrUp', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('filter', c_int, 1), # int filter
)

# Implements image segmentation by pyramids
cvPyrSegmentation = cfunc('cvPyrSegmentation', _cvDLL, None,
    ('src', POINTER(IplImage), 1), # IplImage* src
    ('dst', POINTER(IplImage), 1), # IplImage* dst
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('comp', POINTER(POINTER(CvSeq)), 1), # CvSeq** comp
    ('level', c_int, 1), # int level
    ('threshold1', c_double, 1), # double threshold1
    ('threshold2', c_double, 1), # double threshold2
)

# --- 1.6 Connected Components -----------------------------------------------


# Fills a connected component with given color
cvFloodFill = cfunc('cvFloodFill', _cvDLL, None,
    ('image', c_void_p, 1), # CvArr* image
    ('seed_point', CvPoint, 1), # CvPoint seed_point
    ('new_val', CvScalar, 1), # CvScalar new_val
    ('lo_diff', CvScalar, 1), # CvScalar lo_diff
    ('up_diff', CvScalar, 1), # CvScalar up_diff
    ('comp', POINTER(CvConnectedComp), 1, None), # CvConnectedComp* comp
    ('flags', c_int, 1, 4), # int flags
    ('mask', c_void_p, 1, None), # CvArr* mask
)

CV_FLOODFILL_FIXED_RANGE = 1 << 16
CV_FLOODFILL_MASK_ONLY = 1 << 17

# Finds contours in binary image
cvFindContours = cfunc('cvFindContours', _cvDLL, c_int,
    ('image', c_void_p, 1), # CvArr* image
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('first_contour', POINTER(POINTER(CvSeq)), 1), # CvSeq** first_contour
    ('header_size', c_int, 1), # int header_size
    ('mode', c_int, 1), # int mode
    ('method', c_int, 1), # int method
    ('offset', CvPoint, 1), # CvPoint offset
)

# Initializes contour scanning process
cvStartFindContours = cfunc('cvStartFindContours', _cvDLL, CvContourScanner,
    ('image', c_void_p, 1), # CvArr* image
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('header_size', c_int, 1), # int header_size
    ('mode', c_int, 1), # int mode
    ('method', c_int, 1), # int method
    ('offset', CvPoint, 1), # CvPoint offset
)

# Finds next contour in the image
cvFindNextContour = cfunc('cvFindNextContour', _cvDLL, POINTER(CvSeq),
    ('scanner', CvContourScanner, 1), # CvContourScanner scanner
)

# Replaces retrieved contour
cvSubstituteContour = cfunc('cvSubstituteContour', _cvDLL, None,
    ('scanner', CvContourScanner, 1), # CvContourScanner scanner
    ('new_contour', POINTER(CvSeq), 1), # CvSeq* new_contour
)

# Finishes scanning process
cvEndFindContours = cfunc('cvEndFindContours', _cvDLL, POINTER(CvSeq),
    ('scanner', POINTER(CvContourScanner), 1), # CvContourScanner* scanner
)

# --- 1.7 Image and Contour moments ------------------------------------------

# Calculates all moments up to third order of a polygon or rasterized shape
cvMoments = cfunc('cvMoments', _cvDLL, None,
    ('arr', c_void_p, 1), # const CvArr* arr
    ('moments', POINTER(CvMOMENTS), 1), # CvMoments* moments
    ('binary', c_int, 1, 0), # int binary
)

# Retrieves spatial moment from moment state structure
cvGetSpatialMoment = cfunc('cvGetSpatialMoment', _cvDLL, c_double,
    ('moments', POINTER(CvMOMENTS), 1), # CvMoments* moments
    ('x_order', c_int, 1), # int x_order
    ('y_order', c_int, 1), # int y_order
)

# Retrieves central moment from moment state structure
cvGetCentralMoment = cfunc('cvGetCentralMoment', _cvDLL, c_double,
    ('moments', POINTER(CvMOMENTS), 1), # CvMoments* moments
    ('x_order', c_int, 1), # int x_order
    ('y_order', c_int, 1), # int y_order
)

# Retrieves normalized central moment from moment state structure
cvGetNormalizedCentralMoment = cfunc('cvGetNormalizedCentralMoment', _cvDLL, c_double,
    ('moments', POINTER(CvMOMENTS), 1), # CvMoments* moments
    ('x_order', c_int, 1), # int x_order
    ('y_order', c_int, 1), # int y_order
)

# Calculates seven Hu invariants
cvGetHuMoments = cfunc('cvGetHuMoments', _cvDLL, None,
    ('moments', POINTER(CvMOMENTS), 1), # CvMoments* moments
    ('hu_moments', POINTER(CvHuMoments), 1), # CvHuMoments* hu_moments
)

# --- 1.8 Special Image Transforms -------------------------------------------

# Finds lines in binary image using Hough transform
cvHoughLines2 = cfunc('cvHoughLines2', _cvDLL, POINTER(CvSeq),
    ('image', c_void_p, 1), # CvArr* image
    ('line_storage', c_void_p, 1), # void* line_storage
    ('method', c_int, 1), # int method
    ('rho', c_double, 1), # double rho
    ('theta', c_double, 1), # double theta
    ('threshold', c_int, 1), # int threshold
    ('param1', c_double, 1, 0), # double param1
    ('param2', c_double, 1, 0), # double param2
)

# Finds circles in grayscale image using Hough transform
cvHoughCircles = cfunc('cvHoughCircles', _cvDLL, POINTER(CvSeq),
    ('image', c_void_p, 1), # CvArr* image
    ('circle_storage', c_void_p, 1), # void* circle_storage
    ('method', c_int, 1), # int method
    ('dp', c_double, 1), # double dp
    ('min_dist', c_double, 1), # double min_dist
    ('param1', c_double, 1, 100), # double param1
    ('param2', c_double, 1, 100), # double param2
)

# Calculates distance to closest zero pixel for all non-zero pixels of source image
cvDistTransform = cfunc('cvDistTransform', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('distance_type', c_int, 1), # int distance_type
    ('mask_size', c_int, 1, 3), # int mask_size
    ('mask', POINTER(c_float), 1, None), # const float* mask
    ('labels', c_void_p, 1, None), # CvArr* labels
)

# --- 1.9 Histograms ---------------------------------------------------------
CV_HIST_ARRAY = 0
CV_HIST_SPARSE = 1

# Creates histogram
cvCreateHist = cfunc('cvCreateHist', _cvDLL, POINTER(CvHistogram),
    ('dims', c_int, 1), # int dims
    ('sizes', ListPOINTER(c_int), 1), # int* sizes
    ('type', c_int, 1), # int type
    ('ranges', ListPOINTER2(c_float), 1, None), # float** ranges=NULL
    ('uniform', c_int, 1, 1), # int uniform=1
)

# Sets bounds of histogram bins
cvSetHistBinRanges = cfunc('cvSetHistBinRanges', _cvDLL, None,
    ('hist', POINTER(CvHistogram), 1), # CvHistogram* hist
    ('ranges', ListPOINTER2(c_float), 1), # float** ranges
    ('uniform', c_int, 1, 1), # int uniform
)

# Releases histogram
cvReleaseHist = cfunc('cvReleaseHist', _cvDLL, None,
    ('hist', ByRefArg(POINTER(CvHistogram)), 1), # CvHistogram** hist
)

# Clears histogram
cvClearHist = cfunc('cvClearHist', _cvDLL, None,
    ('hist', POINTER(CvHistogram), 1), # CvHistogram* hist
)

# Makes a histogram out of array
cvMakeHistHeaderForArray = cfunc('cvMakeHistHeaderForArray', _cvDLL, POINTER(CvHistogram),
    ('dims', c_int, 1), # int dims
    ('sizes', POINTER(c_int), 1), # int* sizes
    ('hist', POINTER(CvHistogram), 1), # CvHistogram* hist
    ('data', POINTER(c_float), 1), # float* data
    ('ranges', ListPOINTER2(c_float), 1, None), # float** ranges
    ('uniform', c_int, 1, 1), # int uniform
)

# Finds minimum and maximum histogram bins
cvGetMinMaxHistValue = cfunc('cvGetMinMaxHistValue', _cvDLL, None,
    ('hist', POINTER(CvHistogram), 1), # const CvHistogram* hist
    ('min_value', POINTER(c_float), 2), # float* min_value
    ('max_value', POINTER(c_float), 2), # float* max_value
    ('min_idx', POINTER(c_int), 2), # int* min_idx
    ('max_idx', POINTER(c_int), 2), # int* max_idx
)

# Normalizes histogram
cvNormalizeHist = cfunc('cvNormalizeHist', _cvDLL, None,
    ('hist', POINTER(CvHistogram), 1), # CvHistogram* hist
    ('factor', c_double, 1), # double factor
)

# Thresholds histogram
cvThreshHist = cfunc('cvThreshHist', _cvDLL, None,
    ('hist', POINTER(CvHistogram), 1), # CvHistogram* hist
    ('threshold', c_double, 1), # double threshold
)

CV_COMP_CORREL       = 0
CV_COMP_CHISQR       = 1
CV_COMP_INTERSECT    = 2
CV_COMP_BHATTACHARYYA= 3

# Compares two dense histograms
cvCompareHist = cfunc('cvCompareHist', _cvDLL, c_double,
    ('hist1', POINTER(CvHistogram), 1), # const CvHistogram* hist1
    ('hist2', POINTER(CvHistogram), 1), # const CvHistogram* hist2
    ('method', c_int, 1), # int method
)

# Copies histogram
cvCopyHist = cfunc('cvCopyHist', _cvDLL, None,
    ('src', POINTER(CvHistogram), 1), # const CvHistogram* src
    ('dst', POINTER(POINTER(CvHistogram)), 1), # CvHistogram** dst
)

# Calculate the histogram
cvCalcHist = cfunc('cvCalcArrHist', _cvDLL, None,
    ('image', ListPOINTER(POINTER(IplImage)), 1), # IplImage** image
    ('hist', POINTER(CvHistogram), 1), # CvHistogram* hist
    ('accumulate', c_int, 1, 0), # int accumulate
    ('mask', c_void_p, 1, None), # CvArr* mask
)

# Calculates back projection
cvCalcBackProject = cfunc('cvCalcArrBackProject', _cvDLL, None,
    ('image', ListPOINTER(POINTER(IplImage)), 1), # IplImage** image
    ('back_project', POINTER(IplImage), 1), # IplImage* back_project
    ('hist', POINTER(CvHistogram), 1), # CvHistogram* hist
 )

# Divides one histogram by another
cvCalcProbDensity = cfunc('cvCalcProbDensity', _cvDLL, None,
    ('hist1', POINTER(CvHistogram), 1), # const CvHistogram* hist1
    ('hist2', POINTER(CvHistogram), 1), # const CvHistogram* hist2
    ('dst_hist', POINTER(CvHistogram), 1), # CvHistogram* dst_hist
    ('scale', c_double, 1, 255), # double scale
)

def QueryHistValue_1D(hist, i1, i2):
    return cvGetReal1D(hist[0].bins, i1)

def QueryHistValue_2D(hist, i1, i2):
    return cvGetReal2D(hist[0].bins, i1, i2)

def QueryHistValue_3D(hist, i1, i2, i3):
    return cvGetReal2D(hist[0].bins, i1, i2, i3)

# Equalizes histogram of grayscale image
cvEqualizeHist = cfunc('cvEqualizeHist', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
)

# --- 1.10 Matching ----------------------------------------------------------

# Compares template against overlapped image regions
cvMatchTemplate = cfunc('cvMatchTemplate', _cvDLL, None,
    ('image', c_void_p, 1), # const CvArr* image
    ('templ', c_void_p, 1), # const CvArr* templ
    ('result', c_void_p, 1), # CvArr* result
    ('method', c_int, 1), # int method
)

# Compares two shapes
cvMatchShapes = cfunc('cvMatchShapes', _cvDLL, c_double,
    ('object1', c_void_p, 1), # const void* object1
    ('object2', c_void_p, 1), # const void* object2
    ('method', c_int, 1), # int method
    ('parameter', c_double, 1, 0), # double parameter
)

# Computes "minimal work" distance between two weighted point configurations
CvDistanceFunction = CFUNCTYPE(c_float, # float
    c_void_p, # const float* f1
    c_void_p, # const float* f2
    c_void_p) # void* userdata

cvCalcEMD2 = cfunc('cvCalcEMD2', _cvDLL, c_float,
    ('signature1', c_void_p, 1), # const CvArr* signature1
    ('signature2', c_void_p, 1), # const CvArr* signature2
    ('distance_type', c_int, 1), # int distance_type
    ('distance_func', CvDistanceFunction, 1, None), # CvDistanceFunction distance_func
    ('cost_matrix', c_void_p, 1, None), # const CvArr* cost_matrix
    ('flow', c_void_p, 1, None), # CvArr* flow
    ('lower_bound', POINTER(c_float), 1, None), # float* lower_bound
    ('userdata', c_void_p, 1, None), # void* userdata
)

# --- 2 Structural Analysis --------------------------------------------------

# --- 2.1 Contour Processing Functions ---------------------------------------

# Approximates Freeman chain(s) with polygonal curve
cvApproxChains = cfunc('cvApproxChains', _cvDLL, POINTER(CvSeq),
    ('src_seq', POINTER(CvSeq), 1), # CvSeq* src_seq
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('method', c_int, 1), # int method
    ('parameter', c_double, 1, 0), # double parameter
    ('minimal_perimeter', c_int, 1, 0), # int minimal_perimeter
    ('recursive', c_int, 1, 0), # int recursive
)

# Initializes chain reader
cvStartReadChainPoints = cfunc('cvStartReadChainPoints', _cvDLL, None,
    ('chain', POINTER(CvChain), 1), # CvChain* chain
    ('reader', POINTER(CvChainPtReader), 1), # CvChainPtReader* reader
)

# Gets next chain point
cvReadChainPoint = cfunc('cvReadChainPoint', _cvDLL, CvPoint,
    ('reader', POINTER(CvChainPtReader), 1), # CvChainPtReader* reader
)

# Approximates polygonal curve(s) with desired precision
cvApproxPoly = cfunc('cvApproxPoly', _cvDLL, POINTER(CvSeq),
    ('src_seq', c_void_p, 1), # const void* src_seq
    ('header_size', c_int, 1), # int header_size
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('method', c_int, 1), # int method
    ('parameter', c_double, 1), # double parameter
    ('parameter2', c_int, 1, 0), # int parameter2
)

# Calculates up-right bounding rectangle of point set
cvBoundingRect = cfunc('cvBoundingRect', _cvDLL, CvRect,
    ('points', c_void_p, 1), # CvArr* points
    ('update', c_int, 1, 0), # int update
)

# Calculates area of the whole contour or contour section
cvContourArea = cfunc('cvContourArea', _cvDLL, c_double,
    ('contour', c_void_p, 1), # const CvArr* contour
    ('slice', CvSlice, 1), # CvSlice slice
)

# Calculates contour perimeter or curve length
cvArcLength = cfunc('cvArcLength', _cvDLL, c_double,
    ('curve', c_void_p, 1), # const void* curve
    ('slice', CvSlice, 1), # CvSlice slice
    ('is_closed', c_int, 1), # int is_closed
)

# Creates hierarchical representation of contour
cvCreateContourTree = cfunc('cvCreateContourTree', _cvDLL, POINTER(CvContourTree),
    ('contour', POINTER(CvSeq), 1), # const CvSeq* contour
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('threshold', c_double, 1), # double threshold
)

# Restores contour from tree
cvContourFromContourTree = cfunc('cvContourFromContourTree', _cvDLL, POINTER(CvSeq),
    ('tree', POINTER(CvContourTree), 1), # const CvContourTree* tree
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria
)

# Compares two contours using their tree representations
cvMatchContourTrees = cfunc('cvMatchContourTrees', _cvDLL, c_double,
    ('tree1', POINTER(CvContourTree), 1), # const CvContourTree* tree1
    ('tree2', POINTER(CvContourTree), 1), # const CvContourTree* tree2
    ('method', c_int, 1), # int method
    ('threshold', c_double, 1), # double threshold
)

# --- 2.2 Computational Geometry ---------------------------------------------

# Finds bounding rectangle for two given rectangles
cvMaxRect = cfunc('cvMaxRect', _cvDLL, CvRect,
    ('rect1', POINTER(CvRect), 1), # const CvRect* rect1
    ('rect2', POINTER(CvRect), 1), # const CvRect* rect2
)

# Initializes point sequence header from a point vector
cvPointSeqFromMat = cfunc('cvPointSeqFromMat', _cvDLL, POINTER(CvSeq),
    ('seq_kind', c_int, 1), # int seq_kind
    ('mat', c_void_p, 1), # const CvArr* mat
    ('contour_header', POINTER(CvContour), 1), # CvContour* contour_header
    ('block', POINTER(CvSeqBlock), 1), # CvSeqBlock* block
)

# Finds box vertices
cvBoxPoints = cfunc('cvBoxPoints', _cvDLL, None,
    ('box', CvBox2D, 1), # CvBox2D box
    ('pt', CvPoint2D32f, 1), # CvPoint2D32f pt
)

# Fits ellipse to set of 2D points
cvFitEllipse2 = cfunc('cvFitEllipse2', _cvDLL, CvBox2D,
    ('points', c_void_p, 1), # const CvArr* points
)

# Fits line to 2D or 3D point set
cvFitLine = cfunc('cvFitLine', _cvDLL, None,
    ('points', c_void_p, 1), # const CvArr* points
    ('dist_type', c_int, 1), # int dist_type
    ('param', c_double, 1), # double param
    ('reps', c_double, 1), # double reps
    ('aeps', c_double, 1), # double aeps
    ('line', POINTER(c_float), 1), # float* line
)

# Finds convex hull of point set
cvConvexHull2 = cfunc('cvConvexHull2', _cvDLL, POINTER(CvSeq),
    ('input', c_void_p, 1), # const CvArr* input
    ('hull_storage', c_void_p, 1, None), # void* hull_storage
    ('orientation', c_int, 1), # int orientation
    ('return_points', c_int, 1, 0), # int return_points
)

# Tests contour convex
cvCheckContourConvexity = cfunc('cvCheckContourConvexity', _cvDLL, c_int,
    ('contour', c_void_p, 1), # const CvArr* contour
)

# Finds convexity defects of contour
cvConvexityDefects = cfunc('cvConvexityDefects', _cvDLL, POINTER(CvSeq),
    ('contour', c_void_p, 1), # const CvArr* contour
    ('convexhull', c_void_p, 1), # const CvArr* convexhull
    ('storage', POINTER(CvMemStorage), 1, None), # CvMemStorage* storage
)

# Point in contour test
cvPointPolygonTest = cfunc('cvPointPolygonTest', _cvDLL, c_double,
    ('contour', c_void_p, 1), # const CvArr* contour
    ('pt', CvPoint2D32f, 1), # CvPoint2D32f pt
    ('measure_dist', c_int, 1), # int measure_dist
)

# Finds circumscribed rectangle of minimal area for given 2D point set
cvMinAreaRect2 = cfunc('cvMinAreaRect2', _cvDLL, CvBox2D,
    ('points', c_void_p, 1), # const CvArr* points
    ('storage', POINTER(CvMemStorage), 1, None), # CvMemStorage* storage
)

# Finds circumscribed circle of minimal area for given 2D point set
cvMinEnclosingCircle = cfunc('cvMinEnclosingCircle', _cvDLL, c_int,
    ('points', c_void_p, 1), # const CvArr* points
    ('center', POINTER(CvPoint2D32f), 1), # CvPoint2D32f* center
    ('radius', POINTER(c_float), 1), # float* radius
)

# Calculates pair-wise geometrical histogram for contour
cvCalcPGH = cfunc('cvCalcPGH', _cvDLL, None,
    ('contour', POINTER(CvSeq), 1), # const CvSeq* contour
    ('hist', POINTER(CvHistogram), 1), # CvHistogram* hist
)

# --- 2.3 Planar Subdivisions ------------------------------------------------

# Inserts a single point to Delaunay triangulation
cvSubdivDelaunay2DInsert = cfunc('cvSubdivDelaunay2DInsert', _cvDLL, POINTER(CvSubdiv2DPoint),
    ('subdiv', POINTER(CvSubdiv2D), 1), # CvSubdiv2D* subdiv
    ('pt', CvPoint2D32f, 1), # CvPoint2D32f pt
)

# Inserts a single point to Delaunay triangulation
cvSubdiv2DLocate = cfunc('cvSubdiv2DLocate', _cvDLL, CvSubdiv2DPointLocation,
    ('subdiv', POINTER(CvSubdiv2D), 1), # CvSubdiv2D* subdiv
    ('pt', CvPoint2D32f, 1), # CvPoint2D32f pt
    ('edge', POINTER(CvSubdiv2DEdge), 1), # CvSubdiv2DEdge* edge
    ('vertex', POINTER(POINTER(CvSubdiv2DPoint)), 1, None), # CvSubdiv2DPoint** vertex
)

# Finds the closest subdivision vertex to given point
cvFindNearestPoint2D = cfunc('cvFindNearestPoint2D', _cvDLL, POINTER(CvSubdiv2DPoint),
    ('subdiv', POINTER(CvSubdiv2D), 1), # CvSubdiv2D* subdiv
    ('pt', CvPoint2D32f, 1), # CvPoint2D32f pt
)

# Calculates coordinates of Voronoi diagram cells
cvCalcSubdivVoronoi2D = cfunc('cvCalcSubdivVoronoi2D', _cvDLL, None,
    ('subdiv', POINTER(CvSubdiv2D), 1), # CvSubdiv2D* subdiv
)

# Removes all virtual points
cvClearSubdivVoronoi2D = cfunc('cvClearSubdivVoronoi2D', _cvDLL, None,
    ('subdiv', POINTER(CvSubdiv2D), 1), # CvSubdiv2D* subdiv
)

# --- 3 Motion Analysis and Object Tracking ----------------------------------

# --- 3.1 Accumulation of Background Statistics ------------------------------

# Adds frame to accumulator
cvAcc = cfunc('cvAcc', _cvDLL, None,
    ('image', c_void_p, 1), # const CvArr* image
    ('sum', c_void_p, 1), # CvArr* sum
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Adds the square of source image to accumulator
cvSquareAcc = cfunc('cvSquareAcc', _cvDLL, None,
    ('image', c_void_p, 1), # const CvArr* image
    ('sqsum', c_void_p, 1), # CvArr* sqsum
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Adds product of two input images to accumulator
cvMultiplyAcc = cfunc('cvMultiplyAcc', _cvDLL, None,
    ('image1', c_void_p, 1), # const CvArr* image1
    ('image2', c_void_p, 1), # const CvArr* image2
    ('acc', c_void_p, 1), # CvArr* acc
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# Updates running average
cvRunningAvg = cfunc('cvRunningAvg', _cvDLL, None,
    ('image', c_void_p, 1), # const CvArr* image
    ('acc', c_void_p, 1), # CvArr* acc
    ('alpha', c_double, 1), # double alpha
    ('mask', c_void_p, 1, None), # const CvArr* mask
)

# --- 3.2 Motion Templates ---------------------------------------------------

# Updates motion history image by moving silhouette
cvUpdateMotionHistory = cfunc('cvUpdateMotionHistory', _cvDLL, None,
    ('silhouette', c_void_p, 1), # const CvArr* silhouette
    ('mhi', c_void_p, 1), # CvArr* mhi
    ('timestamp', c_double, 1), # double timestamp
    ('duration', c_double, 1), # double duration
)

# Calculates gradient orientation of motion history image
cvCalcMotionGradient = cfunc('cvCalcMotionGradient', _cvDLL, None,
    ('mhi', c_void_p, 1), # const CvArr* mhi
    ('mask', c_void_p, 1), # CvArr* mask
    ('orientation', c_void_p, 1), # CvArr* orientation
    ('delta1', c_double, 1), # double delta1
    ('delta2', c_double, 1), # double delta2
    ('aperture_size', c_int, 1, 3), # int aperture_size
)

# Calculates global motion orientation of some selected region
cvCalcGlobalOrientation = cfunc('cvCalcGlobalOrientation', _cvDLL, c_double,
    ('orientation', c_void_p, 1), # const CvArr* orientation
    ('mask', c_void_p, 1), # const CvArr* mask
    ('mhi', c_void_p, 1), # const CvArr* mhi
    ('timestamp', c_double, 1), # double timestamp
    ('duration', c_double, 1), # double duration
)

# Segments whole motion into separate moving parts
cvSegmentMotion = cfunc('cvSegmentMotion', _cvDLL, POINTER(CvSeq),
    ('mhi', c_void_p, 1), # const CvArr* mhi
    ('seg_mask', c_void_p, 1), # CvArr* seg_mask
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('timestamp', c_double, 1), # double timestamp
    ('seg_thresh', c_double, 1), # double seg_thresh
)

# --- 3.3 Object Tracking ----------------------------------------------------

# Finds object center on back projection
cvMeanShift = cfunc('cvMeanShift', _cvDLL, c_int,
    ('prob_image', c_void_p, 1), # const CvArr* prob_image
    ('window', CvRect, 1), # CvRect window
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria
    ('comp', POINTER(CvConnectedComp), 1), # CvConnectedComp* comp
)

# Finds object center, size, and orientation
cvCamShift = cfunc('cvCamShift', _cvDLL, c_int,
    ('prob_image', c_void_p, 1), # const CvArr* prob_image
    ('window', CvRect, 1), # CvRect window
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria
    ('comp', POINTER(CvConnectedComp), 2), # CvConnectedComp* comp
    ('box', POINTER(CvBox2D), 2), # CvBox2D* box
)

# Changes contour position to minimize its energy
cvSnakeImage = cfunc('cvSnakeImage', _cvDLL, None,
    ('image', POINTER(IplImage), 1), # const IplImage* image
    ('points', POINTER(CvPoint), 1), # CvPoint* points
    ('length', c_int, 1), # int length
    ('alpha', POINTER(c_float), 1), # float* alpha
    ('beta', POINTER(c_float), 1), # float* beta
    ('gamma', POINTER(c_float), 1), # float* gamma
    ('coeff_usage', c_int, 1), # int coeff_usage
    ('win', CvSize, 1), # CvSize win
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria
    ('calc_gradient', c_int, 1, 1), # int calc_gradient
)

# --- 3.4 Optical Flow -------------------------------------------------------

# Calculates optical flow for two images
cvCalcOpticalFlowHS = cfunc('cvCalcOpticalFlowHS', _cvDLL, None,
    ('prev', c_void_p, 1), # const CvArr* prev
    ('curr', c_void_p, 1), # const CvArr* curr
    ('use_previous', c_int, 1), # int use_previous
    ('velx', c_void_p, 1), # CvArr* velx
    ('vely', c_void_p, 1), # CvArr* vely
    ('lambda', c_double, 1), # double lambda
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria
)

# Calculates optical flow for two images
cvCalcOpticalFlowLK = cfunc('cvCalcOpticalFlowLK', _cvDLL, None,
    ('prev', c_void_p, 1), # const CvArr* prev
    ('curr', c_void_p, 1), # const CvArr* curr
    ('win_size', CvSize, 1), # CvSize win_size
    ('velx', c_void_p, 1), # CvArr* velx
    ('vely', c_void_p, 1), # CvArr* vely
)

# Calculates optical flow for two images by block matching method
cvCalcOpticalFlowBM = cfunc('cvCalcOpticalFlowBM', _cvDLL, None,
    ('prev', c_void_p, 1), # const CvArr* prev
    ('curr', c_void_p, 1), # const CvArr* curr
    ('block_size', CvSize, 1), # CvSize block_size
    ('shift_size', CvSize, 1), # CvSize shift_size
    ('max_range', CvSize, 1), # CvSize max_range
    ('use_previous', c_int, 1), # int use_previous
    ('velx', c_void_p, 1), # CvArr* velx
    ('vely', c_void_p, 1), # CvArr* vely
)

# Calculates optical flow for a sparse feature set using iterative Lucas-Kanade method in   pyramids
cvCalcOpticalFlowPyrLK = cfunc('cvCalcOpticalFlowPyrLK', _cvDLL, None,
    ('prev', c_void_p, 1), # const CvArr* prev
    ('curr', c_void_p, 1), # const CvArr* curr
    ('prev_pyr', c_void_p, 1), # CvArr* prev_pyr
    ('curr_pyr', c_void_p, 1), # CvArr* curr_pyr
    ('prev_features', POINTER(CvPoint2D32f), 1), # const CvPoint2D32f* prev_features
    ('curr_features', POINTER(CvPoint2D32f), 1), # CvPoint2D32f* curr_features
    ('count', c_int, 1), # int count
    ('win_size', CvSize, 1), # CvSize win_size
    ('level', c_int, 1), # int level
    ('status', c_char_p, 1), # char* status
    ('track_error', POINTER(c_float), 1), # float* track_error
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria
    ('flags', c_int, 1), # int flags
)

# --- 3.5 Estimators ---------------------------------------------------------

# Allocates Kalman filter structure
cvCreateKalman = cfunc('cvCreateKalman', _cvDLL, POINTER(CvKalman),
    ('dynam_params', c_int, 1), # int dynam_params
    ('measure_params', c_int, 1), # int measure_params
    ('control_params', c_int, 1, 0), # int control_params
)

# Deallocates Kalman filter structure
cvReleaseKalman = cfunc('cvReleaseKalman', _cvDLL, None,
    ('kalman', POINTER(POINTER(CvKalman)), 1), # CvKalman** kalman
)

# Estimates subsequent model state
cvKalmanPredict = cfunc('cvKalmanPredict', _cvDLL, POINTER(CvMat),
    ('kalman', POINTER(CvKalman), 1), # CvKalman* kalman
    ('control', POINTER(CvMat), 1, None), # const CvMat* control
)

cvKalmanUpdateByTime = cvKalmanPredict

# Adjusts model state
cvKalmanCorrect = cfunc('cvKalmanCorrect', _cvDLL, POINTER(CvMat),
    ('kalman', POINTER(CvKalman), 1), # CvKalman* kalman
    ('measurement', POINTER(CvMat), 1), # const CvMat* measurement
)

cvKalmanUpdateByMeasurement = cvKalmanCorrect

# Allocates ConDensation filter structure
cvCreateConDensation = cfunc('cvCreateConDensation', _cvDLL, POINTER(CvConDensation),
    ('dynam_params', c_int, 1), # int dynam_params
    ('measure_params', c_int, 1), # int measure_params
    ('sample_count', c_int, 1), # int sample_count
)

# Deallocates ConDensation filter structure
cvReleaseConDensation = cfunc('cvReleaseConDensation', _cvDLL, None,
    ('condens', POINTER(POINTER(CvConDensation)), 1), # CvConDensation** condens
)

# Initializes sample set for ConDensation algorithm
cvConDensInitSampleSet = cfunc('cvConDensInitSampleSet', _cvDLL, None,
    ('condens', POINTER(CvConDensation), 1), # CvConDensation* condens
    ('lower_bound', POINTER(CvMat), 1), # CvMat* lower_bound
    ('upper_bound', POINTER(CvMat), 1), # CvMat* upper_bound
)

# Estimates subsequent model state
cvConDensUpdateByTime = cfunc('cvConDensUpdateByTime', _cvDLL, None,
    ('condens', POINTER(CvConDensation), 1), # CvConDensation* condens
)

# --- 4 Pattern Recognition --------------------------------------------------

# --- 4.1 Object Detection ---------------------------------------------------

# Loads a trained cascade classifier from file or the classifier database embedded in OpenCV
cvLoadHaarClassifierCascade = cfunc('cvLoadHaarClassifierCascade', _cvDLL, POINTER(CvHaarClassifierCascade),
    ('directory', c_char_p, 1), # const char* directory
    ('orig_window_size', CvSize, 1), # CvSize orig_window_size
)

# Releases haar classifier cascade
cvReleaseHaarClassifierCascade = cfunc('cvReleaseHaarClassifierCascade', _cvDLL, None,
    ('cascade', POINTER(POINTER(CvHaarClassifierCascade)), 1), # CvHaarClassifierCascade** cascade
)

# Detects objects in the image
cvHaarDetectObjects = cfunc('cvHaarDetectObjects', _cvDLL, POINTER(CvSeq),
    ('image', c_void_p, 1), # const CvArr* image
    ('cascade', POINTER(CvHaarClassifierCascade), 1), # CvHaarClassifierCascade* cascade
    ('storage', POINTER(CvMemStorage), 1), # CvMemStorage* storage
    ('scale_factor', c_double, 1, 1), # double scale_factor
    ('min_neighbors', c_int, 1, 3), # int min_neighbors
    ('flags', c_int, 1, 0), # int flags
    ('min_size', CvSize, 1), # CvSize min_size
)
def ChangeCvSeqToCvRect(result, func, args):
    '''Handle the casting to extract a list of Rects from the Seq returned'''
    res = []
    for i in xrange(result[0].total):
        f = cvGetSeqElem(result, i)
        r = ctypes.cast(f, ctypes.POINTER(CvRect))[0]
        res.append(r)
    return res
cvHaarDetectObjects.errcheck = ChangeCvSeqToCvRect


# Assigns images to the hidden cascade
cvSetImagesForHaarClassifierCascade = cfunc('cvSetImagesForHaarClassifierCascade', _cvDLL, None,
    ('cascade', POINTER(CvHaarClassifierCascade), 1), # CvHaarClassifierCascade* cascade
    ('sum', c_void_p, 1), # const CvArr* sum
    ('sqsum', c_void_p, 1), # const CvArr* sqsum
    ('tilted_sum', c_void_p, 1), # const CvArr* tilted_sum
    ('scale', c_double, 1), # double scale
)

# Runs cascade of boosted classifier at given image location
cvRunHaarClassifierCascade = cfunc('cvRunHaarClassifierCascade', _cvDLL, c_int,
    ('cascade', POINTER(CvHaarClassifierCascade), 1), # CvHaarClassifierCascade* cascade
    ('pt', CvPoint, 1), # CvPoint pt
    ('start_stage', c_int, 1, 0), # int start_stage
)

# --- 5 Camera Calibration and 3D Reconstruction -----------------------------

# --- 5.1 Camera Calibration -------------------------------------------------

# Projects 3D points to image plane
cvProjectPoints2 = cfunc('cvProjectPoints2', _cvDLL, None,
    ('object_points', POINTER(CvMat), 1), # const CvMat* object_points
    ('rotation_vector', POINTER(CvMat), 1), # const CvMat* rotation_vector
    ('translation_vector', POINTER(CvMat), 1), # const CvMat* translation_vector
    ('intrinsic_matrix', POINTER(CvMat), 1), # const CvMat* intrinsic_matrix
    ('distortion_coeffs', POINTER(CvMat), 1), # const CvMat* distortion_coeffs
    ('image_points', POINTER(CvMat), 1), # CvMat* image_points
    ('dpdrot', POINTER(CvMat), 1, None), # CvMat* dpdrot
    ('dpdt', POINTER(CvMat), 1, None), # CvMat* dpdt
    ('dpdf', POINTER(CvMat), 1, None), # CvMat* dpdf
    ('dpdc', POINTER(CvMat), 1, None), # CvMat* dpdc
    ('dpddist', POINTER(CvMat), 1, None), # CvMat* dpddist
)

# Finds perspective transformation between two planes
cvFindHomography = cfunc('cvFindHomography', _cvDLL, None,
    ('src_points', POINTER(CvMat), 1), # const CvMat* src_points
    ('dst_points', POINTER(CvMat), 1), # const CvMat* dst_points
    ('homography', POINTER(CvMat), 1), # CvMat* homography
)

# Finds intrinsic and extrinsic camera parameters using calibration pattern
cvCalibrateCamera2 = cfunc('cvCalibrateCamera2', _cvDLL, None,
    ('object_points', POINTER(CvMat), 1), # const CvMat* object_points
    ('image_points', POINTER(CvMat), 1), # const CvMat* image_points
    ('point_counts', POINTER(CvMat), 1), # const CvMat* point_counts
    ('image_size', CvSize, 1), # CvSize image_size
    ('intrinsic_matrix', POINTER(CvMat), 1), # CvMat* intrinsic_matrix
    ('distortion_coeffs', POINTER(CvMat), 1), # CvMat* distortion_coeffs
    ('rotation_vectors', POINTER(CvMat), 1, None), # CvMat* rotation_vectors
    ('translation_vectors', POINTER(CvMat), 1, None), # CvMat* translation_vectors
    ('flags', c_int, 1, 0), # int flags
)

# Finds extrinsic camera parameters for particular view
cvFindExtrinsicCameraParams2 = cfunc('cvFindExtrinsicCameraParams2', _cvDLL, None,
    ('object_points', POINTER(CvMat), 1), # const CvMat* object_points
    ('image_points', POINTER(CvMat), 1), # const CvMat* image_points
    ('intrinsic_matrix', POINTER(CvMat), 1), # const CvMat* intrinsic_matrix
    ('distortion_coeffs', POINTER(CvMat), 1), # const CvMat* distortion_coeffs
    ('rotation_vector', POINTER(CvMat), 1), # CvMat* rotation_vector
    ('translation_vector', POINTER(CvMat), 1), # CvMat* translation_vector
)

# Converts rotation matrix to rotation vector or vice versa
cvRodrigues2 = cfunc('cvRodrigues2', _cvDLL, c_int,
    ('src', POINTER(CvMat), 1), # const CvMat* src
    ('dst', POINTER(CvMat), 1), # CvMat* dst
    ('jacobian', POINTER(CvMat), 1, 0), # CvMat* jacobian
)

# Transforms image to compensate lens distortion
cvUndistort2 = cfunc('cvUndistort2', _cvDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('intrinsic_matrix', POINTER(CvMat), 1), # const CvMat* intrinsic_matrix
    ('distortion_coeffs', POINTER(CvMat), 1), # const CvMat* distortion_coeffs
)

# Computes undistorion map
cvInitUndistortMap = cfunc('cvInitUndistortMap', _cvDLL, None,
    ('intrinsic_matrix', POINTER(CvMat), 1), # const CvMat* intrinsic_matrix
    ('distortion_coeffs', POINTER(CvMat), 1), # const CvMat* distortion_coeffs
    ('mapx', c_void_p, 1), # CvArr* mapx
    ('mapy', c_void_p, 1), # CvArr* mapy
)

# Finds positions of internal corners of the chessboard
cvFindChessboardCorners = cfunc('cvFindChessboardCorners', _cvDLL, c_int,
    ('image', c_void_p, 1), # const void* image
    ('pattern_size', CvSize, 1), # CvSize pattern_size
    ('corners', POINTER(CvPoint2D32f), 1), # CvPoint2D32f* corners
    ('corner_count', POINTER(c_int), 1, None), # int* corner_count
    ('flags', c_int, 1), # int flags
)

# Renders the detected chessboard corners
cvDrawChessboardCorners = cfunc('cvDrawChessboardCorners', _cvDLL, None,
    ('image', c_void_p, 1), # CvArr* image
    ('pattern_size', CvSize, 1), # CvSize pattern_size
    ('corners', POINTER(CvPoint2D32f), 1), # CvPoint2D32f* corners
    ('count', c_int, 1), # int count
    ('pattern_was_found', c_int, 1), # int pattern_was_found
)

# --- 5.2 Pose Estimation ----------------------------------------------------

# Initializes structure containing object information
cvCreatePOSITObject = cfunc('cvCreatePOSITObject', _cvDLL, POINTER(CvPOSITObject),
    ('points', POINTER(CvPoint3D32f), 1), # CvPoint3D32f* points
    ('point_count', c_int, 1), # int point_count
)

# Implements POSIT algorithm
cvPOSIT = cfunc('cvPOSIT', _cvDLL, None,
    ('posit_object', POINTER(CvPOSITObject), 1), # CvPOSITObject* posit_object
    ('image_points', POINTER(CvPoint2D32f), 1), # CvPoint2D32f* image_points
    ('focal_length', c_double, 1), # double focal_length
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria
    ('rotation_matrix', CvMatr32f, 1), # CvMatr32f rotation_matrix
    ('translation_vector', CvVect32f, 1), # CvVect32f translation_vector
)

# Deallocates 3D object structure
cvReleasePOSITObject = cfunc('cvReleasePOSITObject', _cvDLL, None,
    ('posit_object', POINTER(POINTER(CvPOSITObject)), 1), # CvPOSITObject** posit_object
)

# Calculates homography matrix for oblong planar object (e.g. arm)
cvCalcImageHomography = cfunc('cvCalcImageHomography', _cvDLL, None,
    ('line', POINTER(c_float), 1), # float* line
    ('center', POINTER(CvPoint3D32f), 1), # CvPoint3D32f* center
    ('intrinsic', POINTER(c_float), 1), # float* intrinsic
    ('homography', POINTER(c_float), 1), # float* homography
)

# --- 5.3 Epipolar Geometry --------------------------------------------------

# Calculates fundamental matrix from corresponding points in two images
cvFindFundamentalMat = cfunc('cvFindFundamentalMat', _cvDLL, c_int,
    ('points1', POINTER(CvMat), 1), # const CvMat* points1
    ('points2', POINTER(CvMat), 1), # const CvMat* points2
    ('fundamental_matrix', POINTER(CvMat), 1), # CvMat* fundamental_matrix
    ('method', c_int, 1), # int method
    ('param1', c_double, 1, 1), # double param1
    ('param2', c_double, 1, 0), # double param2
    ('status', POINTER(CvMat), 1, None), # CvMat* status
)

# For points in one image of stereo pair computes the corresponding epilines in the other image
cvComputeCorrespondEpilines = cfunc('cvComputeCorrespondEpilines', _cvDLL, None,
    ('points', POINTER(CvMat), 1), # const CvMat* points
    ('which_image', c_int, 1), # int which_image
    ('fundamental_matrix', POINTER(CvMat), 1), # const CvMat* fundamental_matrix
    ('correspondent_lines', POINTER(CvMat), 1), # CvMat* correspondent_lines
)

# --- 1 Simple GUI -----------------------------------------------------------

# Creates window
cvNamedWindow = cfunc('cvNamedWindow', _hgDLL, c_int,
    ('name', c_char_p, 1), # const char* name
    ('flags', c_int, 1, 1), # int flags
)

# Destroys a window
cvDestroyWindow = cfunc('cvDestroyWindow', _hgDLL, None,
    ('name', c_char_p, 1), # const char* name
)

# Destroys all the HighGUI windows
cvDestroyAllWindows = cfunc('cvDestroyAllWindows', _hgDLL, None,
)

# Sets window size
cvResizeWindow = cfunc('cvResizeWindow', _hgDLL, None,
    ('name', c_char_p, 1), # const char* name
    ('width', c_int, 1), # int width
    ('height', c_int, 1), # int height
)

# Sets window position
cvMoveWindow = cfunc('cvMoveWindow', _hgDLL, None,
    ('name', c_char_p, 1), # const char* name
    ('x', c_int, 1), # int x
    ('y', c_int, 1), # int y
)

# Gets window handle by name
cvGetWindowHandle = cfunc('cvGetWindowHandle', _hgDLL, c_void_p,
    ('name', c_char_p, 1), # const char* name
)

# Gets window name by handle
cvGetWindowName = cfunc('cvGetWindowName', _hgDLL, c_void_p,
    ('window_handle', c_void_p, 1), # void* window_handle
)

# Shows the image in the specified window
cvShowImage = cfunc('cvShowImage', _hgDLL, None,
    ('name', c_char_p, 1), # const char* name
    ('image', c_void_p, 1), # const CvArr* image
)

# Creates the trackbar and attaches it to the specified window
CvTrackbarCallback = CFUNCTYPE(None, # void
    c_int) # int pos

cvCreateTrackbar = cfunc('cvCreateTrackbar', _hgDLL, c_int,
    ('trackbar_name', c_char_p, 1), # const char* trackbar_name
    ('window_name', c_char_p, 1), # const char* window_name
    ('value', POINTER(c_int), 1), # int* value
    ('count', c_int, 1), # int count
    ('on_change', CallableToFunc(CvTrackbarCallback), 1), # CvTrackbarCallback on_change
)

# Retrieves trackbar position
cvGetTrackbarPos = cfunc('cvGetTrackbarPos', _hgDLL, c_int,
    ('trackbar_name', c_char_p, 1), # const char* trackbar_name
    ('window_name', c_char_p, 1), # const char* window_name
)

# Sets trackbar position
cvSetTrackbarPos = cfunc('cvSetTrackbarPos', _hgDLL, None,
    ('trackbar_name', c_char_p, 1), # const char* trackbar_name
    ('window_name', c_char_p, 1), # const char* window_name
    ('pos', c_int, 1), # int pos
)

# Assigns callback for mouse events
CV_EVENT_MOUSEMOVE = 0
CV_EVENT_LBUTTONDOWN = 1
CV_EVENT_RBUTTONDOWN = 2
CV_EVENT_MBUTTONDOWN = 3
CV_EVENT_LBUTTONUP = 4
CV_EVENT_RBUTTONUP = 5
CV_EVENT_MBUTTONUP = 6
CV_EVENT_LBUTTONDBLCLK = 7
CV_EVENT_RBUTTONDBLCLK = 8
CV_EVENT_MBUTTONDBLCLK = 9

CV_EVENT_FLAG_LBUTTON = 1
CV_EVENT_FLAG_RBUTTON = 2
CV_EVENT_FLAG_MBUTTON = 4
CV_EVENT_FLAG_CTRLKEY = 8
CV_EVENT_FLAG_SHIFTKEY = 16
CV_EVENT_FLAG_ALTKEY = 32

CvMouseCallback = CFUNCTYPE(None, # void
    c_int, # int event
    c_int, # int x
    c_int, # int y
    c_int, # int flags
    c_void_p) # void* param

cvSetMouseCallback = cfunc('cvSetMouseCallback', _hgDLL, None,
    ('window_name', c_char_p, 1), # const char* window_name
    ('on_mouse', CallableToFunc(CvMouseCallback), 1), # CvMouseCallback on_mouse
    ('param', c_void_p, 1, None), # void* param
)

# Waits for a pressed key
cvWaitKey = cfunc('cvWaitKey', _hgDLL, c_int,
    ('delay', c_int, 1, 0), # int delay
)

# --- 2 Loading and Saving Images --------------------------------------------

# Loads an image from file
cvLoadImage = cfunc('cvLoadImage', _hgDLL, POINTER(IplImage),
    ('filename', c_char_p, 1), # const char* filename
    ('iscolor', c_int, 1, 1), # int iscolor
)

# Saves an image to the file
cvSaveImage = cfunc('cvSaveImage', _hgDLL, c_int,
    ('filename', c_char_p, 1), # const char* filename
    ('image', c_void_p, 1), # const CvArr* image
)

# --- 3 Video I/O functions --------------------------------------------------

# Initializes capturing video from file
cvCreateFileCapture = cfunc('cvCreateFileCapture', _hgDLL, POINTER(CvCapture),
    ('filename', c_char_p, 1), # const char* filename
)

# Initializes capturing video from camera
cvCreateCameraCapture = cfunc('cvCreateCameraCapture', _hgDLL, POINTER(CvCapture),
    ('index', c_int, 1), # int index
)

# Releases the CvCapture structure
cvReleaseCapture = cfunc('cvReleaseCapture', _hgDLL, None,
    ('capture', POINTER(POINTER(CvCapture)), 1), # CvCapture** capture
)

# Grabs frame from camera or file
cvGrabFrame = cfunc('cvGrabFrame', _hgDLL, c_int,
    ('capture', POINTER(CvCapture), 1), # CvCapture* capture
)

# Gets the image grabbed with cvGrabFrame
cvRetrieveFrame = cfunc('cvRetrieveFrame', _hgDLL, POINTER(IplImage),
    ('capture', POINTER(CvCapture), 1), # CvCapture* capture
)

# Grabs and returns a frame from camera or file
cvQueryFrame = cfunc('cvQueryFrame', _hgDLL, POINTER(IplImage),
    ('capture', POINTER(CvCapture), 1), # CvCapture* capture
)
def CheckNonNull(result, func, args):
    if not result:
        raise RuntimeError, 'QueryFrame failed'
    return args

# Gets video capturing properties
# does this really return a double? On Ubuntu, I'm getting an int when requesting
# the frame width.
cvGetCaptureProperty = cfunc('cvGetCaptureProperty', _hgDLL, c_double,
    ('capture', POINTER(CvCapture), 1), # CvCapture* capture
    ('property_id', c_int, 1), # int property_id
)

# Sets video capturing properties
cvSetCaptureProperty = cfunc('cvSetCaptureProperty', _hgDLL, c_int,
    ('capture', POINTER(CvCapture), 1), # CvCapture* capture
    ('property_id', c_int, 1), # int property_id
    ('value', c_double, 1), # double value
)

# Creates video file writer
cvCreateVideoWriter = cfunc('cvCreateVideoWriter', _hgDLL, POINTER(CvVideoWriter),
    ('filename', c_char_p, 1), # const char* filename
    ('fourcc', c_int, 1), # int fourcc
    ('fps', c_double, 1), # double fps
    ('frame_size', CvSize, 1), # CvSize frame_size
    ('is_color', c_int, 1, 1), # int is_color
)

# Releases AVI writer
cvReleaseVideoWriter = cfunc('cvReleaseVideoWriter', _hgDLL, None,
    ('writer', POINTER(POINTER(CvVideoWriter)), 1), # CvVideoWriter** writer
)

# Writes a frame to video file
cvWriteFrame = cfunc('cvWriteFrame', _hgDLL, c_int,
    ('writer', POINTER(CvVideoWriter), 1), # CvVideoWriter* writer
    ('image', POINTER(IplImage), 1), # const IplImage* image
)

# --- 4 Utility and System Functions -----------------------------------------

# Initializes HighGUI
cvInitSystem = cfunc('cvInitSystem', _hgDLL, c_int,
    ('argc', c_int, 1), # int argc
    ('argv', POINTER(c_char_p), 1), # char** argv
)

CV_CVTIMG_SWAP_RB = 2
CV_CVTIMG_FLIP = 1

# Converts one image to another with optional vertical flip
cvConvertImage = cfunc('cvConvertImage', _hgDLL, None,
    ('src', c_void_p, 1), # const CvArr* src
    ('dst', c_void_p, 1), # CvArr* dst
    ('flags', c_int, 1, 0), # int flags
)

# -- Helpers for access to images for other GUI packages

def cvImageAsString(img):
    return ctypes.string_at(img[0].imageData, img[0].imageSize)

def cvImageAsBuffer(img):
    btype = ctypes.c_byte * img[0].imageSize
    return buffer(btype.from_address(img[0].imageData))

def cvImageAsBitmap(img, flip=True):
    import wx
    sz = cvGetSize(img)
    flags = CV_CVTIMG_SWAP_RB
    if flip:
        flags |= CV_CVTIMG_FLIP
    cvConvertImage(img, img, flags)
    bitmap = wx.BitmapFromBuffer(sz.width, sz.height, cvImageAsBuffer(img))
    return bitmap

# --- Dokumentationsstrings --------------------------------------------------

cvCreateImage.__doc__ = """IplImage* cvCreateImage(CvSize size, int depth, int channels)

Creates header and allocates data
"""

cvCreateImageHeader.__doc__ = """IplImage* cvCreateImageHeader(CvSize size, int depth, int channels)

Allocates, initializes, and returns structure IplImage
"""

cvReleaseImageHeader.__doc__ = """void cvReleaseImageHeader(IplImage** image)

Releases header
"""

cvReleaseImage.__doc__ = """void cvReleaseImage(IplImage** image)

Releases header and image data
"""

cvInitImageHeader.__doc__ = """IplImage* cvInitImageHeader(IplImage* image, CvSize size, int depth, int channels, int origin=0, int align=4)

Initializes allocated by user image header
"""

cvCloneImage.__doc__ = """IplImage* cvCloneImage(const IplImage* image)

Makes a full copy of image
"""

cvSetImageCOI.__doc__ = """void cvSetImageCOI(IplImage* image, int coi)

Sets channel of interest to given value
"""

cvGetImageCOI.__doc__ = """int cvGetImageCOI(const IplImage* image)

Returns index of channel of interest
"""

cvSetImageROI.__doc__ = """void cvSetImageROI(IplImage* image, CvRect rect)

Sets image ROI to given rectangle
"""

cvResetImageROI.__doc__ = """void cvResetImageROI(IplImage* image)

Releases image ROI
"""

cvGetImageROI.__doc__ = """CvRect cvGetImageROI(const IplImage* image)

Returns image ROI coordinates
"""

cvCreateMat.__doc__ = """CvMat* cvCreateMat(int rows, int cols, int type)

Creates new matrix
"""

cvCreateMatHeader.__doc__ = """CvMat* cvCreateMatHeader(int rows, int cols, int type)

Creates new matrix header
"""

cvReleaseMat.__doc__ = """void cvReleaseMat(CvMat** mat)

Deallocates matrix
"""

cvInitMatHeader.__doc__ = """CvMat* cvInitMatHeader(CvMat* mat, int rows, int cols, int type, void* data=NULL, int step=CV_AUTOSTEP)

Initializes matrix header
"""

cvCloneMat.__doc__ = """CvMat* cvCloneMat(const CvMat* mat)

Creates matrix copy
"""

cvCreateMatND.__doc__ = """CvMatND* cvCreateMatND(int dims, const int* sizes, int type)

Creates multi-dimensional dense array
"""

cvCreateMatNDHeader.__doc__ = """CvMatND* cvCreateMatNDHeader(int dims, const int* sizes, int type)

Creates new matrix header
"""

cvInitMatNDHeader.__doc__ = """CvMatND* cvInitMatNDHeader(CvMatND* mat, int dims, const int* sizes, int type, void* data=NULL)

Initializes multi-dimensional array header
"""

cvCloneMatND.__doc__ = """CvMatND* cvCloneMatND(const CvMatND* mat)

Creates full copy of multi-dimensional array
"""

cvCreateData.__doc__ = """void cvCreateData(CvArr* arr)

Allocates array data
"""

cvReleaseData.__doc__ = """void cvReleaseData(CvArr* arr)

Releases array data
"""

cvSetData.__doc__ = """void cvSetData(CvArr* arr, void* data, int step)

Assigns user data to the array header
"""

cvGetRawData.__doc__ = """void cvGetRawData(const CvArr* arr, uchar** data, int* step=NULL, CvSize* roi_size=NULL)

Retrieves low-level information about the array
"""

cvGetMat.__doc__ = """CvMat* cvGetMat(const CvArr* arr, CvMat* header, int* coi=NULL, int allowND=0)

Returns matrix header for arbitrary array
"""

cvGetImage.__doc__ = """IplImage* cvGetImage(const CvArr* arr, IplImage* image_header)

Returns image header for arbitrary array
"""

cvCreateSparseMat.__doc__ = """CvSparseMat* cvCreateSparseMat(int dims, const int* sizes, int type)

Creates sparse array
"""

cvReleaseSparseMat.__doc__ = """void cvReleaseSparseMat(CvSparseMat** mat)

Deallocates sparse array
"""

cvCloneSparseMat.__doc__ = """CvSparseMat* cvCloneSparseMat(const CvSparseMat* mat)

Creates full copy of sparse array
"""

cvGetSubRect.__doc__ = """CvMat* cvGetSubRect(const CvArr* arr, CvMat* submat, CvRect rect)

Returns matrix header corresponding to the rectangular sub-array of input image or matrix
"""

cvGetRows.__doc__ = """CvMat* cvGetRows(const CvArr* arr, CvMat* submat, int start_row, int end_row, int delta_row=1)

Returns array row or row span
"""

cvGetCols.__doc__ = """CvMat* cvGetCols(const CvArr* arr, CvMat* submat, int start_col, int end_col)

Returns array column or column span
"""

cvGetDiag.__doc__ = """CvMat* cvGetDiag(const CvArr* arr, CvMat* submat, int diag=0)

Returns one of array diagonals
"""

cvGetSize.__doc__ = """CvSize cvGetSize(const CvArr* arr)

Returns size of matrix or image ROI
"""

cvGetElemType.__doc__ = """int cvGetElemType(const CvArr* arr)

Returns type of array elements
"""

cvGetDims.__doc__ = """int cvGetDims(const CvArr* arr, int* sizes=NULL)

Return number of array dimensions and their sizes or the size of particular dimension
"""

cvPtr1D.__doc__ = """uchar* cvPtr1D(const CvArr* arr, int idx0, int* type=NULL)

Return pointer to the particular array element
"""

cvGet1D.__doc__ = """CvScalar cvGet1D(const CvArr* arr, int idx0)

Return the particular array element
"""

cvGetReal1D.__doc__ = """double cvGetReal1D(const CvArr* arr, int idx0)

Return the particular element of single-channel array
"""

cvSet1D.__doc__ = """void cvSet1D(CvArr* arr, int idx0, CvScalar value)

Change the particular array element
"""

cvSetReal1D.__doc__ = """void cvSetReal1D(CvArr* arr, int idx0, double value)

Change the particular array element
"""

cvClearND.__doc__ = """void cvClearND(CvArr* arr, int* idx)

Clears the particular array element
"""

cvCopy.__doc__ = """void cvCopy(const CvArr* src, CvArr* dst, const CvArr* mask=NULL)

Copies one array to another
"""

cvSet.__doc__ = """void cvSet(CvArr* arr, CvScalar value, const CvArr* mask=NULL)

Sets every element of array to given value
"""

cvSetZero.__doc__ = """void cvSetZero(CvArr* arr)

Clears the array
"""

cvReshape.__doc__ = """CvMat* cvReshape(const CvArr* arr, CvMat* header, int new_cn, int new_rows=0)

Changes shape of matrix/image without copying data
"""

cvReshapeMatND.__doc__ = """CvArr* cvReshapeMatND(const CvArr* arr, int sizeof_header, CvArr* header, int new_cn, int new_dims, int* new_sizes)

Changes shape of multi-dimensional array w/o copying data
"""

cvRepeat.__doc__ = """void cvRepeat(const CvArr* src, CvArr* dst)

Fill destination array with tiled source array
"""

cvFlip.__doc__ = """void cvFlip(const CvArr* src, CvArr* dst=NULL, int flip_mode=)

Flip a 2D array around vertical, horizontall or both axises
"""

cvSplit.__doc__ = """void cvSplit(const CvArr* src, CvArr* dst0, CvArr* dst1, CvArr* dst2, CvArr* dst3)

Divides multi-channel array into several single-channel arrays or extracts a single channel from the array
"""

cvMerge.__doc__ = """void cvMerge(const CvArr* src0, const CvArr* src1, const CvArr* src2, const CvArr* src3, CvArr* dst)

Composes multi-channel array from several single-channel arrays or inserts a single channel into the array
"""

cvLUT.__doc__ = """void cvLUT(const CvArr* src, CvArr* dst, const CvArr* lut)

Performs look-up table transform of array
"""

cvConvertScale.__doc__ = """void cvConvertScale(const CvArr* src, CvArr* dst, double scale=1, double shift=0)

Converts one array to another with optional linear transformation
"""

cvConvertScaleAbs.__doc__ = """void cvConvertScaleAbs(const CvArr* src, CvArr* dst, double scale=1, double shift=0)

Converts input array elements to 8-bit unsigned integer another with optional linear transformation
"""

cvAdd.__doc__ = """void cvAdd(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)

Computes per-element sum of two arrays
"""

cvAddS.__doc__ = """void cvAddS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)

Computes sum of array and scalar
"""

cvAddWeighted.__doc__ = """void cvAddWeighted(const CvArr* src1, double alpha, const CvArr* src2, double beta, double gamma, CvArr* dst)

Computes weighted sum of two arrays
"""

cvSub.__doc__ = """void cvSub(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)

Computes per-element difference between two arrays
"""

cvSubRS.__doc__ = """void cvSubRS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)

Computes difference between scalar and array
"""

cvMul.__doc__ = """void cvMul(const CvArr* src1, const CvArr* src2, CvArr* dst, double scale=1)

Calculates per-element product of two arrays
"""

cvDiv.__doc__ = """void cvDiv(const CvArr* src1, const CvArr* src2, CvArr* dst, double scale=1)

Performs per-element division of two arrays
"""

cvAnd.__doc__ = """void cvAnd(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)

Calculates per-element bit-wise conjunction of two arrays
"""

cvAndS.__doc__ = """void cvAndS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)

Calculates per-element bit-wise conjunction of array and scalar
"""

cvOr.__doc__ = """void cvOr(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)

Calculates per-element bit-wise disjunction of two arrays
"""

cvOrS.__doc__ = """void cvOrS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)

Calculates per-element bit-wise disjunction of array and scalar
"""

cvXor.__doc__ = """void cvXor(const CvArr* src1, const CvArr* src2, CvArr* dst, const CvArr* mask=NULL)

Performs per-element bit-wise "exclusive or" operation on two arrays
"""

cvXorS.__doc__ = """void cvXorS(const CvArr* src, CvScalar value, CvArr* dst, const CvArr* mask=NULL)

Performs per-element bit-wise "exclusive or" operation on array and scalar
"""

cvNot.__doc__ = """void cvNot(const CvArr* src, CvArr* dst)

Performs per-element bit-wise inversion of array elements
"""

cvCmp.__doc__ = """void cvCmp(const CvArr* src1, const CvArr* src2, CvArr* dst, int cmp_op)

Performs per-element comparison of two arrays
"""

cvCmpS.__doc__ = """void cvCmpS(const CvArr* src, double value, CvArr* dst, int cmp_op)

Performs per-element comparison of array and scalar
"""

cvInRange.__doc__ = """void cvInRange(const CvArr* src, const CvArr* lower, const CvArr* upper, CvArr* dst)

Checks that array elements lie between elements of two other arrays
"""

cvInRangeS.__doc__ = """void cvInRangeS(const CvArr* src, CvScalar lower, CvScalar upper, CvArr* dst)

Checks that array elements lie between two scalars
"""

cvMax.__doc__ = """void cvMax(const CvArr* src1, const CvArr* src2, CvArr* dst)

Finds per-element maximum of two arrays
"""

cvMaxS.__doc__ = """void cvMaxS(const CvArr* src, double value, CvArr* dst)

Finds per-element maximum of array and scalar
"""

cvMin.__doc__ = """void cvMin(const CvArr* src1, const CvArr* src2, CvArr* dst)

Finds per-element minimum of two arrays
"""

cvMinS.__doc__ = """void cvMinS(const CvArr* src, double value, CvArr* dst)

Finds per-element minimum of array and scalar
"""

cvAbsDiff.__doc__ = """void cvAbsDiff(const CvArr* src1, const CvArr* src2, CvArr* dst)

Calculates absolute difference between two arrays
"""

cvAbsDiffS.__doc__ = """void cvAbsDiffS(const CvArr* src, CvArr* dst, CvScalar value)

Calculates absolute difference between array and scalar
"""

cvCountNonZero.__doc__ = """int cvCountNonZero(const CvArr* arr)

Counts non-zero array elements
"""

cvSum.__doc__ = """CvScalar cvSum(const CvArr* arr)

Summarizes array elements
"""

cvAvg.__doc__ = """CvScalar cvAvg(const CvArr* arr, const CvArr* mask=NULL)

Calculates average (mean) of array elements
"""

cvAvgSdv.__doc__ = """void cvAvgSdv(const CvArr* arr, CvScalar* mean, CvScalar* std_dev, const CvArr* mask=NULL)

Calculates average (mean) of array elements
"""

cvMinMaxLoc.__doc__ = """void cvMinMaxLoc(const CvArr* arr, double* min_val, double* max_val, CvPoint* min_loc=NULL, CvPoint* max_loc=NULL, const CvArr* mask=NULL)

Finds global minimum and maximum in array or subarray
"""

cvNorm.__doc__ = """double cvNorm(const CvArr* arr1, const CvArr* arr2=NULL, int norm_type=CV_L2, const CvArr* mask=NULL)

Calculates absolute array norm, absolute difference norm or relative difference norm
"""

cvSetIdentity.__doc__ = """void cvSetIdentity(CvArr* mat, CvScalar value=cvRealScalar(1)

Initializes scaled identity matrix
"""

cvDotProduct.__doc__ = """double cvDotProduct(const CvArr* src1, const CvArr* src2)

Calculates dot product of two arrays in Euclidian metrics
"""

cvCrossProduct.__doc__ = """void cvCrossProduct(const CvArr* src1, const CvArr* src2, CvArr* dst)

Calculates cross product of two 3D vectors
"""

cvScaleAdd.__doc__ = """void cvScaleAdd(const CvArr* src1, CvScalar scale, const CvArr* src2, CvArr* dst)

Calculates sum of scaled array and another array
"""

cvGEMM.__doc__ = """void cvGEMM(const CvArr* src1, const CvArr* src2, double alpha, const CvArr* src3, double beta, CvArr* dst, int tABC=0)

Performs generalized matrix multiplication
"""

cvTransform.__doc__ = """void cvTransform(const CvArr* src, CvArr* dst, const CvMat* transmat, const CvMat* shiftvec=NULL)

Performs matrix transform of every array element
"""

cvPerspectiveTransform.__doc__ = """void cvPerspectiveTransform(const CvArr* src, CvArr* dst, const CvMat* mat)

Performs perspective matrix transform of vector array
"""

cvMulTransposed.__doc__ = """void cvMulTransposed(const CvArr* src, CvArr* dst, int order, const CvArr* delta=NULL)

Calculates product of array and transposed array
"""

cvTrace.__doc__ = """CvScalar cvTrace(const CvArr* mat)

Returns trace of matrix
"""

cvTranspose.__doc__ = """void cvTranspose(const CvArr* src, CvArr* dst)

Transposes matrix
"""

cvDet.__doc__ = """double cvDet(const CvArr* mat)

Returns determinant of matrix
"""

cvInvert.__doc__ = """double cvInvert(const CvArr* src, CvArr* dst, int method=CV_LU)

Finds inverse or pseudo-inverse of matrix
"""

cvSolve.__doc__ = """int cvSolve(const CvArr* src1, const CvArr* src2, CvArr* dst, int method=CV_LU)

Solves linear system or least-squares problem
"""

cvSVD.__doc__ = """void cvSVD(CvArr* A, CvArr* W, CvArr* U=NULL, CvArr* V=NULL, int flags=0)

Performs singular value decomposition of real floating-point matrix
"""

cvSVBkSb.__doc__ = """void cvSVBkSb(const CvArr* W, const CvArr* U, const CvArr* V, const CvArr* B, CvArr* X, int flags)

Performs singular value back substitution
"""

cvEigenVV.__doc__ = """void cvEigenVV(CvArr* mat, CvArr* evects, CvArr* evals, double eps=0)

Computes eigenvalues and eigenvectors of symmetric matrix
"""

cvCalcCovarMatrix.__doc__ = """void cvCalcCovarMatrix(const CvArr** vects, int count, CvArr* cov_mat, CvArr* avg, int flags)

Calculates covariation matrix of the set of vectors
"""

cvMahalanobis.__doc__ = """double cvMahalanobis(const CvArr* vec1, const CvArr* vec2, CvArr* mat)

Calculates Mahalonobis distance between two vectors
"""

cvCbrt.__doc__ = """float cvCbrt(float value)

Calculates cubic root
"""

cvFastArctan.__doc__ = """float cvFastArctan(float y, float x)

Calculates angle of 2D vector
"""

cvCartToPolar.__doc__ = """void cvCartToPolar(const CvArr* x, const CvArr* y, CvArr* magnitude, CvArr* angle=NULL, int angle_in_degrees=0)

Calculates magnitude and/or angle of 2d vectors
"""

cvPolarToCart.__doc__ = """void cvPolarToCart(const CvArr* magnitude, const CvArr* angle, CvArr* x, CvArr* y, int angle_in_degrees=0)

Calculates cartesian coordinates of 2d vectors represented in polar form
"""

cvPow.__doc__ = """void cvPow(const CvArr* src, CvArr* dst, double power)

Raises every array element to power
"""

cvExp.__doc__ = """void cvExp(const CvArr* src, CvArr* dst)

Calculates exponent of every array element
"""

cvLog.__doc__ = """void cvLog(const CvArr* src, CvArr* dst)

Calculates natural logarithm of every array element absolute value
"""

cvSolveCubic.__doc__ = """void cvSolveCubic(const CvArr* coeffs, CvArr* roots)

Finds real roots of a cubic equation
"""

cvRandArr.__doc__ = """void cvRandArr(CvRNG* rng, CvArr* arr, int dist_type, CvScalar param1, CvScalar param2)

Fills array with random numbers and updates the RNG state
"""

cvGetOptimalDFTSize.__doc__ = """int cvGetOptimalDFTSize(int size0)

Returns optimal DFT size for given vector size
"""

cvMulSpectrums.__doc__ = """void cvMulSpectrums(const CvArr* src1, const CvArr* src2, CvArr* dst, int flags)

Performs per-element multiplication of two Fourier spectrums
"""

cvCreateMemStorage.__doc__ = """CvMemStorage* cvCreateMemStorage(int block_size=0)

Creates memory storage
"""

cvCreateChildMemStorage.__doc__ = """CvMemStorage* cvCreateChildMemStorage(CvMemStorage* parent)

Creates child memory storage
"""

cvReleaseMemStorage.__doc__ = """void cvReleaseMemStorage(CvMemStorage** storage)

Releases memory storage
"""

cvClearMemStorage.__doc__ = """void cvClearMemStorage(CvMemStorage* storage)

Clears memory storage
"""

cvMemStorageAlloc.__doc__ = """void* cvMemStorageAlloc(CvMemStorage* storage, size_t size)

Allocates memory buffer in the storage
"""

cvMemStorageAllocString.__doc__ = """CvString cvMemStorageAllocString(CvMemStorage* storage, const char* ptr, int len=-1)

Allocates text string in the storage
"""

cvSaveMemStoragePos.__doc__ = """void cvSaveMemStoragePos(const CvMemStorage* storage, CvMemStoragePos* pos)

Saves memory storage position
"""

cvRestoreMemStoragePos.__doc__ = """void cvRestoreMemStoragePos(CvMemStorage* storage, CvMemStoragePos* pos)

Restores memory storage position
"""

cvCreateSeq.__doc__ = """CvSeq* cvCreateSeq(int seq_flags, int header_size, int elem_size, CvMemStorage* storage)

Creates sequence
"""

cvSetSeqBlockSize.__doc__ = """void cvSetSeqBlockSize(CvSeq* seq, int delta_elems)

Sets up sequence block size
"""

cvSeqPush.__doc__ = """char* cvSeqPush(CvSeq* seq, void* element=NULL)

Adds element to sequence end
"""

cvSeqPop.__doc__ = """void cvSeqPop(CvSeq* seq, void* element=NULL)

Removes element from sequence end
"""

cvSeqPushFront.__doc__ = """char* cvSeqPushFront(CvSeq* seq, void* element=NULL)

Adds element to sequence beginning
"""

cvSeqPopFront.__doc__ = """void cvSeqPopFront(CvSeq* seq, void* element=NULL)

Removes element from sequence beginning
"""

cvSeqPushMulti.__doc__ = """void cvSeqPushMulti(CvSeq* seq, void* elements, int count, int in_front=0)

Pushes several elements to the either end of sequence
"""

cvSeqPopMulti.__doc__ = """void cvSeqPopMulti(CvSeq* seq, void* elements, int count, int in_front=0)

Removes several elements from the either end of sequence
"""

cvSeqInsert.__doc__ = """char* cvSeqInsert(CvSeq* seq, int before_index, void* element=NULL)

Inserts element in sequence middle
"""

cvSeqRemove.__doc__ = """void cvSeqRemove(CvSeq* seq, int index)

Removes element from sequence middle
"""

cvClearSeq.__doc__ = """void cvClearSeq(CvSeq* seq)

Clears sequence
"""

cvGetSeqElem.__doc__ = """char* cvGetSeqElem(const CvSeq* seq, int index)

Returns pointer to sequence element by its index
"""

cvSeqElemIdx.__doc__ = """int cvSeqElemIdx(const CvSeq* seq, const void* element, CvSeqBlock** block=NULL)

Returns index of concrete sequence element
"""

cvCvtSeqToArray.__doc__ = """void* cvCvtSeqToArray(const CvSeq* seq, void* elements, CvSlice slice=CV_WHOLE_SEQ)

Copies sequence to one continuous block of memory
"""

cvMakeSeqHeaderForArray.__doc__ = """CvSeq* cvMakeSeqHeaderForArray(int seq_type, int header_size, int elem_size,                                void* elements, int total,                                CvSeq* seq, CvSeqBlock* block)

Constructs sequence from array
"""

cvSeqSlice.__doc__ = """CvSeq* cvSeqSlice(const CvSeq* seq, CvSlice slice,                   CvMemStorage* storage=NULL, int copy_data=0)

Makes separate header for the sequence slice
"""

cvSeqRemoveSlice.__doc__ = """void cvSeqRemoveSlice(CvSeq* seq, CvSlice slice)

Removes sequence slice
"""

cvSeqInsertSlice.__doc__ = """void cvSeqInsertSlice(CvSeq* seq, int before_index, const CvArr* from_arr)

Inserts array in the middle of sequence
"""

cvSeqInvert.__doc__ = """void cvSeqInvert(CvSeq* seq)

Reverses the order of sequence elements
"""

cvSeqSort.__doc__ = """void cvSeqSort(CvSeq* seq, CvCmpFunc func, void* userdata=NULL)

Sorts sequence element using the specified comparison function
"""

cvSeqSearch.__doc__ = """char* cvSeqSearch(CvSeq* seq, const void* elem, CvCmpFunc func, int is_sorted, int* elem_idx, void* userdata=NULL)

Searches element in sequence
"""

cvStartAppendToSeq.__doc__ = """void cvStartAppendToSeq(CvSeq* seq, CvSeqWriter* writer)

Initializes process of writing data to sequence
"""

cvStartWriteSeq.__doc__ = """void cvStartWriteSeq(int seq_flags, int header_size, int elem_size, CvMemStorage* storage, CvSeqWriter* writer)

Creates new sequence and initializes writer for it
"""

cvEndWriteSeq.__doc__ = """CvSeq* cvEndWriteSeq(CvSeqWriter* writer)

Finishes process of writing sequence
"""

cvFlushSeqWriter.__doc__ = """void cvFlushSeqWriter(CvSeqWriter* writer)

Updates sequence headers from the writer state
"""

cvStartReadSeq.__doc__ = """void cvStartReadSeq(const CvSeq* seq, CvSeqReader* reader, int reverse=0)

Initializes process of sequential reading from sequence
"""

cvGetSeqReaderPos.__doc__ = """int cvGetSeqReaderPos(CvSeqReader* reader)

Returns the current reader position
"""

cvSetSeqReaderPos.__doc__ = """void cvSetSeqReaderPos(CvSeqReader* reader, int index, int is_relative=0)

Moves the reader to specified position
"""

cvCreateSet.__doc__ = """CvSET* cvCreateSet(int set_flags, int header_size, int elem_size, CvMemStorage* storage)

Creates empty set
"""

cvSetAdd.__doc__ = """int cvSetAdd(CvSET* set_header, CvSetElem* elem=NULL, CvSetElem** inserted_elem=NULL)

Occupies a node in the set
"""

cvSetRemove.__doc__ = """void cvSetRemove(CvSET* set_header, int index)

Removes element from set
"""

cvClearSet.__doc__ = """void cvClearSet(CvSET* set_header)

Clears set
"""

cvCreateGraph.__doc__ = """CvGraph* cvCreateGraph(int graph_flags, int header_size, int vtx_size, int edge_size, CvMemStorage* storage)

Creates empty graph
"""

cvGraphAddVtx.__doc__ = """int cvGraphAddVtx(CvGraph* graph, const CvGraphVtx* vtx=NULL, CvGraphVtx** inserted_vtx=NULL)

Adds vertex to graph
"""

cvGraphRemoveVtx.__doc__ = """int cvGraphRemoveVtx(CvGraph* graph, int index)

Removes vertex from graph
"""

cvGraphRemoveVtxByPtr.__doc__ = """int cvGraphRemoveVtxByPtr(CvGraph* graph, CvGraphVtx* vtx)

Removes vertex from graph
"""

cvGraphAddEdge.__doc__ = """int cvGraphAddEdge(CvGraph* graph, int start_idx, int end_idx, const CvGraphEdge* edge=NULL, CvGraphEdge** inserted_edge=NULL)

Adds edge to graph
"""

cvGraphAddEdgeByPtr.__doc__ = """int cvGraphAddEdgeByPtr(CvGraph* graph, CvGraphVtx* start_vtx, CvGraphVtx* end_vtx, const CvGraphEdge* edge=NULL, CvGraphEdge** inserted_edge=NULL)

Adds edge to graph
"""

cvGraphRemoveEdge.__doc__ = """void cvGraphRemoveEdge(CvGraph* graph, int start_idx, int end_idx)

Removes edge from graph
"""

cvGraphRemoveEdgeByPtr.__doc__ = """void cvGraphRemoveEdgeByPtr(CvGraph* graph, CvGraphVtx* start_vtx, CvGraphVtx* end_vtx)

Removes edge from graph
"""

cvFindGraphEdge.__doc__ = """CvGraphEdge* cvFindGraphEdge(const CvGraph* graph, int start_idx, int end_idx)

Finds edge in graph
"""

cvFindGraphEdgeByPtr.__doc__ = """CvGraphEdge* cvFindGraphEdgeByPtr(const CvGraph* graph, const CvGraphVtx* start_vtx, const CvGraphVtx* end_vtx)

Finds edge in graph
"""

cvGraphVtxDegree.__doc__ = """int cvGraphVtxDegree(const CvGraph* graph, int vtx_idx)

Counts edges indicent to the vertex
"""

cvGraphVtxDegreeByPtr.__doc__ = """int cvGraphVtxDegreeByPtr(const CvGraph* graph, const CvGraphVtx* vtx)

Finds edge in graph
"""

cvClearGraph.__doc__ = """void cvClearGraph(CvGraph* graph)

Clears graph
"""

cvCloneGraph.__doc__ = """CvGraph* cvCloneGraph(const CvGraph* graph, CvMemStorage* storage)

Clone graph
"""

cvCreateGraphScanner.__doc__ = """CvGraphScanner* cvCreateGraphScanner(CvGraph* graph, CvGraphVtx* vtx=NULL, int mask=CV_GRAPH_ALL_ITEMS)

Creates structure for depth-first graph traversal
"""

cvNextGraphItem.__doc__ = """int cvNextGraphItem(CvGraphScanner* scanner)

Makes one or more steps of the graph traversal procedure
"""

cvReleaseGraphScanner.__doc__ = """void cvReleaseGraphScanner(CvGraphScanner** scanner)

Finishes graph traversal procedure
"""

cvInitTreeNodeIterator.__doc__ = """void cvInitTreeNodeIterator(CvTreeNodeIterator* tree_iterator, const void* first, int max_level)

Initializes tree node iterator
"""

cvNextTreeNode.__doc__ = """void* cvNextTreeNode(CvTreeNodeIterator* tree_iterator)

Returns the currently observed node and moves iterator toward the next node
"""

cvPrevTreeNode.__doc__ = """void* cvPrevTreeNode(CvTreeNodeIterator* tree_iterator)

Returns the currently observed node and moves iterator toward the previous node
"""

cvTreeToNodeSeq.__doc__ = """CvSeq* cvTreeToNodeSeq(const void* first, int header_size, CvMemStorage* storage)

Gathers all node pointers to the single sequence
"""

cvInsertNodeIntoTree.__doc__ = """void cvInsertNodeIntoTree(void* node, void* parent, void* frame)

Adds new node to the tree
"""

cvRemoveNodeFromTree.__doc__ = """void cvRemoveNodeFromTree(void* node, void* frame)

Removes node from tree
"""

cvLine.__doc__ = """void cvLine(CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness=1, int line_type=8, int shift=0)

Draws a line segment connecting two points
"""

cvRectangle.__doc__ = """void cvRectangle(CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color,                  int thickness=1, int line_type=8, int shift=0)

Draws simple, thick or filled rectangle
"""

cvCircle.__doc__ = """void cvCircle(CvArr* img, CvPoint center, int radius, CvScalar color, int thickness=1, int line_type=8, int shift=0)

Draws a circle
"""

cvEllipse.__doc__ = """void cvEllipse(CvArr* img, CvPoint center, CvSize axes, double angle, double start_angle, double end_angle, CvScalar color, int thickness=1, int line_type=8, int shift=0)

Draws simple or thick elliptic arc or fills ellipse sector
"""

cvFillPoly.__doc__ = """void cvFillPoly(CvArr* img, CvPoint** pts, int* npts, int contours, CvScalar color, int line_type=8, int shift=0)

Fills polygons interior
"""

cvFillConvexPoly.__doc__ = """void cvFillConvexPoly(CvArr* img, CvPoint* pts, int npts, CvScalar color, int line_type=8, int shift=0)

Fills convex polygon
"""

cvPolyLine.__doc__ = """void cvPolyLine(CvArr* img, CvPoint** pts, int* npts, int contours, int is_closed, CvScalar color, int thickness=1, int line_type=8, int shift=0)

Draws simple or thick polygons
"""

cvInitFont.__doc__ = """void cvInitFont(CvFont* font, int font_face, double hscale, double vscale, double shear=0, int thickness=1, int line_type=8)

Initializes font structure
"""

cvPutText.__doc__ = """void cvPutText(CvArr* img, const char* text, CvPoint org, const CvFont* font, CvScalar color)

Draws text string
"""

cvGetTextSize.__doc__ = """void cvGetTextSize(const char* text_string, const CvFont* font, CvSize* text_size, int* baseline)

Retrieves width and height of text string
"""

cvDrawContours.__doc__ = """void cvDrawContours(CvArr* img, CvSeq* contour, CvScalar external_color, CvScalar hole_color, int max_level, int thickness=1, int line_type=8)

Draws contour outlines or interiors in the image
"""

cvInitLineIterator.__doc__ = """int cvInitLineIterator(const CvArr* image, CvPoint pt1, CvPoint pt2, CvLineIterator* line_iterator, int connectivity=8, int left_to_right=0)

Initializes line iterator
"""

cvClipLine.__doc__ = """int cvClipLine(CvSize img_size, CvPoint* pt1, CvPoint* pt2)

Clips the line against the image rectangle
"""

cvEllipse2Poly.__doc__ = """int cvEllipse2Poly(CvPoint center, CvSize axes, int angle, int arc_start, int arc_end, CvPoint* pts, int delta)

Approximates elliptic arc with polyline
"""

cvOpenFileStorage.__doc__ = """CvFileStorage* cvOpenFileStorage(const char* filename, CvMemStorage* memstorage, int flags)

Opens file storage for reading or writing data
"""

cvReleaseFileStorage.__doc__ = """void cvReleaseFileStorage(CvFileStorage** fs)

Releases file storage
"""

cvStartWriteStruct.__doc__ = """void cvStartWriteStruct(CvFileStorage* fs, const char* name, int struct_flags, const char* type_name=NULL, CvAttrList attributes=cvAttrLis)

Starts writing a new structure
"""

cvEndWriteStruct.__doc__ = """void cvEndWriteStruct(CvFileStorage* fs)

Ends writing a structure
"""

cvWriteInt.__doc__ = """void cvWriteInt(CvFileStorage* fs, const char* name, int value)

Writes an integer value
"""

cvWriteReal.__doc__ = """void cvWriteReal(CvFileStorage* fs, const char* name, double value)

Writes a floating-point value
"""

cvWriteString.__doc__ = """void cvWriteString(CvFileStorage* fs, const char* name, const char* str, int quote=0)

Writes a text string
"""

cvWriteComment.__doc__ = """void cvWriteComment(CvFileStorage* fs, const char* comment, int eol_comment)

Writes comment
"""

cvStartNextStream.__doc__ = """void cvStartNextStream(CvFileStorage* fs)

Starts the next stream
"""

cvWrite.__doc__ = """void cvWrite(CvFileStorage* fs, const char* name, const void* ptr, CvAttrList attributes=cvAttrList)

Writes user object
"""

cvWriteRawData.__doc__ = """void cvWriteRawData(CvFileStorage* fs, const void* src, int len, const char* dt)

Writes multiple numbers
"""

cvWriteFileNode.__doc__ = """void cvWriteFileNode(CvFileStorage* fs, const char* new_node_name, const CvFileNode* node, int embed)

Writes file node to another file storage
"""

cvGetRootFileNode.__doc__ = """CvFileNode* cvGetRootFileNode(const CvFileStorage* fs, int stream_index=0)

Retrieves one of top-level nodes of the file storage
"""

cvGetFileNodeByName.__doc__ = """CvFileNode* cvGetFileNodeByName(const CvFileStorage* fs, const CvFileNode* map, const char* name)

Finds node in the map or file storage
"""

cvGetHashedKey.__doc__ = """CvStringHashNode* cvGetHashedKey(CvFileStorage* fs, const char* name, int len=-1, int create_missing=0)

Returns a unique pointer for given name
"""

cvGetFileNode.__doc__ = """CvFileNode* cvGetFileNode(CvFileStorage* fs, CvFileNode* map, const CvStringHashNode* key, int create_missing=0)

Finds node in the map or file storage
"""

cvGetFileNodeName.__doc__ = """const char* cvGetFileNodeName(const CvFileNode* node)

Returns name of file node
"""

cvRead.__doc__ = """void* cvRead(CvFileStorage* fs, CvFileNode* node, CvAttrList* attributes=NULL)

Decodes object and returns pointer to it
"""

cvReadRawData.__doc__ = """void cvReadRawData(const CvFileStorage* fs, const CvFileNode* src, void* dst, const char* dt)

Reads multiple numbers
"""

cvStartReadRawData.__doc__ = """void cvStartReadRawData(const CvFileStorage* fs, const CvFileNode* src, CvSeqReader* reader)

Initializes file node sequence reader
"""

cvReadRawDataSlice.__doc__ = """void cvReadRawDataSlice(const CvFileStorage* fs, CvSeqReader* reader, int count, void* dst, const char* dt)

Initializes file node sequence reader
"""

cvRegisterType.__doc__ = """void cvRegisterType(const CvTypeInfo* info)

Registers new type
"""

cvUnregisterType.__doc__ = """void cvUnregisterType(const char* type_name)

Unregisters the type
"""

cvFirstType.__doc__ = """CvTypeInfo* cvFirstType(voi)

Returns the beginning of type list
"""

cvFindType.__doc__ = """CvTypeInfo* cvFindType(const char* type_name)

Finds type by its name
"""

cvTypeOf.__doc__ = """CvTypeInfo* cvTypeOf(const void* struct_ptr)

Returns type of the object
"""

cvRelease.__doc__ = """void cvRelease(void** struct_ptr)

Releases the object
"""

cvClone.__doc__ = """void* cvClone(const void* struct_ptr)

Makes a clone of the object
"""

cvSave.__doc__ = """void cvSave(const char* filename, const void* struct_ptr, const char* name=NULL, const char* comment=NULL, CvAttrList attributes=cvAttrLis)

Saves object to file
"""

cvLoad.__doc__ = """void* cvLoad(const char* filename, CvMemStorage* memstorage=NULL, const char* name=NULL, const char** real_name=NULL)

Loads object from file
"""

cvCheckArr.__doc__ = """int cvCheckArr(const CvArr* arr, int flags=0, double min_val=0, double max_val=)

Checks every element of input array for invalid values
"""

cvKMeans2.__doc__ = """void cvKMeans2(const CvArr* samples, int cluster_count, CvArr* labels, CvTermCriteria termcrit)

Splits set of vectors by given number of clusters
"""

cvSeqPartition.__doc__ = """int cvSeqPartition(const CvSeq* seq, CvMemStorage* storage, CvSeq** labels, CvCmpFunc is_equal, void* userdata)

Splits sequence into equivalency classes
"""

cvGetErrStatus.__doc__ = """int cvGetErrStatus(void)

Returns the current error status
"""

cvSetErrStatus.__doc__ = """void cvSetErrStatus(int status)

Sets the error status
"""

cvGetErrMode.__doc__ = """int cvGetErrMode(void)

Returns the current error mode
"""

cvError.__doc__ = """int cvError(int status, const char* func_name, const char* err_msg, const char* file_name, int line)

Raises an error
"""

cvErrorStr.__doc__ = """const char* cvErrorStr(int status)

Returns textual description of error status code
"""

cvNulDevReport.__doc__ = """int cvNulDevReport(int status, const char* func_name, const char* err_msg, const char* file_name, int line, void* userdata)

Provide standard error handling
"""

cvAlloc.__doc__ = """void* cvAlloc(size_t size)

Allocates memory buffer
"""

#cvFree.__doc__ = """void cvFree(void** ptr)
#
#Deallocates memory buffer
#"""

cvGetTickCount.__doc__ = """int64 cvGetTickCount(void)

Returns number of tics
"""

cvGetTickFrequency.__doc__ = """double cvGetTickFrequency(void)

Returns number of tics per microsecond
"""

cvRegisterModule.__doc__ = """int cvRegisterModule(const CvModuleInfo* module_info)

Registers another module
"""

cvGetModuleInfo.__doc__ = """void cvGetModuleInfo(const char* module_name, const char** version, const char** loaded_addon_plugins)

Retrieves information about the registered module(s) and plugins
"""

cvUseOptimized.__doc__ = """int cvUseOptimized(int on_off)

Switches between optimized/non-optimized modes
"""

cvSobel.__doc__ = """void cvSobel(const CvArr* src, CvArr* dst, int xorder, int yorder, int aperture_size=3)

Calculates first, second, third or mixed image derivatives using extended Sobel operator
"""

cvLaplace.__doc__ = """void cvLaplace(const CvArr* src, CvArr* dst, int aperture_size=3)

Calculates Laplacian of the image
"""

cvCanny.__doc__ = """void cvCanny(const CvArr* image, CvArr* edges, double threshold1, double threshold2, int aperture_size=3)

Implements Canny algorithm for edge detection
"""

cvPreCornerDetect.__doc__ = """void cvPreCornerDetect(const CvArr* image, CvArr* corners, int aperture_size=3)

Calculates feature map for corner detection
"""

cvCornerEigenValsAndVecs.__doc__ = """void cvCornerEigenValsAndVecs(const CvArr* image, CvArr* eigenvv, int block_size, int aperture_size=3)

Calculates eigenvalues and eigenvectors of image blocks for corner detection
"""

cvCornerMinEigenVal.__doc__ = """void cvCornerMinEigenVal(const CvArr* image, CvArr* eigenval, int block_size, int aperture_size=3)

Calculates minimal eigenvalue of gradient matrices for corner detection
"""

cvCornerHarris.__doc__ = """void cvCornerHarris(const CvArr* image, CvArr* harris_responce, int block_size, int aperture_size=3, double k=0.04)

Harris edge detector
"""

cvFindCornerSubPix.__doc__ = """void cvFindCornerSubPix(const CvArr* image, CvPoint2D32f* corners, int count, CvSize win, CvSize zero_zone, CvTermCriteria criteria)

Refines corner locations
"""

cvGoodFeaturesToTrack.__doc__ = """void cvGoodFeaturesToTrack(const CvArr* image, CvArr* eig_image, CvArr* temp_image, CvPoint2D32f* corners, int* corner_count, double quality_level, double min_distance, const CvArr* mask=NULL, int block_size=3, int use_harris=0, double k=0.04)

Determines strong corners on image
"""

cvSampleLine.__doc__ = """int cvSampleLine(const CvArr* image, CvPoint pt1, CvPoint pt2, void* buffer, int connectivity=8)

Reads raster line to buffer
"""

cvGetRectSubPix.__doc__ = """void cvGetRectSubPix(const CvArr* src, CvArr* dst, CvPoint2D32f center)

Retrieves pixel rectangle from image with sub-pixel accuracy
"""

cvGetQuadrangleSubPix.__doc__ = """void cvGetQuadrangleSubPix(const CvArr* src, CvArr* dst, const CvMat* map_matrix)

Retrieves pixel quadrangle from image with sub-pixel accuracy
"""

cvResize.__doc__ = """void cvResize(const CvArr* src, CvArr* dst, int interpolation=CV_INTER_LINEAR)

Resizes image
"""

cvGetAffineTransform.__doc__ = """CvMat* cvGetAffineTransform(const CvPoint2D32f* src, const CvPoint2D32f* dst, CvMat* map_matrix)

Calculates affine transform from 3 corresponding points
"""

cvWarpAffine.__doc__ = """void cvWarpAffine(const CvArr* src, CvArr* dst, const CvMat* map_matrix, int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, CvScalar fillval=cvScalarAll(0)

Applies affine transformation to the image
"""

cv2DRotationMatrix.__doc__ = """CvMat* cv2DRotationMatrix(CvPoint2D32f center, double angle, double scale, CvMat* map_matrix)

Calculates affine matrix of 2d rotation
"""

cvWarpPerspective.__doc__ = """void cvWarpPerspective(const CvArr* src, CvArr* dst, const CvMat* map_matrix, int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, CvScalar fillval=cvScalarAll(0)

Applies perspective transformation to the image
"""

cvGetPerspectiveTransform.__doc__ = """CvMat* cvGetPerspectiveTransform(const CvPoint2D32f* src, const CvPoint2D32f* dst, CvMat* map_matrix)

Calculates perspective transform from 4 corresponding points
"""

cvRemap.__doc__ = """void cvRemap(const CvArr* src, CvArr* dst, const CvArr* mapx, const CvArr* mapy, int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, CvScalar fillval=cvScalarAll(0)

Applies generic geometrical transformation to the image
"""

cvLogPolar.__doc__ = """void cvLogPolar(const CvArr* src, CvArr* dst, CvPoint2D32f center, double M, int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS)

Remaps image to log-polar space
"""

cvCreateStructuringElementEx.__doc__ = """IplConvKernel* cvCreateStructuringElementEx(int cols, int rows, int anchor_x, int anchor_y, int shape, int* values=NULL)

Creates structuring element
"""

cvReleaseStructuringElement.__doc__ = """void cvReleaseStructuringElement(IplConvKernel** element)

Deletes structuring element
"""

cvErode.__doc__ = """void cvErode(const CvArr* src, CvArr* dst, IplConvKernel* element=NULL, int iterations=1)

Erodes image by using arbitrary structuring element
"""

cvDilate.__doc__ = """void cvDilate(const CvArr* src, CvArr* dst, IplConvKernel* element=NULL, int iterations=1)

Dilates image by using arbitrary structuring element
"""

cvMorphologyEx.__doc__ = """void cvMorphologyEx(const CvArr* src, CvArr* dst, CvArr* temp, IplConvKernel* element, int operation, int iterations=1)

Performs advanced morphological transformations
"""

cvSmooth.__doc__ = """void cvSmooth(const CvArr* src, CvArr* dst, int smoothtype=CV_GAUSSIAN, int param1=3, int param2=0, double param3=0)

Smooths the image in one of several ways
"""

cvFilter2D.__doc__ = """void cvFilter2D(const CvArr* src, CvArr* dst, const CvMat* kernel, CvPoint anchor=cvPoint(-1, -1)

Convolves image with the kernel
"""

cvCopyMakeBorder.__doc__ = """void cvCopyMakeBorder(const CvArr* src, CvArr* dst, CvPoint offset, int bordertype, CvScalar value=cvScalarAll(0)

Copies image and makes border around it
"""

cvIntegral.__doc__ = """void cvIntegral(const CvArr* image, CvArr* sum, CvArr* sqsum=NULL, CvArr* tilted_sum=NULL)

Calculates integral images
"""

cvCvtColor.__doc__ = """void cvCvtColor(const CvArr* src, CvArr* dst, int code)

Converts image from one color space to another
"""

cvThreshold.__doc__ = """void cvThreshold(const CvArr* src, CvArr* dst, double threshold, double max_value, int threshold_type)

Applies fixed-level threshold to array elements
"""

cvAdaptiveThreshold.__doc__ = """void cvAdaptiveThreshold(const CvArr* src, CvArr* dst, double max_value, int adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C, int threshold_type=CV_THRESH_BINARY, int block_size=3, double param1=5)

Applies adaptive threshold to array
"""

cvPyrDown.__doc__ = """void cvPyrDown(const CvArr* src, CvArr* dst, int filter=CV_GAUSSIAN_5x5)

Downsamples image
"""

cvPyrUp.__doc__ = """void cvPyrUp(const CvArr* src, CvArr* dst, int filter=CV_GAUSSIAN_5x5)

Upsamples image
"""

cvPyrSegmentation.__doc__ = """void cvPyrSegmentation(IplImage* src, IplImage* dst, CvMemStorage* storage, CvSeq** comp, int level, double threshold1, double threshold2)

Implements image segmentation by pyramids
"""

cvFloodFill.__doc__ = """void cvFloodFill(CvArr* image, CvPoint seed_point, CvScalar new_val, CvScalar lo_diff=cvScalarAll(0), CvScalar up_diff=cvScalarAll(0), CvConnectedComp* comp=NULL, int flags=4, CvArr* mask=NULL)

Fills a connected component with given color
"""

cvFindContours.__doc__ = """int cvFindContours(CvArr* image, CvMemStorage* storage, CvSeq** first_contour, int header_size=sizeofCvContour, int mode=CV_RETR_LIST, int method=CV_CHAIN_APPROX_SIMPLE, CvPoint offset=cvPoint(0, 0)

Finds contours in binary image
"""

cvStartFindContours.__doc__ = """CvContourScanner cvStartFindContours(CvArr* image, CvMemStorage* storage, int header_size=sizeofCvContour, int mode=CV_RETR_LIST, int method=CV_CHAIN_APPROX_SIMPLE, CvPoint offset=cvPoint(0, 0)

Initializes contour scanning process
"""

cvFindNextContour.__doc__ = """CvSeq* cvFindNextContour(CvContourScanner scanner)

Finds next contour in the image
"""

cvSubstituteContour.__doc__ = """void cvSubstituteContour(CvContourScanner scanner, CvSeq* new_contour)

Replaces retrieved contour
"""

cvEndFindContours.__doc__ = """CvSeq* cvEndFindContours(CvContourScanner* scanner)

Finishes scanning process
"""

cvMoments.__doc__ = """void cvMoments(const CvArr* arr, CvMOMENTS* moments, int binary=0)

Calculates all moments up to third order of a polygon or rasterized shape
"""

cvGetSpatialMoment.__doc__ = """double cvGetSpatialMoment(CvMOMENTS* moments, int x_order, int y_order)

Retrieves spatial moment from moment state structure
"""

cvGetCentralMoment.__doc__ = """double cvGetCentralMoment(CvMOMENTS* moments, int x_order, int y_order)

Retrieves central moment from moment state structure
"""

cvGetNormalizedCentralMoment.__doc__ = """double cvGetNormalizedCentralMoment(CvMOMENTS* moments, int x_order, int y_order)

Retrieves normalized central moment from moment state structure
"""

cvGetHuMoments.__doc__ = """void cvGetHuMoments(CvMOMENTS* moments, CvHuMoments* hu_moments)

Calculates seven Hu invariants
"""

cvHoughLines2.__doc__ = """CvSeq* cvHoughLines2(CvArr* image, void* line_storage, int method, double rho, double theta, int threshold, double param1=0, double param2=0)

Finds lines in binary image using Hough transform
"""

cvHoughCircles.__doc__ = """CvSeq* cvHoughCircles(CvArr* image, void* circle_storage, int method, double dp, double min_dist, double param1=100, double param2=100)

Finds circles in grayscale image using Hough transform
"""

cvDistTransform.__doc__ = """void cvDistTransform(const CvArr* src, CvArr* dst, int distance_type=CV_DIST_L2, int mask_size=3, const float* mask=NULL, CvArr* labels=NULL)

Calculates distance to closest zero pixel for all non-zero pixels of source image
"""

cvCreateHist.__doc__ = """CvHistogram* cvCreateHist(int dims, int* sizes, int type, float** ranges=NULL, int uniform=1)

Creates histogram
"""

cvSetHistBinRanges.__doc__ = """void cvSetHistBinRanges(CvHistogram* hist, float** ranges, int uniform=1)

Sets bounds of histogram bins
"""

cvReleaseHist.__doc__ = """void cvReleaseHist(CvHistogram** hist)

Releases histogram
"""

cvClearHist.__doc__ = """void cvClearHist(CvHistogram* hist)

Clears histogram
"""

cvMakeHistHeaderForArray.__doc__ = """CvHistogram* cvMakeHistHeaderForArray(int dims, int* sizes, CvHistogram* hist, float* data, float** ranges=NULL, int uniform=1)

Makes a histogram out of array
"""

cvGetMinMaxHistValue.__doc__ = """void cvGetMinMaxHistValue(const CvHistogram* hist, float* min_value, float* max_value, int* min_idx=NULL, int* max_idx=NULL)

Finds minimum and maximum histogram bins
"""

cvNormalizeHist.__doc__ = """void cvNormalizeHist(CvHistogram* hist, double factor)

Normalizes histogram
"""

cvThreshHist.__doc__ = """void cvThreshHist(CvHistogram* hist, double threshold)

Thresholds histogram
"""

cvCompareHist.__doc__ = """double cvCompareHist(const CvHistogram* hist1, const CvHistogram* hist2, int method)

Compares two dense histograms
"""

cvCopyHist.__doc__ = """void cvCopyHist(const CvHistogram* src, CvHistogram** dst)

Copies histogram
"""

cvCalcProbDensity.__doc__ = """void cvCalcProbDensity(const CvHistogram* hist1, const CvHistogram* hist2, CvHistogram* dst_hist, double scale=255)

Divides one histogram by another
"""

cvEqualizeHist.__doc__ = """void cvEqualizeHist(const CvArr* src, CvArr* dst)

Equalizes histogram of grayscale image
"""

cvMatchTemplate.__doc__ = """void cvMatchTemplate(const CvArr* image, const CvArr* templ, CvArr* result, int method)

Compares template against overlapped image regions
"""

cvMatchShapes.__doc__ = """double cvMatchShapes(const void* object1, const void* object2, int method, double parameter=0)

Compares two shapes
"""

cvApproxChains.__doc__ = """CvSeq* cvApproxChains(CvSeq* src_seq, CvMemStorage* storage, int method=CV_CHAIN_APPROX_SIMPLE, double parameter=0, int minimal_perimeter=0, int recursive=0)

Approximates Freeman chain(s) with polygonal curve
"""

cvStartReadChainPoints.__doc__ = """void cvStartReadChainPoints(CvChain* chain, CvChainPtReader* reader)

Initializes chain reader
"""

cvReadChainPoint.__doc__ = """CvPoint cvReadChainPoint(CvChainPtReader* reader)

Gets next chain point
"""

cvApproxPoly.__doc__ = """CvSeq* cvApproxPoly(const void* src_seq, int header_size, CvMemStorage* storage, int method, double parameter, int parameter2=0)

Approximates polygonal curve(s) with desired precision
"""

cvBoundingRect.__doc__ = """CvRect cvBoundingRect(CvArr* points, int update=0)

Calculates up-right bounding rectangle of point set
"""

cvContourArea.__doc__ = """double cvContourArea(const CvArr* contour, CvSlice slice=CV_WHOLE_SEQ)

Calculates area of the whole contour or contour section
"""

cvArcLength.__doc__ = """double cvArcLength(const void* curve, CvSlice slice=CV_WHOLE_SEQ, int is_closed=-1)

Calculates contour perimeter or curve length
"""

cvCreateContourTree.__doc__ = """CvContourTree* cvCreateContourTree(const CvSeq* contour, CvMemStorage* storage, double threshold)

Creates hierarchical representation of contour
"""

cvContourFromContourTree.__doc__ = """CvSeq* cvContourFromContourTree(const CvContourTree* tree, CvMemStorage* storage, CvTermCriteria criteria)

Restores contour from tree
"""

cvMatchContourTrees.__doc__ = """double cvMatchContourTrees(const CvContourTree* tree1, const CvContourTree* tree2, int method, double threshold)

Compares two contours using their tree representations
"""

cvMaxRect.__doc__ = """CvRect cvMaxRect(const CvRect* rect1, const CvRect* rect2)

Finds bounding rectangle for two given rectangles
"""

cvPointSeqFromMat.__doc__ = """CvSeq* cvPointSeqFromMat(int seq_kind, const CvArr* mat, CvContour* contour_header, CvSeqBlock* block)

Initializes point sequence header from a point vector
"""

cvBoxPoints.__doc__ = """void cvBoxPoints(CvBox2D box, CvPoint2D32f pt[4])

Finds box vertices
"""

cvFitEllipse2.__doc__ = """CvBox2D cvFitEllipse2(const CvArr* points)

Fits ellipse to set of 2D points
"""

cvFitLine.__doc__ = """void cvFitLine(const CvArr* points, int dist_type, double param, double reps, double aeps, float* line)

Fits line to 2D or 3D point set
"""

cvConvexHull2.__doc__ = """CvSeq* cvConvexHull2(const CvArr* input, void* hull_storage=NULL, int orientation=CV_CLOCKWISE, int return_points=0)

Finds convex hull of point set
"""

cvCheckContourConvexity.__doc__ = """int cvCheckContourConvexity(const CvArr* contour)

Tests contour convex
"""

cvConvexityDefects.__doc__ = """CvSeq* cvConvexityDefects(const CvArr* contour, const CvArr* convexhull, CvMemStorage* storage=NULL)

Finds convexity defects of contour
"""

cvPointPolygonTest.__doc__ = """double cvPointPolygonTest(const CvArr* contour, CvPoint2D32f pt, int measure_dist)

Point in contour test
"""

cvMinAreaRect2.__doc__ = """CvBox2D cvMinAreaRect2(const CvArr* points, CvMemStorage* storage=NULL)

Finds circumscribed rectangle of minimal area for given 2D point set
"""

cvMinEnclosingCircle.__doc__ = """int cvMinEnclosingCircle(const CvArr* points, CvPoint2D32f* center, float* radius)

Finds circumscribed circle of minimal area for given 2D point set
"""

cvCalcPGH.__doc__ = """void cvCalcPGH(const CvSeq* contour, CvHistogram* hist)

Calculates pair-wise geometrical histogram for contour
"""

cvSubdivDelaunay2DInsert.__doc__ = """CvSubdiv2DPoint* cvSubdivDelaunay2DInsert(CvSubdiv2D* subdiv, CvPoint2D32f p)

Inserts a single point to Delaunay triangulation
"""

cvSubdiv2DLocate.__doc__ = """CvSubdiv2DPointLocation cvSubdiv2DLocate(CvSubdiv2D* subdiv, CvPoint2D32f pt, CvSubdiv2DEdge* edge, CvSubdiv2DPoint** vertex=NULL)

Inserts a single point to Delaunay triangulation
"""

cvFindNearestPoint2D.__doc__ = """CvSubdiv2DPoint* cvFindNearestPoint2D(CvSubdiv2D* subdiv, CvPoint2D32f pt)

Finds the closest subdivision vertex to given point
"""

cvCalcSubdivVoronoi2D.__doc__ = """void cvCalcSubdivVoronoi2D(CvSubdiv2D* subdiv)

Calculates coordinates of Voronoi diagram cells
"""

cvClearSubdivVoronoi2D.__doc__ = """void cvClearSubdivVoronoi2D(CvSubdiv2D* subdiv)

Removes all virtual points
"""

cvAcc.__doc__ = """void cvAcc(const CvArr* image, CvArr* sum, const CvArr* mask=NULL)

Adds frame to accumulator
"""

cvSquareAcc.__doc__ = """void cvSquareAcc(const CvArr* image, CvArr* sqsum, const CvArr* mask=NULL)

Adds the square of source image to accumulator
"""

cvMultiplyAcc.__doc__ = """void cvMultiplyAcc(const CvArr* image1, const CvArr* image2, CvArr* acc, const CvArr* mask=NULL)

Adds product of two input images to accumulator
"""

cvRunningAvg.__doc__ = """void cvRunningAvg(const CvArr* image, CvArr* acc, double alpha, const CvArr* mask=NULL)

Updates running average
"""

cvUpdateMotionHistory.__doc__ = """void cvUpdateMotionHistory(const CvArr* silhouette, CvArr* mhi, double timestamp, double duration)

Updates motion history image by moving silhouette
"""

cvCalcMotionGradient.__doc__ = """void cvCalcMotionGradient(const CvArr* mhi, CvArr* mask, CvArr* orientation, double delta1, double delta2, int aperture_size=3)

Calculates gradient orientation of motion history image
"""

cvCalcGlobalOrientation.__doc__ = """double cvCalcGlobalOrientation(const CvArr* orientation, const CvArr* mask, const CvArr* mhi, double timestamp, double duration)

Calculates global motion orientation of some selected region
"""

cvSegmentMotion.__doc__ = """CvSeq* cvSegmentMotion(const CvArr* mhi, CvArr* seg_mask, CvMemStorage* storage, double timestamp, double seg_thresh)

Segments whole motion into separate moving parts
"""

cvMeanShift.__doc__ = """int cvMeanShift(const CvArr* prob_image, CvRect window, CvTermCriteria criteria, CvConnectedComp* comp)

Finds object center on back projection
"""

cvCamShift.__doc__ = """int cvCamShift(const CvArr* prob_image, CvRect window, CvTermCriteria criteria, CvConnectedComp* comp, CvBox2D* box=NULL)

Finds object center, size, and orientation
"""

cvSnakeImage.__doc__ = """void cvSnakeImage(const IplImage* image, CvPoint* points, int length, float* alpha, float* beta, float* gamma, int coeff_usage, CvSize win, CvTermCriteria criteria, int calc_gradient=1)

Changes contour position to minimize its energy
"""

cvCalcOpticalFlowHS.__doc__ = """void cvCalcOpticalFlowHS(const CvArr* prev, const CvArr* curr, int use_previous, CvArr* velx, CvArr* vely, double lambda, CvTermCriteria criteria)

Calculates optical flow for two images
"""

cvCalcOpticalFlowLK.__doc__ = """void cvCalcOpticalFlowLK(const CvArr* prev, const CvArr* curr, CvSize win_size, CvArr* velx, CvArr* vely)

Calculates optical flow for two images
"""

cvCalcOpticalFlowBM.__doc__ = """void cvCalcOpticalFlowBM(const CvArr* prev, const CvArr* curr, CvSize block_size, CvSize shift_size, CvSize max_range, int use_previous, CvArr* velx, CvArr* vely)

Calculates optical flow for two images by block matching method
"""

cvCalcOpticalFlowPyrLK.__doc__ = """void cvCalcOpticalFlowPyrLK(const CvArr* prev, const CvArr* curr, CvArr* prev_pyr, CvArr* curr_pyr, const CvPoint2D32f* prev_features, CvPoint2D32f* curr_features, int count, CvSize win_size, int level, char* status, float* track_error, CvTermCriteria criteria, int flags)

Calculates optical flow for a sparse feature set using iterative Lucas-Kanade method in   pyramids
"""

cvCreateKalman.__doc__ = """CvKalman* cvCreateKalman(int dynam_params, int measure_params, int control_params=0)

Allocates Kalman filter structure
"""

cvReleaseKalman.__doc__ = """void cvReleaseKalman(CvKalman** kalman)

Deallocates Kalman filter structure
"""

cvKalmanPredict.__doc__ = """const CvMat* cvKalmanPredict(CvKalman* kalman, const CvMat* control=NULL)

Estimates subsequent model state
"""

cvKalmanCorrect.__doc__ = """const CvMat* cvKalmanCorrect(CvKalman* kalman, const CvMat* measurement)

Adjusts model state
"""

cvCreateConDensation.__doc__ = """CvConDensation* cvCreateConDensation(int dynam_params, int measure_params, int sample_count)

Allocates ConDensation filter structure
"""

cvReleaseConDensation.__doc__ = """void cvReleaseConDensation(CvConDensation** condens)

Deallocates ConDensation filter structure
"""

cvConDensInitSampleSet.__doc__ = """void cvConDensInitSampleSet(CvConDensation* condens, CvMat* lower_bound, CvMat* upper_bound)

Initializes sample set for ConDensation algorithm
"""

cvConDensUpdateByTime.__doc__ = """void cvConDensUpdateByTime(CvConDensation* condens)

Estimates subsequent model state
"""

cvLoadHaarClassifierCascade.__doc__ = """CvHaarClassifierCascade* cvLoadHaarClassifierCascade(const char* directory, CvSize orig_window_size)

Loads a trained cascade classifier from file or the classifier database embedded in OpenCV
"""

cvReleaseHaarClassifierCascade.__doc__ = """void cvReleaseHaarClassifierCascade(CvHaarClassifierCascade** cascade)

Releases haar classifier cascade
"""

cvHaarDetectObjects.__doc__ = """CvSeq* cvHaarDetectObjects(const CvArr* image, CvHaarClassifierCascade* cascade, CvMemStorage* storage, double scale_factor=1.1, int min_neighbors=3, int flags=0, CvSize min_size=cvSize(0, 0)

Detects objects in the image
"""

cvSetImagesForHaarClassifierCascade.__doc__ = """void cvSetImagesForHaarClassifierCascade(CvHaarClassifierCascade* cascade, const CvArr* sum, const CvArr* sqsum, const CvArr* tilted_sum, double scale)

Assigns images to the hidden cascade
"""

cvRunHaarClassifierCascade.__doc__ = """int cvRunHaarClassifierCascade(CvHaarClassifierCascade* cascade, CvPoint pt, int start_stage=0)

Runs cascade of boosted classifier at given image location
"""

cvProjectPoints2.__doc__ = """void cvProjectPoints2(const CvMat* object_points, const CvMat* rotation_vector, const CvMat* translation_vector, const CvMat* intrinsic_matrix, const CvMat* distortion_coeffs, CvMat* image_points, CvMat* dpdrot=NULL, CvMat* dpdt=NULL, CvMat* dpdf=NULL, CvMat* dpdc=NULL, CvMat* dpddist=NULL)

Projects 3D points to image plane
"""

cvFindHomography.__doc__ = """void cvFindHomography(const CvMat* src_points, const CvMat* dst_points, CvMat* homography)

Finds perspective transformation between two planes
"""

cvCalibrateCamera2.__doc__ = """void cvCalibrateCamera2(const CvMat* object_points, const CvMat* image_points, const CvMat* point_counts, CvSize image_size, CvMat* intrinsic_matrix, CvMat* distortion_coeffs, CvMat* rotation_vectors=NULL, CvMat* translation_vectors=NULL, int flags=0)

Finds intrinsic and extrinsic camera parameters using calibration pattern
"""

cvFindExtrinsicCameraParams2.__doc__ = """void cvFindExtrinsicCameraParams2(const CvMat* object_points, const CvMat* image_points, const CvMat* intrinsic_matrix, const CvMat* distortion_coeffs, CvMat* rotation_vector, CvMat* translation_vector)

Finds extrinsic camera parameters for particular view
"""

cvRodrigues2.__doc__ = """int cvRodrigues2(const CvMat* src, CvMat* dst, CvMat* jacobian=0)

Converts rotation matrix to rotation vector or vice versa
"""

cvUndistort2.__doc__ = """void cvUndistort2(const CvArr* src, CvArr* dst, const CvMat* intrinsic_matrix, const CvMat* distortion_coeffs)

Transforms image to compensate lens distortion
"""

cvInitUndistortMap.__doc__ = """void cvInitUndistortMap(const CvMat* intrinsic_matrix, const CvMat* distortion_coeffs, CvArr* mapx, CvArr* mapy)

Computes undistorion map
"""

cvFindChessboardCorners.__doc__ = """int cvFindChessboardCorners(const void* image, CvSize pattern_size, CvPoint2D32f* corners, int* corner_count=NULL, int flags=CV_CALIB_CB_ADAPTIVE_THRESH)

Finds positions of internal corners of the chessboard
"""

cvDrawChessboardCorners.__doc__ = """void cvDrawChessboardCorners(CvArr* image, CvSize pattern_size, CvPoint2D32f* corners, int count, int pattern_was_found)

Renders the detected chessboard corners
"""

cvCreatePOSITObject.__doc__ = """CvPOSITObject* cvCreatePOSITObject(CvPoint3D32f* points, int point_count)

Initializes structure containing object information
"""

cvPOSIT.__doc__ = """void cvPOSIT(CvPOSITObject* posit_object, CvPoint2D32f* image_points, double focal_length, CvTermCriteria criteria, CvMatr32f rotation_matrix, CvVect32f translation_vector)

Implements POSIT algorithm
"""

cvReleasePOSITObject.__doc__ = """void cvReleasePOSITObject(CvPOSITObject** posit_object)

Deallocates 3D object structure
"""

cvCalcImageHomography.__doc__ = """void cvCalcImageHomography(float* line, CvPoint3D32f* center, float* intrinsic, float* homography)

Calculates homography matrix for oblong planar object (e.g. arm)
"""

cvFindFundamentalMat.__doc__ = """int cvFindFundamentalMat(const CvMat* points1, const CvMat* points2, CvMat* fundamental_matrix, int method=CV_FM_RANSAC, double param1=1., double param2=0.99, CvMat* status=NUL)

Calculates fundamental matrix from corresponding points in two images
"""

cvComputeCorrespondEpilines.__doc__ = """void cvComputeCorrespondEpilines(const CvMat* points, int which_image, const CvMat* fundamental_matrix, CvMat* correspondent_line)

For points in one image of stereo pair computes the corresponding epilines in the other image
"""

cvNamedWindow.__doc__ = """int cvNamedWindow(const char* name, int flags)

Creates window
"""

cvDestroyWindow.__doc__ = """void cvDestroyWindow(const char* name)

Destroys a window
"""

cvDestroyAllWindows.__doc__ = """void cvDestroyAllWindows(oi)

Destroys all the HighGUI windows
"""

cvResizeWindow.__doc__ = """void cvResizeWindow(const char* name, int width, int height)

Sets window size
"""

cvMoveWindow.__doc__ = """void cvMoveWindow(const char* name, int x, int y)

Sets window position
"""

cvGetWindowHandle.__doc__ = """void* cvGetWindowHandle(const char* name)

Gets window handle by name
"""

cvGetWindowName.__doc__ = """constchar* cvGetWindowName(void* window_handle)

Gets window name by handle
"""

cvShowImage.__doc__ = """void cvShowImage(const char* name, const CvArr* image)

Shows the image in the specified window
"""

cvGetTrackbarPos.__doc__ = """int cvGetTrackbarPos(const char* trackbar_name, const char* window_name)

Retrieves trackbar position
"""

cvSetTrackbarPos.__doc__ = """void cvSetTrackbarPos(const char* trackbar_name, const char* window_name, int pos)

Sets trackbar position
"""

cvWaitKey.__doc__ = """int cvWaitKey(int delay=0)

Waits for a pressed key
"""

cvLoadImage.__doc__ = """IplImage* cvLoadImage(const char* filename, int iscolor=1)

Loads an image from file
"""

cvSaveImage.__doc__ = """int cvSaveImage(const char* filename, const CvArr* image)

Saves an image to the file
"""

cvCreateFileCapture.__doc__ = """CvCapture* cvCreateFileCapture(const char* filename)

Initializes capturing video from file
"""

cvCreateCameraCapture.__doc__ = """CvCapture* cvCreateCameraCapture(int index)

Initializes capturing video from camera
"""

cvReleaseCapture.__doc__ = """void cvReleaseCapture(CvCapture** capture)

Releases the CvCapture structure
"""

cvGrabFrame.__doc__ = """int cvGrabFrame(CvCapture* capture)

Grabs frame from camera or file
"""

cvRetrieveFrame.__doc__ = """IplImage* cvRetrieveFrame(CvCapture* capture)

Gets the image grabbed with cvGrabFrame
"""

cvQueryFrame.__doc__ = """IplImage* cvQueryFrame(CvCapture* capture)

Grabs and returns a frame from camera or file
"""

cvGetCaptureProperty.__doc__ = """double cvGetCaptureProperty(CvCapture* capture, int property_id)

Gets video capturing properties
"""

cvSetCaptureProperty.__doc__ = """int cvSetCaptureProperty(CvCapture* capture, int property_id, double value)

Sets video capturing properties
"""

cvCreateVideoWriter.__doc__ = """CvVideoWriter* cvCreateVideoWriter(const char* filename, int fourcc, double fps, CvSize frame_size, int is_color=1)

Creates video file writer
"""

cvReleaseVideoWriter.__doc__ = """void cvReleaseVideoWriter(CvVideoWriter** writer)

Releases AVI writer
"""

cvWriteFrame.__doc__ = """int cvWriteFrame(CvVideoWriter* writer, const IplImage* image)

Writes a frame to video file
"""

cvInitSystem.__doc__ = """int cvInitSystem(int argc, char** argv)

Initializes HighGUI
"""

cvConvertImage.__doc__ = """void cvConvertImage(const CvArr* src, CvArr* dst, int flags=0)

Converts one image to another with optional vertical flip
"""

# --- SOME FUNCTION COPIES FROM THE C HEADERS (reverse compatibility?) ---
cvGetSubArr = cvGetSubRect
cvZero = cvSetZero
cvCvtScale = cvConvertScale
cvScale = cvConvertScale
cvCvtScaleAbs = cvConvertScaleAbs
cvCheckArray = cvCheckArr
cvMatMulAddEx = cvGEMM
cvMatMulAddS = cvTransform
cvT = cvTranspose
cvMirror = cvFlip
cvInv = cvInvert
cvMahalonobis = cvMahalanobis
cvFFT = cvDFT
cvGraphFindEdge = cvFindGraphEdge
cvGraphFindEdgeByPtr = cvFindGraphEdgeByPtr
cvDrawRect = cvRectangle
cvDrawLine = cvLine
cvDrawCircle = cvCircle
cvDrawEllipse = cvEllipse
cvDrawPolyLine = cvPolyLine

# wrapup all the functions into a single object so we can say
# from OpenCV import cv
# cv.Foo instead of OpenCV.cvFoo
class namespace:
    pass
nsp = namespace()

mdict = locals()
for sym, val in mdict.items():
    if sym.startswith('CV_'):
        sname = sym[3:]
        if sname == 'SVD':
            sname = '_SVD'
    elif sym.startswith('cv'):
        sname = sym[2:]
    elif sym.startswith('Cv'):
        sname = sym[2:]
    else:
        continue
    if not hasattr(nsp, sname):
        setattr(nsp, sname, val)
    else:
        print 'name collision', sname, getattr(nsp, sname)

cv = nsp

# --- Hauptprogramm ----------------------------------------------------------

if __name__ == "__main__":

    print __doc__
