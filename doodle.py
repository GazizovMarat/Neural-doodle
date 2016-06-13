#!/usr/bin/env python3
"""                         _       _                 _ _       
 _ __   ___ _   _ _ __ __ _| |   __| | ___   ___   __| | | ___  
| '_ \ / _ \ | | | '__/ _` | |  / _` |/ _ \ / _ \ / _` | |/ _ \ 
| | | |  __/ |_| | | | (_| | | | (_| | (_) | (_) | (_| | |  __/ 
|_| |_|\___|\__,_|_|  \__,_|_|  \__,_|\___/ \___/ \__,_|_|\___| 

"""
#
# Copyright (c) 2016, Alex J. Champandard.
#
# Neural Doodle is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General
# Public License version 3. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#
# Research and Development sponsored by the nucl.ai Conference!
#   http://events.nucl.ai/
#   July 18-20, 2016 in Vienna/Austria.
#

import os
import sys
import math
import time
import pickle
import argparse
import itertools
import collections


# Configure all options first so we can later custom-load other libraries (Theano) based on device specified by user.
parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg('--content',         default=None, type=str,         help='Subject image path to repaint in new style.')
add_arg('--style',           default=None, type=str,         help='Texture image path to extract patches from.')
add_arg('--passes',          default=2, type=int,            help='Number of times to go over the whole image.')
add_arg('--variety',         default=[.2,.1,.0], nargs='+', type=float, help='Bias selecting diverse patches')
add_arg('--layers',          default=[5, 4, 3], nargs='+',  type=int,   help='The layers/scales to process.')
add_arg('--layer-weight',    default=[1.0], nargs='+',      type=float, help='Weight of previous layer features.')
add_arg('--content-weight',  default=[.3,.2,.1], nargs='+', type=float, help='Weight of input content features each layer.')
add_arg('--noise-weight',    default=[.2,.1,.0], nargs='+', type=float, help='Weight of noise added into features.')
add_arg('--iterations',      default=[4, 4, 1], nargs='+',  type=int,   help='Number of times to repeat layer optimization.')
add_arg('--shapes',          default=[3], nargs='+', type=int,     help='Size of kernels used for patch extraction.')
add_arg('--seed',            default=None, type=int,         help='Initial state for the random number generator.')
add_arg('--semantic-ext',    default='_sem.png', type=str,   help='File extension for the semantic maps.')
add_arg('--semantic-weight', default=0.0, type=float,        help='Global weight of semantics vs. style features.')
add_arg('--output',          default='output.png', type=str, help='Filename or path to save output once done.')
add_arg('--output-size',     default=None, type=str,         help='Size of the output image, e.g. 512x512.')
add_arg('--frames',          default=False, action='store_true',   help='Render intermediate frames, takes more time.')
add_arg('--device',          default='cpu', type=str,        help='Index of the GPU number to use, for theano.')
add_arg('--model',           default='gelu3', type=str,      help='Filename for convolution weights of neural network.')
args = parser.parse_args()


#----------------------------------------------------------------------------------------------------------------------

# Color coded output helps visualize the information a little better, plus it looks cool!
class ansi:
    BOLD = '\033[1;97m'
    WHITE = '\033[0;97m'
    YELLOW = '\033[0;33m'
    YELLOW_B = '\033[1;33m'
    RED = '\033[0;31m'
    RED_B = '\033[1;31m'
    BLUE = '\033[0;94m'
    BLUE_B = '\033[1;94m'
    CYAN = '\033[0;36m'
    CYAN_B = '\033[1;36m'
    ENDC = '\033[0m'

def error(message, *lines):
    string = "\n{}ERROR: " + message + "{}\n" + "\n".join(lines) + "{}\n"
    print(string.format(ansi.RED_B, ansi.RED, ansi.ENDC))
    sys.exit(-1)

def extend(lst): return itertools.chain(lst, itertools.repeat(lst[-1]))
def snap(value, grid=2**(args.layers[0]-1)): return int(grid * math.floor(value / grid))

print("""{}   {}High-quality image synthesis powered by Deep Learning!{}
  - Code licensed as AGPLv3, models under CC BY-NC-SA.{}""".format(ansi.CYAN_B, __doc__, ansi.CYAN, ansi.ENDC))

# Load the underlying deep learning libraries based on the device specified.  If you specify THEANO_FLAGS manually,
# the code assumes you know what you are doing and they are not overriden!
os.environ.setdefault('THEANO_FLAGS', 'floatX=float32,device={},force_device=True,allow_gc=True,'\
                                      'print_active_device=False'.format(args.device))

# Scientific & Imaging Libraries
import numpy as np
import scipy.optimize, scipy.ndimage, scipy.misc
import numba
import PIL.ImageOps
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

# Numeric Computing (GPU)
import theano
import theano.tensor as T
import theano.tensor.nnet.neighbours

# Support ansi colors in Windows too.
if sys.platform == 'win32':
    import colorama

# Deep Learning Framework
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer, Deconv2DLayer as DeconvLayer, Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, ConcatLayer

print('{}  - Using the device `{}` for tensor computation.{}'.format(ansi.CYAN, theano.config.device, ansi.ENDC))


#----------------------------------------------------------------------------------------------------------------------
# Convolutional Neural Network
#----------------------------------------------------------------------------------------------------------------------
class Model(object):
    """Store all the data related to the neural network (aka. "model"). This is currently based on VGG19.
    """

    def __init__(self):
        self.setup_model()
        self.load_data()

    def setup_model(self, previous=None):
        """Use lasagne to create a network of convolution layers, first using VGG19 as the framework
        and then adding augmentations for Semantic Style Transfer.
        """
        net, self.channels = {}, {}
        self.units = {1: 48, 2: 80, 3: 128, 4: 208, 5: 328, 6: 536}

        net['map'] = InputLayer((1, None, None, None))
        for j in range(6):
            net['map%i'%(j+1)] = PoolLayer(net['map'], 2**j, mode='average_exc_pad')

        self.tensor_latent = []
        for l in args.layers:
            self.tensor_latent.append((str(l), T.tensor4()))
            net['lat%i'%l] = InputLayer((None, self.units[l], None, None), var=self.tensor_latent[-1][1])

        def EncdLayer(previous, channels, filter_size, pad, stride=(1,1), nonlinearity=lasagne.nonlinearities.elu):
            incoming = net['lat'+previous[0]] if int(previous[0]) in args.layers and previous[1:] == '_1' else net['enc'+previous]
            return ConvLayer(incoming, channels, filter_size, pad=pad, stride=stride, nonlinearity=nonlinearity)

        # Encoder part of the neural network, takes an input image and turns it into abstract patterns.
        net['img']    = previous or InputLayer((None, 3, None, None))
        net['enc0_0'], net['lat0'] = net['img'], net['img']
        net['enc1_1'] = EncdLayer('0_0',  48, 3, pad=1)
        net['enc1_2'] = EncdLayer('1_1',  48, 3, pad=1)
        net['enc2_1'] = EncdLayer('1_2',  80, 2, pad=0, stride=(2,2))
        net['enc2_2'] = EncdLayer('2_1',  80, 3, pad=1)
        net['enc3_1'] = EncdLayer('2_2', 128, 2, pad=0, stride=(2,2))
        net['enc3_2'] = EncdLayer('3_1', 128, 3, pad=1)
        net['enc3_3'] = EncdLayer('3_2', 128, 3, pad=1)
        net['enc4_1'] = EncdLayer('3_3', 208, 2, pad=0, stride=(2,2))
        net['enc4_2'] = EncdLayer('4_1', 208, 3, pad=1)
        net['enc4_3'] = EncdLayer('4_2', 208, 3, pad=1)
        net['enc5_1'] = EncdLayer('4_3', 328, 2, pad=0, stride=(2,2))
        net['enc5_2'] = EncdLayer('5_1', 328, 3, pad=1)
        net['enc5_3'] = EncdLayer('5_2', 328, 3, pad=1)
        net['enc6_1'] = EncdLayer('5_3', 536, 2, pad=0, stride=(2,2))

        def DecdLayer(copy, previous, channels, nonlinearity=lasagne.nonlinearities.elu):
            # Dynamically injects intermediate "pitstop" output layers in the decoder based on what the user specified as layers.
            dup, incoming = net['enc'+copy], net['lat'+copy[0]] if int(copy[0]) in args.layers and copy[1:] == '_1' else net[previous]
            return DeconvLayer(incoming, channels, dup.filter_size, stride=dup.stride, crop=dup.pad, nonlinearity=nonlinearity)

        # Decoder part of the neural network, takes abstract patterns and converts them into an image!
        net['dec5_3'] = DecdLayer('6_1', 'enc6_1', 328)
        net['dec5_2'] = DecdLayer('5_3', 'dec5_3', 328)
        net['dec5_1'] = DecdLayer('5_2', 'dec5_2', 328)
        net['dec4_3'] = DecdLayer('5_1', 'dec5_1', 208)
        net['dec4_2'] = DecdLayer('4_3', 'dec4_3', 208)
        net['dec4_1'] = DecdLayer('4_2', 'dec4_2', 208)
        net['dec3_3'] = DecdLayer('4_1', 'dec4_1', 128)
        net['dec3_2'] = DecdLayer('3_3', 'dec3_3', 128)
        net['dec3_1'] = DecdLayer('3_2', 'dec3_2', 128)
        net['dec2_2'] = DecdLayer('3_1', 'dec3_1',  80)
        net['dec2_1'] = DecdLayer('2_2', 'dec2_2',  80)
        net['dec1_2'] = DecdLayer('2_1', 'dec2_1',  48)
        net['dec1_1'] = DecdLayer('1_2', 'dec1_2',  48)
        net['dec0_1'] = DecdLayer('1_1', 'dec1_1',   3, nonlinearity=lasagne.nonlinearities.tanh)
        net['dec0_0'] = lasagne.layers.ScaleLayer(net['dec0_1'], shared_axes=(0,1,2,3))
        net['out']    = lasagne.layers.NonlinearityLayer(net['dec0_0'], nonlinearity=lambda x: T.clip(127.5*(x+1.0), 0.0, 255.0))

        def ConcatenateLayer(incoming, layer):
            # TODO: The model is constructed too soon, we don't yet know if semantic_weight is needed. Fails if not.
            return ConcatLayer([incoming, net['map%i'%layer]]) if args.semantic_weight > 0.0 else incoming

        # Auxiliary network for the semantic layers, and the nearest neighbors calculations.
        self.pm_inputs, self.pm_buffers, self.pm_candidates = {}, {}, {}
        for layer, upper, lower in zip(args.layers, [None] + args.layers[:-1], args.layers[1:] + [None]):
            self.channels[layer] = net['enc%i_1'%layer].num_filters
            net['sem%i'%layer] = ConcatenateLayer(net['enc%i_1'%layer], layer)
            self.pm_inputs[layer] = T.ftensor4()
            self.pm_buffers[layer] = T.ftensor4()
            self.pm_candidates[layer] = T.itensor4()
        self.network = net

    def load_data(self):
        """Open the serialized parameters from a pre-trained network, and load them into the model created.
        """
        data_file = os.path.join(os.path.dirname(__file__), '{}_conv.pkl'.format(args.model))
        if not os.path.exists(data_file):
            error("Model file with pre-trained convolution layers not found. Download from here...",
                  "https://github.com/alexjc/neural-doodle/releases/download/v0.0/{}_conv.pkl".format(args.model))

        data = pickle.load(open(data_file, 'rb'))
        for layer, values in data.items():
            if layer not in self.network: continue
            for p, v in zip(self.network[layer].get_params(), values):
                assert p.get_value().shape == v.shape, "Layer `{}` in network has size {} but data is {}."\
                                                       .format(layer, p.get_value().shape, v.shape)
                p.set_value(v.astype(np.float32))

    def setup(self, layers):
        """Setup the inputs and outputs, knowing the layers that are required by the optimization algorithm.
        """
        self.tensor_img = T.tensor4()
        self.tensor_map = T.tensor4()
        tensor_inputs = {self.network['img']: self.tensor_img, self.network['map']: self.tensor_map}
        outputs = lasagne.layers.get_output([self.network[l] for l in layers], tensor_inputs)
        self.tensor_outputs = {k: v for k, v in zip(layers, outputs)}

    def get_outputs(self, type, layers):
        """Fetch the output tensors for the network layers.
        """
        return [self.tensor_outputs[type+l] for l in layers]

    def prepare_image(self, image):
        """Given an image loaded from disk, turn it into a representation compatible with the model. The format is
        (b,c,y,x) with batch=1 for a single image, channels=3 for RGB, and y,x matching the resolution.
        """
        image = np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)[::-1, :, :]
        image = image.astype(np.float32) / 127.5 - 1.0
        return image[np.newaxis]

    def finalize_image(self, image, resolution):
        """Convert network output into an image format that can be saved to disk, shuffling dimensions as appropriate.
        """
        image = np.swapaxes(np.swapaxes(image[::-1], 0, 1), 1, 2)
        image = np.clip(image, 0, 255).astype('uint8')
        return scipy.misc.imresize(image, resolution, interp='bicubic')



@numba.guvectorize([(numba.float32[:,:,:,:], numba.float32[:,:,:,:], numba.int32[:,:,:,:], numba.int32[:,:], numba.float32[:,:])],
                    '(n,c,x,y),(n,c,z,w),(a,b,m,i)->(a,b),(a,b)', nopython=True, target='parallel')
def compute_matches(inputs, buffers, indices, selected, best):
    for a in range(indices.shape[0]):
        for b in range(indices.shape[1]):
            scores = np.zeros((indices.shape[2],), dtype=np.float32)
            for m in range(indices.shape[2]):
                i0, i1, i2 = indices[a,b,m,0], indices[a,b,m,1], indices[a,b,m,2]
                for y, x in [(-1,-1),(-1,0),(-1,+1),(0,-1),(0,0),(0,+1),(+1,-1),(+1,0),(+1,+1)]:
                    candidates = buffers[i0,:,i1+y,i2+x]
                    reference = inputs[0,:,1+a+y,1+b+x]
                    scores[m] += np.sum(candidates * reference)
            selected[a,b] = scores.argmax()
            best[a,b] = scores.max()


#----------------------------------------------------------------------------------------------------------------------
# Semantic Style Transfer
#----------------------------------------------------------------------------------------------------------------------
class NeuralGenerator(object):
    """This is the main part of the application that generates an image using optimization and LBFGS.
    The images will be processed at increasing resolutions in the run() method.
    """

    def __init__(self):
        """Constructor sets up global variables, loads and validates files, then builds the model.
        """
        self.start_time = time.time()
        np.random.seed(args.seed)

        # Prepare file output and load files specified as input.
        if args.frames is not False:
            os.makedirs('frames', exist_ok=True)
        if args.output is not None and os.path.isfile(args.output):
            os.remove(args.output)

        # Finalize the parameters based on what we loaded, then create the model.
        args.semantic_weight = math.sqrt(9.0 / args.semantic_weight) if args.semantic_weight else 0.0
        self.model = Model()


    #------------------------------------------------------------------------------------------------------------------
    # Helper Functions
    #------------------------------------------------------------------------------------------------------------------

    def rescale_image(self, img, scale):
        """Re-implementing skimage.transform.scale without the extra dependency. Saves a lot of space and hassle!
        """
        output = scipy.misc.toimage(img, cmin=0.0, cmax=255)
        return np.asarray(PIL.ImageOps.fit(output, [snap(dim*scale) for dim in output.size], PIL.Image.ANTIALIAS))

    def load_images(self, name, filename, scale=1.0):
        """If the image and map files exist, load them. Otherwise they'll be set to default values later.
        """
        basename, _ = os.path.splitext(filename)
        mapname = basename + args.semantic_ext
        img = scipy.ndimage.imread(filename, mode='RGB') if os.path.exists(filename) else None
        map = scipy.ndimage.imread(mapname) if os.path.exists(mapname) and args.semantic_weight > 0.0 else None

        shp = img.shape if img is not None else (map.shape if map is not None else '??')
        print('\n{}{} {}x{}{} at scale {:3.1f}'.format(ansi.BLUE_B, name.capitalize(), shp[1], shp[0], ansi.BLUE, 1.0))
        if img is not None: print('  - Loading `{}` for {} data.'.format(filename, name))
        if map is not None: print('  - Adding `{}` as semantic map.'.format(mapname))

        if img is not None and map is not None and img.shape[:2] != map.shape[:2]:
            error("The {} image and its semantic map have different resolutions. Either:".format(name),
                  "  - Resize {} to {}, or\n  - Resize {} to {}."\
                  .format(filename, map.shape[1::-1], mapname, img.shape[1::-1]))
        return [(self.rescale_image(i, scale) if i is not None else None) for i in [img, map]]

    def compile(self, arguments, function, **opts):
        """Build a Theano function that will run the specified expression on the GPU.
        """
        return theano.function(list(arguments), function, on_unused_input='ignore', allow_input_downcast=True, **opts)

    def compute_norms(self, backend, layer, array):
        ni = backend.sqrt(backend.sum(array[:,:self.model.channels[layer]] ** 2.0, axis=(1,), keepdims=True))
        ns = backend.sqrt(backend.sum(array[:,self.model.channels[layer]:] ** 2.0, axis=(1,), keepdims=True))
        return [ni, ns]

    def normalize_components(self, layer, array, norms):
        if args.semantic_weight > 0.0:
            print(layer, self.model.channels, len(norms))
            array[:,self.model.channels[layer]:] /= (norms[1] * args.semantic_weight)
        array[:,:self.model.channels[layer]] /= (norms[0] * 3.0)


    #------------------------------------------------------------------------------------------------------------------
    # Initialization & Setup
    #------------------------------------------------------------------------------------------------------------------

    def prepare_style(self, scale=1.0):
        """Called each phase of the optimization, process the style image according to the scale, then run it
        through the model to extract intermediate outputs (e.g. sem4_1) and turn them into patches.
        """
        style_img_original, style_map_original = self.load_images('style', args.style, scale)

        if style_map_original is None:
            style_map_original = np.zeros(style_img_original.shape[:2]+(2,)) - 1.0
            args.semantic_weight = 0.0

        if style_img_original is None:
            error("Couldn't find style image as expected.",
                  "  - Try making sure `{}` exists and is a valid image.".format(args.style))

        self.style_img = self.model.prepare_image(style_img_original)
        self.style_map = style_map_original.transpose((2, 0, 1))[np.newaxis].astype(np.float32)

        input_tensors = self.model.tensor_latent[1:] + [('0', self.model.tensor_img)]
        self.encoders = []
        for layer, (input, tensor_latent), shape in zip(args.layers, input_tensors, extend(args.shapes)):
            output = lasagne.layers.get_output(self.model.network['sem%i'%layer],
                                              {self.model.network['lat'+input]: tensor_latent,
                                               self.model.network['map']: self.model.tensor_map})
            fn = self.compile([tensor_latent, self.model.tensor_map], [output] + self.do_extract_patches(layer, output))
            self.encoders.append(fn)

        # Store all the style patches layer by layer, resized to match slice size and cast to 16-bit for size. 
        self.style_data, feature = {}, self.style_img
        for layer, encoder in reversed(list(zip(args.layers, self.encoders))):
            feature, *data = encoder(feature, self.style_map)
            feature, patches = feature[:,:self.model.channels[layer]], data[0]
            self.style_data[layer] = [d.astype(np.float16) for d in data]\
                                   + [np.zeros((patches.shape[0],), dtype=np.float16), -1]
            print('  - Layer {} as {} patches {} in {:,}kb.'.format(layer, patches.shape[:2], patches.shape[2:], patches.size//1000))

    def prepare_content(self, scale=1.0):
        """Called each phase of the optimization, rescale the original content image and its map to use as inputs.
        """
        content_img_original, content_map_original = self.load_images('content', args.content or args.output, scale)

        if content_map_original is not None and self.style_map is None:
            basename, _ = os.path.splitext(args.style)
            error("Expecting a semantic map for the input style image too.",
                  "  - Try creating the file `{}_sem.png` with your annotations.".format(basename))

        if self.style_map.max() >= 0.0 and content_map_original is None:
            basename, _ = os.path.splitext(args.content or args.output)
            error("Expecting a semantic map for the input content image too.",
                  "  - Try creating the file `{}_sem.png` with your annotations.".format(basename))

        if content_map_original is None:
            if content_img_original is None and args.output_size:
                shape = tuple([int(i) for i in args.output_size.split('x')])
            else:
                if content_img_original is None:
                    shape = self.style_img.shape[2:]
                else:
                    shape = content_img_original.shape[:2]

            content_map_original = np.zeros(shape+(2,))
            args.semantic_weight = 0.0

        if content_img_original is None:
            print("  - No content image found; seed was set to random noise.")
            content_img_original = np.random.uniform(0, 256, content_map_original.shape[:2]+(3,)).astype(np.float32)

        if content_map_original.shape[2] != self.style_map.shape[1]:
            error("Mismatch in number of channels for style and content semantic map.",
                  "  - Make sure both images are RGB, RGBA, or L.")

        self.content_img = self.model.prepare_image(content_img_original)
        self.content_map = content_map_original.transpose((2, 0, 1))[np.newaxis].astype(np.float32)
        self.content_shape = content_img_original.shape

        # Feed-forward calculation only, returns the result of the convolution post-activation
        self.content_features, feature = [], self.content_img
        for layer, encoder in reversed(list(zip(args.layers, self.encoders))):
            feature, *_ = encoder(feature, self.content_map)
            feature = feature[:,:self.model.channels[layer]]
            style = self.style_data[layer][0]

            sn, sx = style.min(axis=(0,2,3), keepdims=True), style.max(axis=(0,2,3), keepdims=True)
            cn, cx = feature.min(axis=(0,2,3), keepdims=True), feature.max(axis=(0,2,3), keepdims=True)

            self.content_features.insert(0, sn + (feature - cn) * (sx - sn) / (cx - cn))
            print("  - Layer {} as {} array in {:,}kb.".format(layer, feature.shape[1:], feature.size//1000))

    def prepare_generation(self):
        """Layerwise synthesis images requires two sets of Theano functions to be compiled.
        """
        # Patch matching calculation that uses only pre-calculated features and a slice of the patches.
        # self.compute_matches = {l: self.compile([self.model.pm_inputs[l], self.model.pm_buffers[l], self.model.pm_candidates[l]], self.do_match_patches(l)) for l in args.layers}
        self.pm_previous = {}
        LayerInput = collections.namedtuple('LayerInput', ['array', 'weight'])
        self.layer_inputs = [[LayerInput(np.copy(self.content_features[i]), w) for _, w in zip(args.layers, extend(args.layer_weight))]
                                                                               for i, _ in enumerate(args.layers)]

        # Decoding intermediate features into more specialized features and all the way to the output image.
        self.encoders, input_tensors = [], self.model.tensor_latent[1:] + [('0', self.model.tensor_img)]
        for name, (input, tensor_latent) in zip(args.layers, input_tensors):
            layer = lasagne.layers.get_output(self.model.network['enc%i_1'%name],
                                             {self.model.network['lat'+input]: tensor_latent,
                                              self.model.network['map']: self.model.tensor_map})
            fn = self.compile([tensor_latent, self.model.tensor_map], layer)
            self.encoders.append(fn)

        self.decoders, output_layers = [], (['dec%i_1'%l for l in args.layers[1:]] + ['out'])
        for name, (input, tensor_latent), output in zip(args.layers, self.model.tensor_latent, output_layers):
            layer = lasagne.layers.get_output(self.model.network[output],
                                             {self.model.network['lat'+input]: tensor_latent,
                                              self.model.network['map']: self.model.tensor_map})
            fn = self.compile([tensor_latent, self.model.tensor_map], layer)
            self.decoders.append(fn)


    #------------------------------------------------------------------------------------------------------------------
    # Theano Computation
    #------------------------------------------------------------------------------------------------------------------

    def do_extract_patches(self, layer, output):
        """This function builds a Theano expression that will get compiled an run on the GPU. It extracts 3x3 patches
        from the intermediate outputs in the model.
        """
        return [output] + self.compute_norms(T, layer, output)


    #------------------------------------------------------------------------------------------------------------------
    # Optimization Loop
    #------------------------------------------------------------------------------------------------------------------

    def evaluate_patches(self, l, f, v):
        buffers = self.style_data[l][0].astype(np.float32)
        self.normalize_components(l, buffers, self.style_data[l][1:3])
        self.normalize_components(l, f, self.compute_norms(np, l, f))

        SAMPLES = 24
        indices = np.zeros((f.shape[2]-2, f.shape[3]-2, SAMPLES, 3), dtype=np.int32) # TODO: patchsize

        ref_idx, ref_val = self.pm_previous.get(l, (None, 0.0))
        for i in itertools.count():
            indices[:,:,:,1] = np.random.randint(low=1, high=buffers.shape[2]-1, size=indices.shape[:3]) # TODO: patchsize
            indices[:,:,:,2] = np.random.randint(low=1, high=buffers.shape[3]-1, size=indices.shape[:3]) # TODO: patchsize

            if ref_idx is not None:
                indices[:,:,0,:] = ref_idx + (0,0,0)
                indices[:,:,1,:] = ref_idx + (0,0,+1)
                indices[:,:,2,:] = ref_idx + (0,+1,0)
                indices[:,:,3,:] = ref_idx + (0,0,-1)
                indices[:,:,4,:] = ref_idx + (0,-1,0)

                indices[:-1,:,5,:] = ref_idx[+1:,:] + (0,-1,0)
                indices[+1:,:,6,:] = ref_idx[:-1,:] + (0,+1,0)
                indices[:,:-1,7,:] = ref_idx[:,+1:] + (0,0,-1)
                indices[:,+1:,8,:] = ref_idx[:,:-1] + (0,0,+1)

                indices[:,:,:9,1].clip(1, buffers.shape[2]-2, out=indices[:,:,:9,1])
                indices[:,:,:9,2].clip(1, buffers.shape[3]-2, out=indices[:,:,:9,2])
            
            previous = self.pm_previous.get(l+1, None)
            if i == 0 and previous is not None:
                prev_idx, ref_val = previous
                resh_idx = np.concatenate([scipy.ndimage.zoom(np.pad(prev_idx[:,:,i]*2, 1, mode='reflect'), 2, order=1)[:,:,np.newaxis] for i in range(1, 3)], axis=(2))
                indices[:,:,10,1:] = resh_idx[+1:-1,+1:-1]
                indices[:,:,11,1:] = resh_idx[+2:  ,+2:  ]
                indices[:,:,12,1:] = resh_idx[+2:  ,  :-2]
                indices[:,:,13,1:] = resh_idx[  :-2,+2:  ]
                indices[:,:,14,1:] = resh_idx[  :-2,  :-2]
                
                indices[:,:,10:15,1].clip(1, buffers.shape[2]-2, out=indices[:,:,10:15,1])
                indices[:,:,10:15,2].clip(1, buffers.shape[3]-2, out=indices[:,:,10:15,2])
            
            t = time.time()
            best_idx, best_val = compute_matches(f, buffers, indices)
            print('patch_match', time.time() - t)

            # Numpy array indexing rules seem to require injecting the identity matrix back into the array.
            identity = np.indices(f.shape[2:]).transpose((1,2,0))[:-2,:-2] + 1 # TODO: patchsize
            accessors = np.concatenate([identity - 1, best_idx[:,:,np.newaxis]], axis=2)
            ref_idx = indices[accessors[:,:,0],accessors[:,:,1],accessors[:,:,2]]

            rv = np.percentile(best_val, 5.0)
            print('values', ref_idx.shape, (rv - ref_val) / rv, rv - ref_val, rv)
            if abs(rv - ref_val) < 0.00005:
                break
            ref_val = rv

        self.pm_previous[l] = (ref_idx, ref_val)
        return ref_idx, best_val

    def evaluate_feature(self, layer, feature, variety=0.0):
        """Compute best matching patches for this layer, then merge patches into a single feature array of same size.
        """
        iter_time = time.time()
        B, indices = self.style_data[layer][0][:,:,:,:,np.newaxis,np.newaxis].astype(np.float32), self.style_data[layer][-1]
        best_idx, best_val = self.evaluate_patches(layer, feature, variety)
        # better_patches = buffers[best_idx[:,:,0],:,best_idx[:,:,1],best_idx[:,:,2]]
        i0, i1, i2 = best_idx[:,:,0], best_idx[:,:,1], best_idx[:,:,2]

        better_patches = np.concatenate([np.concatenate([B[i0,:,i1-1,i2-1], B[i0,:,i1-1,i2+0], B[i0,:,i1-1,i2+1]], axis=4),
                                         np.concatenate([B[i0,:,i1+0,i2-1], B[i0,:,i1+0,i2+0], B[i0,:,i1+0,i2+1]], axis=4),
                                         np.concatenate([B[i0,:,i1+1,i2-1], B[i0,:,i1+1,i2+0], B[i0,:,i1+1,i2+1]], axis=4)], axis=3)

        better_patches = better_patches.reshape((-1,)+better_patches.shape[2:]).transpose((0,2,3,1))
        better_shape = feature.shape[2:] + (feature.shape[1],)
        better_feature = reconstruct_from_patches_2d(better_patches, better_shape)

        # used = 99. * len(set(best_idx)) / best_idx.shape[0]
        # duplicates = 99. * len([v for v in np.bincount(best_idx) if v>1]) / best_idx.shape[0]
        # changed = 99. * (1.0 - np.where(indices == best_idx)[0].shape[0] / best_idx.shape[0])
        used, duplicates, changed = -1.0, -2.0, -3.0

        err = best_val.mean()
        print('  {}layer{} {:>1}   {}patches{}  used {:2.0f}%  dups {:2.0f}%  chgd {:2.0f}%   {}error{} {:3.2e}   {}time{} {:3.1f}s'\
             .format(ansi.BOLD, ansi.ENDC, layer, ansi.BOLD, ansi.ENDC, used, duplicates, changed,
                     ansi.BOLD, ansi.ENDC, err, ansi.BOLD, ansi.ENDC, time.time() - iter_time))
                     
        # self.style_data[layer][-1] = best_idx
        return better_feature.astype(np.float32).transpose((2, 0, 1))[np.newaxis]

    def evaluate_features(self):
        params = zip(*[extend(a) for a in [args.content_weight, args.noise_weight, args.variety, args.iterations]])
        
        for i, (l, c, p) in enumerate(zip(args.layers, self.content_features, params)):
            content_weight, noise_weight, variety, iterations = p
            for j in range(iterations):
                blended = sum([a*w for a, w in self.layer_inputs[i]]) / sum([w for _, w in self.layer_inputs[i]])
                self.render(blended, l, 'blended-L{}I{}'.format(l, j+1))
                feature = blended * (1.0 - content_weight) + c * content_weight \
                        + np.random.normal(0.0, 1.0, size=c.shape).astype(np.float32) * (0.1 * noise_weight)
                self.render(feature, l, 'mixed-L{}I{}'.format(l, j+1))
                result = self.evaluate_feature(l, feature, variety)
                self.render(result, l, 'output-L{}I{}'.format(l, j+1))
                self.layer_inputs[i][i].array[:] = result 

            if i+1 < len(args.layers):
                for j in range(0, i+1):
                    self.layer_inputs[i+1][j].array[:] = self.decoders[i](self.layer_inputs[i][j].array, self.content_map)

        for i in range(len(args.layers)-1, 0, -1):
            for j in range(i, len(args.layers)):
                self.layer_inputs[i-1][j].array[:] = self.encoders[i-1](self.layer_inputs[i][j].array, self.content_map)

    def evaluate(self, Xn):
        """Feed-forward evaluation of the output based on current image. Can be called multiple times.
        """
        self.frame = 0
        for i, c in zip(args.layers, self.content_features):
            self.render(c, i, 'orig-L{}'.format(i))

        for j in range(args.passes):
            self.frame += 1
            print('\n{}Pass #{}{}: variety {}, weights {}.{}'.format(ansi.CYAN_B, self.frame, ansi.CYAN, 0.0, 0.0, ansi.ENDC))
            self.evaluate_features()

        return self.decoders[-1](self.layer_inputs[-1][-1].array, self.content_map)

    def render(self, features, layer, suffix):
        """Decode features at a specific layer and save the result to disk for visualization. (Takes 50% more time.) 
        """
        if not args.frames: return
        for l, compute in list(zip(args.layers, self.decoders))[args.layers.index(layer):]:
            features = compute(features[:,:self.model.channels[l]], self.content_map)

        output = self.model.finalize_image(features.reshape(self.content_img.shape[1:]), self.content_shape)
        filename = os.path.splitext(os.path.basename(args.output))[0]
        scipy.misc.toimage(output, cmin=0, cmax=255).save('frames/{}-{:03d}-{}.png'.format(filename, self.frame, suffix))

    def run(self):
        """The main entry point for the application, runs through multiple phases at increasing resolutions.
        """
        self.model.setup(layers=['enc%i_1'%l for l in args.layers] + ['sem%i'%l for l in args.layers] + ['dec%i_1'%l for l in args.layers])
        self.prepare_style()
        self.prepare_content()
        self.prepare_generation()

        Xn = self.evaluate((self.content_img[0] + 1.0) * 127.5)
        output = self.model.finalize_image(Xn.reshape(self.content_img.shape[1:]), self.content_shape)
        scipy.misc.toimage(output, cmin=0, cmax=255).save(args.output)

        print('\n{}Optimization finished in {:3.1f}s!{}\n'.format(ansi.CYAN, time.time()-self.start_time,  ansi.ENDC))


if __name__ == "__main__":
    generator = NeuralGenerator()
    generator.run()
