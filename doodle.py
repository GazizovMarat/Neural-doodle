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
add_arg('--layers',          default=['5_1','4_1','3_1'], nargs='+', type=str, help='The layers/scales to process.')
add_arg('--variety',         default=[0.2, 0.1, 0.0], nargs='+', type=float,   help='Bias selecting diverse patches')
add_arg('--previous-weight', default=[0.0, 0.2], nargs='+', type=float,        help='Weight of previous layer features.')
add_arg('--content-weight',  default=[0.0], nargs='+', type=float, help='Weight of input content features each layer.')
add_arg('--noise-weight',    default=[0.0], nargs='+', type=float, help='Weight of noise added into features.')
add_arg('--iterations',      default=1, type=int,            help='Number of iterations to run in each phase.')
add_arg('--shapes',          default=[3], nargs='+', type=int, help='Size of kernels used for patch extraction.')
add_arg('--semantic-ext',    default='_sem.png', type=str,   help='File extension for the semantic maps.')
add_arg('--semantic-weight', default=3.0, type=float,        help='Global weight of semantics vs. style features.')
add_arg('--output',          default='output.png', type=str, help='Filename or path to save output once done.')
add_arg('--output-size',     default=None, type=str,         help='Size of the output image, e.g. 512x512.')
add_arg('--frames',          default=False, action='store_true',   help='Render intermediate frames, takes more time.')
add_arg('--slices',          default=2, type=int,            help='Split patches up into this number of batches.')
add_arg('--device',          default='cpu', type=str,        help='Index of the GPU number to use, for theano.')
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
def snap(value, grid=2**(int(args.layers[0][0])-1)): return int(grid * math.floor(value / grid))

print("""{}   {}High-quality image synthesis powered by Deep Learning!{}
  - Code licensed as AGPLv3, models under CC BY-NC-SA.{}""".format(ansi.CYAN_B, __doc__, ansi.CYAN, ansi.ENDC))

# Load the underlying deep learning libraries based on the device specified.  If you specify THEANO_FLAGS manually,
# the code assumes you know what you are doing and they are not overriden!
os.environ.setdefault('THEANO_FLAGS', 'floatX=float32,device={},force_device=True,allow_gc=True,'\
                                      'print_active_device=False'.format(args.device))

# Scientific & Imaging Libraries
import numpy as np
import scipy.optimize, scipy.ndimage, scipy.misc
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

        net['map'] = InputLayer((1, 1, None, None))
        for j in range(6):
            net['map%i'%(j+1)] = PoolLayer(net['map'], 2**j, mode='average_exc_pad')

        self.tensor_latent = []
        for l in args.layers:
            self.tensor_latent.append((l, T.tensor4()))
            # TODO: Move equation to calculate unit numbers into a common function, call from below too.
            net['lat'+l] = InputLayer((None, min(768, 32 * 2**(int(l[0])-1)), None, None), var=self.tensor_latent[-1][1])

        def EncdLayer(previous, channels, filter_size, **params):
            incoming = net['lat'+previous] if previous in args.layers else net['enc'+previous]
            return ConvLayer(incoming, channels, filter_size, **params)

        custom = {'nonlinearity': lasagne.nonlinearities.elu}
        # Encoder part of the neural network, takes an input image and turns it into abstract patterns.
        net['img']    = previous or InputLayer((None, 3, None, None))
        net['enc0_0'], net['lat0_0'] = net['img'], net['img']
        net['enc1_1'] = EncdLayer('0_0',  32, 3, pad=1, **custom)
        net['enc1_2'] = EncdLayer('1_1',  32, 3, pad=1, **custom)
        net['enc2_1'] = EncdLayer('1_2',  64, 2, pad=0, stride=(2,2), **custom)
        net['enc2_2'] = EncdLayer('2_1',  64, 3, pad=1, **custom)
        net['enc3_1'] = EncdLayer('2_2', 128, 2, pad=0, stride=(2,2), **custom)
        net['enc3_2'] = EncdLayer('3_1', 128, 3, pad=1, **custom)
        net['enc3_3'] = EncdLayer('3_2', 128, 3, pad=1, **custom)
        net['enc3_4'] = EncdLayer('3_3', 128, 3, pad=1, **custom)
        net['enc4_1'] = EncdLayer('3_4', 256, 2, pad=0, stride=(2,2), **custom)
        net['enc4_2'] = EncdLayer('4_1', 256, 3, pad=1, **custom)
        net['enc4_3'] = EncdLayer('4_2', 256, 3, pad=1, **custom)
        net['enc4_4'] = EncdLayer('4_3', 256, 3, pad=1, **custom)
        net['enc5_1'] = EncdLayer('4_4', 512, 2, pad=0, stride=(2,2), **custom)
        net['enc5_2'] = EncdLayer('5_1', 512, 3, pad=1, **custom)
        net['enc5_3'] = EncdLayer('5_2', 512, 3, pad=1, **custom)
        net['enc5_4'] = EncdLayer('5_3', 512, 3, pad=1, **custom)
        net['enc6_1'] = EncdLayer('5_4', 768, 2, pad=0, stride=(2,2), **custom)

        def DecdLayer(copy, previous, channels, **params):
            # Dynamically injects intermediate "pitstop" output layers in the decoder based on what the user
            # specified as layers. It's rather inelegant... Needs a rework!
            dup, incoming = net['enc'+copy], net['lat'+copy] if copy in args.layers else net[previous]
            return DeconvLayer(incoming, channels, dup.filter_size, stride=dup.stride, crop=dup.pad,
                               nonlinearity=params.get('nonlinearity', lasagne.nonlinearities.elu))

        # Decoder part of the neural network, takes abstract patterns and converts them into an image!
        net['dec6_1'] = DecdLayer('6_1', 'enc6_1', 512)
        net['dec5_4'] = DecdLayer('5_4', 'dec6_1', 512)
        net['dec5_3'] = DecdLayer('5_3', 'dec5_4', 512)
        net['dec5_2'] = DecdLayer('5_2', 'dec5_3', 512)
        net['dec5_1'] = DecdLayer('5_1', 'dec5_2', 256)
        net['dec4_4'] = DecdLayer('4_4', 'dec5_1', 256)
        net['dec4_3'] = DecdLayer('4_3', 'dec4_4', 256)
        net['dec4_2'] = DecdLayer('4_2', 'dec4_3', 256)
        net['dec4_1'] = DecdLayer('4_1', 'dec4_2', 128)
        net['dec3_4'] = DecdLayer('3_4', 'dec4_1', 128)
        net['dec3_3'] = DecdLayer('3_3', 'dec3_4', 128)
        net['dec3_2'] = DecdLayer('3_2', 'dec3_3', 128)
        net['dec3_1'] = DecdLayer('3_1', 'dec3_2',  64)
        net['dec2_2'] = DecdLayer('2_2', 'dec3_1',  64)
        net['dec2_1'] = DecdLayer('2_1', 'dec2_2',  32)
        net['dec1_2'] = DecdLayer('1_2', 'dec2_1',  32)
        net['dec1_1'] = DecdLayer('1_1', 'dec1_2',   3, nonlinearity=lasagne.nonlinearities.tanh)
        net['dec0_0'] = lasagne.layers.ScaleLayer(net['dec1_1'])
        net['out']    = lasagne.layers.NonlinearityLayer(net['dec0_0'], nonlinearity=lambda x: T.clip(127.5*(x+1.0), 0.0, 255.0))

        def ConcatenateLayer(incoming, layer):
            return ConcatLayer([incoming, net['map%i'%int(layer[0])]]) if args.semantic_weight > 0.0 else incoming

        # Auxiliary network for the semantic layers, and the nearest neighbors calculations.
        for layer, upper, lower in zip(args.layers, [None] + args.layers[:-1], args.layers[1:] + [None]):
            self.channels[layer] = net['enc'+layer].num_filters
            net['sem'+layer] = ConcatenateLayer(net['enc'+layer], layer)
            net['dup'+layer] = InputLayer(net['enc'+layer].output_shape)
            net['nn'+layer]  = ConvLayer(ConcatenateLayer(net['dup'+layer], layer), 1, 3, b=None, pad=0, flip_filters=False)
        self.network = net

    def load_data(self):
        """Open the serialized parameters from a pre-trained network, and load them into the model created.
        """
        data_file = os.path.join(os.path.dirname(__file__), 'gelu2_conv.pkl')
        if not os.path.exists(data_file):
            error("Model file with pre-trained convolution layers not found. Download from here...",
                  "https://github.com/alexjc/neural-doodle/releases/download/v0.0/gelu2_conv.pkl")

        data = pickle.load(open(data_file, 'rb'))
        for layer, values in data.items():
            assert layer in self.network, "Layer `{}` not found as expected.".format(layer)
            for p, v in zip(self.network[layer].get_params(), values):
                assert p.get_value().shape == v.shape, "Layer `{}` in network has size {} but data is {}."\
                                                       .format(layer, v.shape, p.get_value().shape)
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

    def compile(self, arguments, function):
        """Build a Theano function that will run the specified expression on the GPU.
        """
        return theano.function(list(arguments), function, on_unused_input='ignore', allow_input_downcast=True)

    def compute_norms(self, backend, layer, array):
        ni = backend.sqrt(backend.sum(array[:,:self.model.channels[layer]] ** 2.0, axis=(1,), keepdims=True))
        ns = backend.sqrt(backend.sum(array[:,self.model.channels[layer]:] ** 2.0, axis=(1,), keepdims=True))
        return [ni, ns]

    def normalize_components(self, layer, array, norms):
        if args.semantic_weight > 0.0:
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

        input_tensors = self.model.tensor_latent[1:] + [('0_0', self.model.tensor_img)]
        self.encoders = []
        for layer, (input, tensor_latent), shape in zip(args.layers, input_tensors, extend(args.shapes)):
            output = lasagne.layers.get_output(self.model.network['sem'+layer],
                                              {self.model.network['lat'+input]: tensor_latent,
                                               self.model.network['map']: self.model.tensor_map})
            fn = self.compile([tensor_latent, self.model.tensor_map], [output] + self.do_extract_patches([layer], [output], [shape]))
            self.encoders.append(fn)

        # Store all the style patches layer by layer, resized to match slice size and cast to 16-bit for size. 
        self.style_data, feature = {}, self.style_img
        for layer, encoder in reversed(list(zip(args.layers, self.encoders))):
            feature, *data = encoder(feature, self.style_map)
            feature = feature[:,:self.model.channels[layer]]
            patches, l = data[0], self.model.network['nn'+layer]
            l.num_filters = patches.shape[0] // args.slices
            self.style_data[layer] = [d[:l.num_filters*args.slices].astype(np.float16) for d in data]\
                                   + [np.zeros((patches.shape[0],), dtype=np.float16)]
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
            content_img_original = np.random.uniform(0, 255, content_map_original.shape[:2]+(3,)).astype(np.float32)

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
            self.content_features.insert(0, feature)
            print('  - Layer {} as {} array in {:,}kb.'.format(layer, feature.shape[1:], feature.size//1000))

    def prepare_generation(self):
        """Layerwise synthesis images requires two sets of Theano functions to be compiled.
        """
        # Patch matching calculation that uses only pre-calculated features and a slice of the patches.
        self.matcher_tensors = {l: lasagne.utils.shared_empty(dim=4) for l in args.layers}
        self.matcher_history = {l: T.vector() for l in args.layers}
        self.matcher_inputs = {self.model.network['dup'+l]: self.matcher_tensors[l] for l in args.layers}
        self.matcher_inputs.update({self.model.network['map']: self.model.tensor_map})
        nn_layers = [self.model.network['nn'+l] for l in args.layers]
        self.matcher_outputs = dict(zip(args.layers, lasagne.layers.get_output(nn_layers, self.matcher_inputs)))
        self.compute_matches = {l: self.compile([self.matcher_history[l], self.model.tensor_map],
                                                self.do_match_patches(l)) for l in args.layers}

        # Decoding intermediate features into more specialized features and all the way to the output image.
        self.encoders, input_tensors = [], self.model.tensor_latent[1:] + [('0_0', self.model.tensor_img)]
        for layer, (input, tensor_latent) in zip(args.layers, input_tensors):
            layer = lasagne.layers.get_output(self.model.network['enc'+layer],
                                             {self.model.network['lat'+input]: tensor_latent,
                                              self.model.network['map']: self.model.tensor_map})
            fn = self.compile([tensor_latent, self.model.tensor_map], layer)
            self.encoders.append(fn)

        self.decoders, output_layers = [], (['dec'+l for l in args.layers[1:]] + ['out'])
        for layer, (tt, tensor_latent), output in zip(args.layers, self.model.tensor_latent, output_layers):
            output = output.replace('_1', '_2')
            layer = lasagne.layers.get_output(self.model.network[output],
                                             {self.model.network['lat'+layer]: tensor_latent,
                                              self.model.network['map']: self.model.tensor_map})
            fn = self.compile([tensor_latent, self.model.tensor_map], layer)
            self.decoders.append(fn)


    #------------------------------------------------------------------------------------------------------------------
    # Theano Computation
    #------------------------------------------------------------------------------------------------------------------

    def do_extract_patches(self, layers, outputs, sizes, stride=1):
        """This function builds a Theano expression that will get compiled an run on the GPU. It extracts 3x3 patches
        from the intermediate outputs in the model.
        """
        results = []
        for layer, output, size in zip(layers, outputs, sizes):
            # Use a Theano helper function to extract "neighbors" of specific size, seems a bit slower than doing
            # it manually but much simpler!
            patches = theano.tensor.nnet.neighbours.images2neibs(output, (size, size), (stride, stride), mode='valid')
            # Make sure the patches are in the shape required to insert them into the model as another layer.
            patches = patches.reshape((-1, patches.shape[0] // output.shape[1], size, size)).dimshuffle((1, 0, 2, 3))
            # Calculate the magnitude that we'll use for normalization at runtime, then store...
            results.extend([patches] + self.compute_norms(T, layer, patches))
        return results

    def do_match_patches(self, layer):
        # Use node in the model to compute the result of the normalized cross-correlation, using results from the
        # nearest-neighbor layers called 'nn3_1' and 'nn4_1'.
        dist = self.matcher_outputs[layer]
        dist = dist.reshape((dist.shape[1], -1))
        # Compute the score of each patch, taking into account statistics from previous iteration. This equalizes
        # the chances of the patches being selected when the user requests more variety.
        offset = self.matcher_history[layer].reshape((-1, 1))
        scores = dist - offset
        # Pick the best style patches for each patch in the current image, the result is an array of indices.
        # Also return the maximum value along both axis, used to compare slices and add patch variety.
        return [scores.argmax(axis=0), scores.max(axis=0), dist.max(axis=1)]


    #------------------------------------------------------------------------------------------------------------------
    # Optimization Loop
    #------------------------------------------------------------------------------------------------------------------

    def iterate_batches(self, *arrays, batch_size):
        """Break down the data in arrays batch by batch and return them as a generator.
        """ 
        total_size = arrays[0].shape[0]
        indices = np.arange(total_size)
        for index in range(0, total_size, batch_size):
            excerpt = indices[index:index + batch_size]
            yield excerpt, [a[excerpt] for a in arrays]

    def evaluate_slices(self, l, f, v):
        self.normalize_components(l, f, self.compute_norms(np, l, f))
        self.matcher_tensors[l].set_value(f)

        layer, data = self.model.network['nn'+l], self.style_data[l]
        history = data[-1]

        best_idx, best_val = None, 0.0
        for idx, (bp, bi, bs, bh) in self.iterate_batches(*data, batch_size=layer.num_filters):
            weights = bp.astype(np.float32)
            self.normalize_components(l, weights, (bi, bs))
            layer.W.set_value(weights)

            cur_idx, cur_val, cur_match = self.compute_matches[l](history[idx], self.content_map)
            if best_idx is None:
                best_idx, best_val = cur_idx, cur_val
            else:
                i = np.where(cur_val > best_val)
                best_idx[i] = idx[cur_idx[i]]
                best_val[i] = cur_val[i]
            history[idx] = cur_match * v

        return best_idx, best_val

    def evaluate_feature(self, layer, feature, variety=0.0):
        """Compute best matching patches for this layer, then merge patches into a single feature array of same size.
        """
        iter_time = time.time()

        patches = self.style_data[layer][0]
        best_idx, best_val = self.evaluate_slices(layer, feature, variety)
        better_patches = patches[best_idx,:self.model.channels[layer]].astype(np.float32).transpose((0, 2, 3, 1))
        better_shape = feature.shape[2:] + (feature.shape[1],)
        better_feature = reconstruct_from_patches_2d(better_patches, better_shape)

        used = 99.9 * len(set(best_idx)) / best_idx.shape[0]
        dups = 99.9 * len([v for v in np.bincount(best_idx) if v>1]) / best_idx.shape[0]
        err = best_val.mean()
        print('  {}layer{} {:>3}   {}patches{}  used {:2.0f}%  dups {:2.0f}%   {}error{} {:3.2e}   {}time{} {:3.1f}s'\
              .format(ansi.BOLD, ansi.ENDC, layer, ansi.BOLD, ansi.ENDC, used, dups,
                      ansi.BOLD, ansi.ENDC, err, ansi.BOLD, ansi.ENDC, time.time() - iter_time))

        return better_feature.astype(np.float32).transpose((2, 0, 1))[np.newaxis]

    def evaluate_exchange(self, features):
        decoded, encoded, ready = features, features, {f.shape: [f] for f in features}
        for i in range(len(features)-1):
            decoded = [decode(data, self.content_map) for decode, data in zip(self.decoders[+i:len(self.decoders)], decoded[:-1])]
            encoded = [encode(data, self.content_map) for encode, data in zip(self.encoders[:len(self.encoders)-i], encoded[+1:])]
            for d in decoded: ready[d.shape].append(d)
            for e in encoded: ready[e.shape].append(e)
        # TODO: Weighted contribution of features of this layer with other layers...
        return [sum(ready.get(f.shape, [f])) / len(ready.get(f.shape, [f])) for f in features]

    def evaluate_merge(self, features):
        params, result = zip(*[extend(a) for a in [args.content_weight, args.noise_weight]]), []
        for f, c, p in zip(features, self.content_features, params):
            content_weight, noise_weight = p
            mixed = f * (1.0 - content_weight) + c * content_weight \
                  + np.random.normal(0.0, 1.0, size=f.shape).astype(np.float32) * noise_weight
            result.append(mixed)
        return result

    def evaluate(self, Xn):
        """Feed-forward evaluation of the output based on current image. Can be called multiple times.
        """
        frame = 0
        current_features = [np.copy(f) for f in self.content_features]
        self.render(frame, args.layers[0], current_features[0])

        for j in range(args.iterations):
            frame += 1
            print('\n{}Iteration {}{}: variety {}, weights {}.{}'.format(ansi.CYAN_B, frame, ansi.CYAN, 0.0, 0.0, ansi.ENDC))
            current_features = self.evaluate_merge(current_features)
            current_features = [self.evaluate_feature(l, f, v) for l, f, v in zip(args.layers, current_features, extend(args.variety))]
            current_features = self.evaluate_exchange(current_features)
            self.render(frame, args.layers[-1], current_features[-1])

        return self.decoders[-1](current_features[-1], self.content_map)

    def render(self, frame, layer, features):
        """Decode features at a specific layer and save the result to disk for visualization. (Takes 50% more time.) 
        """
        if not args.frames: return
        for l, compute in list(zip(args.layers, self.decoders))[args.layers.index(layer):]:
            features = compute(features[:,:self.model.channels[l]], self.content_map)

        output = self.model.finalize_image(features.reshape(self.content_img.shape[1:]), self.content_shape)
        filename = os.path.splitext(os.path.basename(args.output))[0]
        scipy.misc.toimage(output, cmin=0, cmax=255).save('frames/{}-{:03d}.png'.format(filename, frame))

    def run(self):
        """The main entry point for the application, runs through multiple phases at increasing resolutions.
        """
        self.model.setup(layers=['enc'+l for l in args.layers] + ['sem'+l for l in args.layers] + ['dec'+l for l in args.layers])
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
