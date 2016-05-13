#!/usr/bin/env python3
#
# Neural Doodle!
# Copyright (c) 2016, Alex J. Champandard.
#
# Research and Development sponsored by the nucl.ai Conference!
#   http://events.nucl.ai/
#   July 18-20, 2016 in Vienna/Austria.
#

import os
import sys
import bz2
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

add_arg('--content',        default=None, type=str,         help='Subject image path to repaint in new style.')
add_arg('--style',          default=None, type=str,         help='Texture image path to extract patches from.')
add_arg('--balance',        default=[1.0], nargs='+', type=float, help='Weight of style relative to content.')
add_arg('--variety',        default=[0.0], nargs='+', type=float, help='Bias toward selecting diverse patches, e.g. 0.5.')
add_arg('--layers',         default=['4_1'], nargs='+', type=str, help='The layer with which to match content.')
add_arg('--iterations',     default=[2], nargs='+', type=int,       help='Number of iterations to run at each resolution.')
add_arg('--shapes',         default=[3], nargs='+', type=int,     help='Size of kernels used for patch extraction.')
add_arg('--semantic-ext',   default='_sem.png', type=str,   help='File extension for the semantic maps.')
add_arg('--semantic-weight', default=3.0, type=float,       help='Global weight of semantics vs. features.')
add_arg('--output',         default='output.png', type=str, help='Filename or path to save output once done.')
add_arg('--output-size',    default=None, type=str,         help='Size of the output image, e.g. 512x512.')
add_arg('--slices',         default=2, type=int,            help='Split patches up into this number of batches.')
add_arg('--seed',           default='content', type=str,    help='Seed image path, "noise" or "content".')
add_arg('--seed-range',     default='16:240', type=str,     help='Random colors chosen in range, e.g. 0:255.')
add_arg('--device',         default='cpu', type=str,        help='Index of the GPU number to use, for theano.')
args = parser.parse_args()


#----------------------------------------------------------------------------------------------------------------------

# Color coded output helps visualize the information a little better, plus looks cool!
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

print("""{}NOTICE: This is R&D in progress. Terms and Conditions:{}
  - Trained models are for non-commercial use, no redistribution.
  - For derived/inspired research, please cite this project.\n{}""".format(ansi.YELLOW_B, ansi.YELLOW, ansi.ENDC))

print('{}Neural Doodle for semantic style transfer.{}'.format(ansi.CYAN_B, ansi.ENDC))

# Load the underlying deep learning libraries based on the device specified.  If you specify THEANO_FLAGS manually,
# the code assumes you know what you are doing and they are not overriden!
os.environ.setdefault('THEANO_FLAGS', 'floatX=float32,device={},force_device=True,'\
                                      'print_active_device=False'.format(args.device))

# Scientific & Imaging Libraries
import numpy as np
import scipy.optimize, scipy.ndimage, scipy.misc
import PIL
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

print('{}  - Using device `{}` for processing the images.{}'.format(ansi.CYAN, theano.config.device, ansi.ENDC))


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


        def DecvLayer(copy, previous, channels, **params):
            # Dynamically injects intermediate pitstop layers in the encoder based on what the user
            # specified as layers. It's rather inelegant... Needs a rework!
            if copy in args.layers:
                if len(self.tensor_latent) > 0:
                    l = self.tensor_latent[-1][0]
                    if args.semantic_weight > 0.0:
                        net['out'+l] = ConcatLayer([previous, net['map%i'%(int(l[0])-1)]])
                    else:
                        net['out'+l] = previous

                self.tensor_latent.append((copy, T.tensor4()))
                net['lat'+copy] = InputLayer((1, previous.num_filters, None, None), var=self.tensor_latent[-1][1])
                previous = net['lat'+copy]

            dup = net['enc'+copy]
            return DeconvLayer(previous, channels, dup.filter_size, stride=dup.stride, crop=dup.pad,
                               nonlinearity=params.get('nonlinearity', lasagne.nonlinearities.elu))

        custom = {'nonlinearity': lasagne.nonlinearities.elu}
        # Encoder part of the neural network, takes an input image and turns it into abstract patterns.
        net['img']    = previous or InputLayer((1, 3, None, None))
        net['enc1_1'] = ConvLayer(net['img'],     32, 3, pad=1, **custom)
        net['enc1_2'] = ConvLayer(net['enc1_1'],  32, 3, pad=1, **custom)
        net['enc2_1'] = ConvLayer(net['enc1_2'],  64, 2, pad=0, stride=(2,2), **custom)
        net['enc2_2'] = ConvLayer(net['enc2_1'],  64, 3, pad=1, **custom)
        net['enc3_1'] = ConvLayer(net['enc2_2'], 128, 2, pad=0, stride=(2,2), **custom)
        net['enc3_2'] = ConvLayer(net['enc3_1'], 128, 3, pad=1, **custom)
        net['enc3_3'] = ConvLayer(net['enc3_2'], 128, 3, pad=1, **custom)
        net['enc3_4'] = ConvLayer(net['enc3_3'], 128, 3, pad=1, **custom)
        net['enc4_1'] = ConvLayer(net['enc3_4'], 256, 2, pad=0, stride=(2,2), **custom)
        net['enc4_2'] = ConvLayer(net['enc4_1'], 256, 3, pad=1, **custom)
        net['enc4_3'] = ConvLayer(net['enc4_2'], 256, 3, pad=1, **custom)
        net['enc4_4'] = ConvLayer(net['enc4_3'], 256, 3, pad=1, **custom)
        net['enc5_1'] = ConvLayer(net['enc4_4'], 512, 2, pad=0, stride=(2,2), **custom)
        net['enc5_2'] = ConvLayer(net['enc5_1'], 512, 3, pad=1, **custom)
        net['enc5_3'] = ConvLayer(net['enc5_2'], 512, 3, pad=1, **custom)
        net['enc5_4'] = ConvLayer(net['enc5_3'], 512, 3, pad=1, **custom)
        net['enc6_1'] = ConvLayer(net['enc5_4'], 768, 2, pad=0, stride=(2,2), **custom)

        # Decoder part of the neural network, takes abstract patterns and converts them into an image!
        self.tensor_latent = []
        net['dec6_1'] = DecvLayer('6_1', net['enc6_1'], 512)
        net['dec5_4'] = DecvLayer('5_4', net['dec6_1'], 512)
        net['dec5_3'] = DecvLayer('5_3', net['dec5_4'], 512)
        net['dec5_2'] = DecvLayer('5_2', net['dec5_3'], 512)
        net['dec5_1'] = DecvLayer('5_1', net['dec5_2'], 256)
        net['dec4_4'] = DecvLayer('4_4', net['dec5_1'], 256)
        net['dec4_3'] = DecvLayer('4_3', net['dec4_4'], 256)
        net['dec4_2'] = DecvLayer('4_2', net['dec4_3'], 256)
        net['dec4_1'] = DecvLayer('4_1', net['dec4_2'], 128)
        net['dec3_4'] = DecvLayer('3_4', net['dec4_1'], 128)
        net['dec3_3'] = DecvLayer('3_3', net['dec3_4'], 128)
        net['dec3_2'] = DecvLayer('3_2', net['dec3_3'], 128)
        net['dec3_1'] = DecvLayer('3_1', net['dec3_2'],  64)
        net['dec2_2'] = DecvLayer('2_2', net['dec3_1'],  64)
        net['dec2_1'] = DecvLayer('2_1', net['dec2_2'],  32)
        net['dec1_2'] = DecvLayer('1_2', net['dec2_1'],  32)
        net['dec1_1'] = DecvLayer('1_1', net['dec1_2'],   3, nonlinearity=lasagne.nonlinearities.tanh)
        net['dec0_0'] = lasagne.layers.ScaleLayer(net['dec1_1'])

        l = self.tensor_latent[-1][0]
        net['out'+l]  = lasagne.layers.NonlinearityLayer(net['dec0_0'], nonlinearity=lambda x: T.clip(127.5*(x+1.0), 0.0, 255.0))

        # Auxiliary network for the semantic layers, and the nearest neighbors calculations.
        for j, i in itertools.product(range(6), range(4)):
            suffix = '%i_%i' % (j+1, i+1)
            if 'enc'+suffix not in net: continue

            self.channels[suffix] = net['enc'+suffix].num_filters
            if args.semantic_weight > 0.0:
                net['sem'+suffix] = ConcatLayer([net['enc'+suffix], net['map%i'%(j+1)]])
            else:
                net['sem'+suffix] = net['enc'+suffix]

            net['dup'+suffix] = InputLayer(net['sem'+suffix].output_shape)
            net['nn'+suffix] = ConvLayer(net['dup'+suffix], 1, 3, b=None, pad=0, flip_filters=False)

        self.network = net

    def load_data(self):
        """Open the serialized parameters from a pre-trained network, and load them into the model created.
        """
        data_file = os.path.join(os.path.dirname(__file__), 'gelu2_conv.pkl.bz2')
        if not os.path.exists(data_file):
            error("Model file with pre-trained convolution layers not found. Download here...",
                  "https://github.com/alexjc/neural-doodle/releases/download/v0.0/gelu2_conv.pkl.bz2")

        data = pickle.load(bz2.open(data_file, 'rb'))
        for layer, values in data.items():
            assert layer in self.network, "Layer `{}` not found as expected.".format(layer)
            for p, v in zip(self.network[layer].get_params(), values):
                assert p.get_value().shape == v.shape, "Layer `{}` in network has size {} but data is {}."\
                                                       .format(layer, v.shape, p.get_value().shape)
                p.set_value(v)

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
        """Given an image loaded from disk, turn it into a representation compatible with the model.
        The format is (b,c,y,x) with batch=1 for a single image, channels=3 for RGB, and y,x matching
        the resolution.
        """
        image = np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)[::-1, :, :]
        image = image.astype(np.float32) / 127.5 - 1.0
        return image[np.newaxis]

    def finalize_image(self, image, resolution):
        """Based on the output of the neural network, convert it into an image format that can be saved
        to disk -- shuffling dimensions as appropriate.
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
        if args.output is not None and os.path.isfile(args.output):
            os.remove(args.output)

        print(ansi.CYAN, end='')
        target = args.content or args.output
        self.content_img_original, self.content_map_original = self.load_images('content', target)
        self.style_img_original, self.style_map_original = self.load_images('style', args.style)

        if self.content_map_original is None and self.content_img_original is None:
            print("  - No content files found; result depends on seed only.")
        print(ansi.ENDC, end='')

        # Display some useful errors if the user's input can't be undrestood.
        if self.style_img_original is None:
            error("Couldn't find style image as expected.",
                  "  - Try making sure `{}` exists and is a valid image.".format(args.style))

        if self.content_map_original is not None and self.style_map_original is None:
            basename, _ = os.path.splitext(args.style)
            error("Expecting a semantic map for the input style image too.",
                  "  - Try creating the file `{}_sem.png` with your annotations.".format(basename))

        if self.style_map_original is not None and self.content_map_original is None:
            basename, _ = os.path.splitext(target)
            error("Expecting a semantic map for the input content image too.",
                  "  - Try creating the file `{}_sem.png` with your annotations.".format(basename))

        if self.content_map_original is None:
            if self.content_img_original is None and args.output_size:
                shape = tuple([int(i) for i in args.output_size.split('x')])
            else:
                if self.content_img_original is None:
                    shape = self.style_img_original.shape[:2]
                else:
                    shape = self.content_img_original.shape[:2]

            self.content_map_original = np.zeros(shape+(3,))
            args.semantic_weight = 0.0

        if self.style_map_original is None:
            self.style_map_original = np.zeros(self.style_img_original.shape[:2]+(3,))
            args.semantic_weight = 0.0

        if self.content_img_original is None:
            self.content_img_original = np.zeros(self.content_map_original.shape[:2]+(3,))
            args.content_weight = 0.0

        if self.content_map_original.shape[2] != self.style_map_original.shape[2]:
            error("Mismatch in number of channels for style and content semantic map.",
                  "  - Make sure both images are RGB, RGBA, or L.")

        # Finalize the parameters based on what we loaded, then create the model.
        args.semantic_weight = math.sqrt(9.0 / args.semantic_weight) if args.semantic_weight else 0.0
        self.model = Model()


    #------------------------------------------------------------------------------------------------------------------
    # Helper Functions
    #------------------------------------------------------------------------------------------------------------------

    def load_images(self, name, filename):
        """If the image and map files exist, load them. Otherwise they'll be set to default values later.
        """
        basename, _ = os.path.splitext(filename)
        mapname = basename + args.semantic_ext
        img = scipy.ndimage.imread(filename, mode='RGB') if os.path.exists(filename) else None
        map = scipy.ndimage.imread(mapname) if os.path.exists(mapname) and args.semantic_weight > 0.0 else None

        if img is not None: print('  - Loading `{}` for {} data.'.format(filename, name))
        if map is not None: print('  - Adding `{}` as semantic map.'.format(mapname))

        if img is not None and map is not None and img.shape[:2] != map.shape[:2]:
            error("The {} image and its semantic map have different resolutions. Either:".format(name),
                  "  - Resize {} to {}, or\n  - Resize {} to {}."\
                  .format(filename, map.shape[1::-1], mapname, img.shape[1::-1]))
        return img, map

    def compile(self, arguments, function):
        """Build a Theano function that will run the specified expression on the GPU.
        """
        return theano.function(list(arguments), function, on_unused_input='ignore', allow_input_downcast=True)

    def compute_norms(self, backend, layer, array):
        ni = backend.sqrt(backend.sum(array[:,:self.model.channels[layer]] ** 2.0, axis=(1,), keepdims=True))
        ns = backend.sqrt(backend.sum(array[:,self.model.channels[layer]:] ** 2.0, axis=(1,), keepdims=True))
        return [ni] + [ns]

    def normalize_components(self, layer, array, norms):
        if args.semantic_weight > 0.0:
            array[:,self.model.channels[layer]:] /= (norms[1] * args.semantic_weight)
        array[:,:self.model.channels[layer]] /= (norms[0] * 3.0)


    #------------------------------------------------------------------------------------------------------------------
    # Initialization & Setup
    #------------------------------------------------------------------------------------------------------------------

    def rescale_image(self, img, scale):
        """Re-implementing skimage.transform.scale without the extra dependency. Saves a lot of space and hassle!
        """
        def snap(value, grid=2**(int(args.layers[0][0])-1)): return int(grid * math.floor(value / grid))
        output = scipy.misc.toimage(img, cmin=0.0, cmax=255)
        output.thumbnail((snap(output.size[0]*scale), snap(output.size[1]*scale)), PIL.Image.ANTIALIAS)
        return np.asarray(output)

    def prepare_content(self, scale=1.0):
        """Called each phase of the optimization, rescale the original content image and its map to use as inputs.
        """
        content_img = self.rescale_image(self.content_img_original, scale)
        self.content_img = self.model.prepare_image(content_img)

        content_map = self.rescale_image(self.content_map_original, scale)
        self.content_map = content_map.transpose((2, 0, 1))[np.newaxis].astype(np.float32)

    def prepare_style(self, scale=1.0):
        """Called each phase of the optimization, process the style image according to the scale, then run it
        through the model to extract intermediate outputs (e.g. sem4_1) and turn them into patches.
        """
        style_img = self.rescale_image(self.style_img_original, scale)
        self.style_img = self.model.prepare_image(style_img)

        style_map = self.rescale_image(self.style_map_original, scale)
        self.style_map = style_map.transpose((2, 0, 1))[np.newaxis].astype(np.float32)

        # Compile a function to run on the GPU to extract patches for all layers at once.
        layer_patches = self.do_extract_patches(args.layers, self.model.get_outputs('sem', args.layers), args.shapes)
        extractor = self.compile([self.model.tensor_img, self.model.tensor_map], layer_patches)
        result = extractor(self.style_img, self.style_map)

        # Store all the style patches layer by layer, resized to match slice size and cast to 16-bit for size. 
        self.style_data = {}
        for layer, *data in zip(args.layers, result[0::3], result[1::3], result[2::3]):
            patches = data[0]
            l = self.model.network['nn'+layer]
            l.num_filters = patches.shape[0] // args.slices
            self.style_data[layer] = [d[:l.num_filters*args.slices].astype(np.float16) for d in data]\
                                   + [np.zeros((patches.shape[0],), dtype=np.float16)]
            print('  - Style layer {}: {} patches in {:,}kb.'.format(layer, patches.shape, patches.size//1000))

    def prepare_optimization(self):
        """Optimization requires a function to compute the error (aka. loss) which is done in multiple components.
        Here we compile a function to run on the GPU that returns all components separately.
        """

        # Feed-forward calculation only, returns the result of the convolution post-activation 
        self.compute_features = self.compile([self.model.tensor_img, self.model.tensor_map],
                                             self.model.get_outputs('sem', args.layers))

        # Patch matching calculation that uses only pre-calculated features and a slice of the patches.
        self.matcher_tensors = {l: lasagne.utils.shared_empty(dim=4) for l in args.layers}
        self.matcher_history = {l: T.vector() for l in args.layers}
        self.matcher_inputs = {self.model.network['dup'+l]: self.matcher_tensors[l] for l in args.layers}
        nn_layers = [self.model.network['nn'+l] for l in args.layers]
        self.matcher_outputs = dict(zip(args.layers, lasagne.layers.get_output(nn_layers, self.matcher_inputs)))

        self.compute_matches = {l: self.compile([self.matcher_history[l]], self.do_match_patches(l))\
                                                for l in args.layers}

        self.compute_output = []
        for layer, (_, tensor_latent) in zip(args.layers, self.model.tensor_latent):
            output = lasagne.layers.get_output(self.model.network['out'+layer],
                                              {self.model.network['lat'+layer]: tensor_latent,
                                               self.model.network['map']: self.model.tensor_map})
            fn = self.compile([tensor_latent, self.model.tensor_map], output)
            self.compute_output.append(fn)


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
    # Error/Loss Functions
    #------------------------------------------------------------------------------------------------------------------

    def content_loss(self):
        """Return a list of Theano expressions for the error function, measuring how different the current image is
        from the reference content that was loaded.
        """

        content_loss = []
        if args.content_weight == 0.0:
            return content_loss

        # First extract all the features we need from the model, these results after convolution.
        extractor = theano.function([self.model.tensor_img], self.model.get_outputs('enc', args.layers))
        result = extractor(self.content_img)

        # Build a list of loss components that compute the mean squared error by comparing current result to desired.
        for l, ref in zip(args.layers, result):
            layer = self.model.tensor_outputs['enc'+l]
            loss = T.mean((layer - ref) ** 2.0)
            content_loss.append(('content', l, args.content_weight * loss))
            print('  - Content layer conv{}: {} features in {:,}kb.'.format(l, ref.shape[1], ref.size//1000))
        return content_loss

    def style_loss(self):
        """Returns a list of loss components as Theano expressions. Finds the best style patch for each patch in the
        current image using normalized cross-correlation, then computes the mean squared error for all patches.
        """
        style_loss = []
        if args.style_weight == 0.0:
            return style_loss

        # Extract the patches from the current image, as well as their magnitude.
        result = self.do_extract_patches(args.layers, self.model.get_outputs('enc', args.layers), args.shapes)

        # Multiple style layers are optimized separately, usually conv3_1 and conv4_1 — semantic data not used here.
        for l, matches, patches in zip(args.layers, self.tensor_matches, result[0::3]):
            # Compute the mean squared error between the current patch and the best matching style patch.
            # Ignore the last channels (from semantic map) so errors returned are indicative of image only.
            loss = T.mean((patches - matches[:,:self.model.channels[l]]) ** 2.0)
            style_loss.append(('style', l, args.style_weight * loss))
        return style_loss

    def total_variation_loss(self):
        """Return a loss component as Theano expression for the smoothness prior on the result image.
        """
        x = self.model.tensor_img
        loss = (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).mean()
        return [('smooth', 'img', args.smoothness * loss)]


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

    def evaluate_slices(self, f, l, v):
        layer, data = self.model.network['nn'+l], self.style_data[l]
        history = data[-1]

        best_idx, best_val = None, 0.0
        for idx, (bp, bi, bs, bh) in self.iterate_batches(*data, batch_size=layer.num_filters):
            weights = bp.astype(np.float32)
            self.normalize_components(l, weights, (bi, bs))
            layer.W.set_value(weights)

            cur_idx, cur_val, cur_match = self.compute_matches[l](history[idx])
            if best_idx is None:
                best_idx, best_val = cur_idx, cur_val
            else:
                i = np.where(cur_val > best_val)
                best_idx[i] = idx[cur_idx[i]]
                best_val[i] = cur_val[i]

            history[idx] = cur_match * v

        return best_idx, best_val

    def evaluate(self, Xn):
        """Feed-forward evaluation of the output based on current image. Can be called multiple times.
        """
        frame = 0

        # Adjust the representation to be compatible with the model before computing results.
        current_img = Xn.reshape(self.content_img.shape).astype(np.float32) / 127.5 - 1.0
        current_features = self.compute_features(current_img, self.content_map)

        # Iterate through each of the style layers one by one, computing best matches.
        desired_feature = current_features[0]

        for l, iterations, balance, variety, current_feature, compute in\
                zip(args.layers, args.iterations, args.balance, args.variety, current_features, self.compute_output):

            print('{}patches {}{}'.format(ansi.BOLD, l, ansi.ENDC))

            self.iter_time = time.time()
            channels = self.model.channels[l]
            f = np.copy(desired_feature)

            for j in range(iterations):
                self.normalize_components(l, f, self.compute_norms(np, l, f))
                self.matcher_tensors[l].set_value(f)

                # Compute best matching patches this style layer, going through all slices.
                warmup = bool(j == 0 and variety > 0.0)
                for _ in range(2 if warmup else 1):
                    best_idx, best_val = self.evaluate_slices(f, l, variety)

                patches = self.style_data[l][0]
                current_best = patches[best_idx].astype(np.float32)

                better_patches = current_best.transpose((0, 2, 3, 1))
                better_features = reconstruct_from_patches_2d(better_patches, f.shape[2:] + (f.shape[1],))
                better_features = better_features.astype(np.float32).transpose((2, 0, 1))[np.newaxis]
                f = better_features


                used = 100.0 * len(set(best_idx)) / best_idx.shape[0]
                dups = 100.0 * len([v for v in collections.Counter(best_idx).values() if v>1]) / best_idx.shape[0]
                err = best_val.mean()
                print('{:>3} used {:2.0f}% dups {:2.0f}% loss {:3.1f}'.format(frame, used, dups, err), end='')

                current_time = time.time()
                print('  {}time{} {:3.1f}s '.format(ansi.BOLD, ansi.ENDC, current_time - self.iter_time), flush=True)
                self.iter_time = current_time
                frame += 1

            f = (1.0 - balance) * current_feature[:,:channels] + (0.0 + balance) * better_features[:,:channels]
            desired_feature = compute(f, self.content_map)

        return desired_feature

    def run(self):
        """The main entry point for the application, runs through multiple phases at increasing resolutions.
        """
        self.frame = 0

        shape = self.content_img_original.shape
        print('\n{}Resolution {}x{}{}'\
                .format(ansi.BLUE_B, shape[1], shape[0], ansi.BLUE))

        # Precompute all necessary data for the various layers, put patches in place into augmented network.
        self.model.setup(layers=['sem'+l for l in args.layers])
        self.prepare_content()
        self.prepare_style()

        # Now setup the model with the new data, ready for the optimization loop.
        self.model.setup(layers=['sem'+l for l in args.layers] + ['out'+l for l in args.layers])
        self.prepare_optimization()
        print('{}'.format(ansi.ENDC))

        # Setup the seed for the optimization as specified by the user.
        shape = self.content_img.shape[2:]
        if args.seed == 'content':
            Xn = (self.content_img[0] + 1.0) * 127.5
        elif args.seed == 'noise':
            bounds = [int(i) for i in args.seed_range.split(':')]
            Xn = np.random.uniform(bounds[0], bounds[1], shape + (3,)).astype(np.float32)
        elif os.path.exists(args.seed):
            seed_image = scipy.ndimage.imread(args.seed, mode='RGB')
            seed_image = scipy.misc.imresize(seed_image, shape, interp='bicubic')
            self.seed_image = self.model.prepare_image(seed_image)
            Xn = (self.seed_image[0] + 1.0) * 127.5
        if Xn is None:
            error("Seed for optimization was not found. You can either...",
                    "  - Set the `--seed` to `content` or `noise`.", "  - Specify `--seed` as a valid filename.")

        Xn = self.evaluate(Xn)
        output = self.model.finalize_image(Xn.reshape(self.content_img.shape[1:]), self.content_img_original.shape)
        scipy.misc.toimage(output, cmin=0, cmax=255).save(args.output)

        print('\n{}Optimization finished in {:3.1f}s!{}\n'.format(ansi.CYAN, time.time()-self.start_time,  ansi.ENDC))


if __name__ == "__main__":
    generator = NeuralGenerator()
    generator.run()
