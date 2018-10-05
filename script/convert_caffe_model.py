import numpy as np  
import sys,os  
import cv2
import argparse
caffe_root = '/home/yaochuanqi/work/ssd/caffe'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe  

def get_aligned(c):
    if c == 1: #dw
        return 1
    if c % 4 != 0:
        return c + 4 - c % 4
    return c

def get_layer_names(net):
    all_layer_names = []
    names = net._layer_names
    for i in range(len(names)):
       all_layer_names = all_layer_names + [names[i]] 
    return all_layer_names

def get_layer_by_name(net, name):
    layer_names = get_layer_names(net)
    index = layer_names.index(name)
    return net.layers[index]

def get_prev_layer(net, cur_layer):
    layer_names = get_layer_names(net)
    bottom = net.bottom_names[cur_layer][0]

    for name in net.top_names:
        top = net.top_names[name][0]
        if bottom == top and top != net.bottom_names[name][0]:
           return get_layer_by_name(net, name)

def convert_net(net, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for key in net.params.iterkeys():
        if type(net.params[key]) is not caffe._caffe.BlobVec:
            print(key)
        else:
            for i in range(len(net.params[key])):
                layer_type = get_layer_by_name(net, key).type
                save_name = save_dir + '/' + key.replace('/', '_') + '-' + str(i) + '.dat'
                print(save_name)
                tosave = net.params[key][i].data
                shape = tosave.shape # o, i, h, w
                #padding the channel number divided by 4 to get benefits of 4x batch operation of rendercript

                if len(shape) == 1:
                   if layer_type == "Convolution" or layer_type == "Deconvolution":
                       pad = get_aligned(shape[0]) - shape[0] 
                       tosave = np.pad(tosave, (0,pad), "constant")
                else:
                   outc = shape[0]
                   inc = shape[1]
                   padinc = get_aligned(inc)
                   padoutc = get_aligned(outc)
                   padin = padinc - inc
                   padout = padoutc - outc
                   if layer_type == "Convolution":
                       tosave =  np.pad(tosave, ((0,padout),(0,padin),(0,0),(0,0)), "constant")
                       tosave = tosave.transpose(0, 2, 3, 1)
                   elif layer_type == "Deconvolution":
                       tosave =  np.pad(tosave, ((0,padout),(0, 0),(0,0),(0,0)), "constant")
                       tosave = tosave.transpose(1, 2, 3, 0)
                   elif layer_type == "InnerProduct":
                       bottom_layer = get_prev_layer(net, key)
                       if bottom_layer is not None and bottom_layer.type == "ROIPooling": # for faster-rcnn
                           print("bottom type=" + bottom_layer.type)
                           tosave = tosave.reshape(-1, 256, 6, 6)
                           tosave = tosave.transpose(0, 2, 3, 1)
                   else:
                       print("error: layer type " + layer_type + " not supported.")
                       return
                print(layer_type)
                print(tosave.shape)
                tosave.tofile(save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m','--model',
        type=str,
        required=True,
        help='The input caffe prototxt file'
    )
    parser.add_argument(
        '-w','--weights',
        type=str,
        default=None,
        help='The input caffemodel file'
    )
    parser.add_argument(
        '-o','--savedir',
        type=str,
        default=None,
        help='The output directory'
    )
    FLAGS, unparsed = parser.parse_known_args()

    net = caffe.Net(FLAGS.model, FLAGS.weights, caffe.TEST)  
    convert_net(net, FLAGS.savedir)
    os.system("cp " + FLAGS.model + " " + FLAGS.savedir)

