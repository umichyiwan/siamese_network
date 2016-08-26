import mxnet as mx


def get_single_branch(data, rois,
                conv1_1_weight, conv1_1_bias, conv1_2_weight, conv1_2_bias,
                conv2_1_weight, conv2_1_bias, conv2_2_weight, conv2_2_bias,
                conv3_1_weight, conv3_1_bias, conv3_2_weight, conv3_2_bias, conv3_3_weight, conv3_3_bias,
                conv4_1_weight, conv4_1_bias, conv4_2_weight, conv4_2_bias, conv4_3_weight, conv4_3_bias,
                conv5_1_weight, conv5_1_bias, conv5_2_weight, conv5_2_bias, conv5_3_weight, conv5_3_bias,
                fc6_weight, fc6_bias):
    """
    shared convolutional layers
    :param data: Symbol
    :return: Symbol
    """
    # group 1
    conv1_1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, weight=conv1_1_weight, bias=conv1_1_bias)
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu")

    conv1_2 = mx.symbol.Convolution(data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, weight=conv1_2_weight, bias=conv1_2_bias)
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu")

    pool1 = mx.symbol.Pooling(data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2))

    # group 2
    conv2_1 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, weight=conv2_1_weight, bias=conv2_1_bias)
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu")

    conv2_2 = mx.symbol.Convolution(data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, weight=conv2_2_weight, bias=conv2_2_bias)
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu")

    pool2 = mx.symbol.Pooling(data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2))

    # group 3
    conv3_1 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, weight=conv3_1_weight, bias=conv3_1_bias)
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu")

    conv3_2 = mx.symbol.Convolution(data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, weight=conv3_2_weight, bias=conv3_2_bias)
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu")

    conv3_3 = mx.symbol.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, weight=conv3_3_weight, bias=conv3_3_bias)
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu")

    # group 4
    conv4_1 = mx.symbol.Convolution(data=relu3_3, kernel=(3, 3), pad=(1, 1), num_filter=512, weight=conv4_1_weight, bias=conv4_1_bias)
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu")

    conv4_2 = mx.symbol.Convolution(data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, weight=conv4_2_weight, bias=conv4_2_bias)
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu")

    conv4_3 = mx.symbol.Convolution(data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, weight=conv4_3_weight, bias=conv4_3_bias)
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu")

    pool4 = mx.symbol.ROIPooling(data=relu4_3, rois=rois, pooled_size=(7, 7), spatial_scale=0.125)
    flatten4 = mx.symbol.Flatten(data=pool4)

    # group 5
    conv5_1 = mx.symbol.Convolution(data=relu4_3, kernel=(3, 3), pad=(1, 1), num_filter=512, weight=conv5_1_weight, bias=conv5_1_bias)
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu")

    conv5_2 = mx.symbol.Convolution(data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, weight=conv5_2_weight, bias=conv5_2_bias)
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu")

    conv5_3 = mx.symbol.Convolution(data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, weight=conv5_3_weight, bias=conv5_3_bias)
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu")

    pool5 = mx.symbol.ROIPooling(data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
    flatten5 = mx.symbol.Flatten(data=pool5)

    # group 6
    fc6 = mx.symbol.FullyConnected(data=flatten5, num_hidden=4096, weight=fc6_weight, bias=fc6_bias)
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5)
    cat = mx.symbol.Concat(flatten5, drop6, flatten4)
    l2_norm = mx.symbol.L2Normalization(cat)

    return l2_norm


def get_siamese_train(batch_size):
    data = mx.sym.Variable('data')
    data_siamese = mx.sym.Variable('data_siamese')
    label = mx.symbol.Variable(name='label')
    label = mx.symbol.Reshape(data=label, shape=(-1, 1), name='label_reshape')
    conv1_1_weight = mx.sym.Variable('conv1_1_weight')
    conv1_1_bias = mx.sym.Variable('conv1_1_bias')
    conv1_2_weight = mx.sym.Variable('conv1_2_weight')
    conv1_2_bias = mx.sym.Variable('conv1_2_bias')
    conv2_1_weight = mx.sym.Variable('conv2_1_weight')
    conv2_1_bias = mx.sym.Variable('conv2_1_bias')
    conv2_2_weight = mx.sym.Variable('conv2_2_weight')
    conv2_2_bias = mx.sym.Variable('conv2_2_bias')
    conv3_1_weight = mx.sym.Variable('conv3_1_weight')
    conv3_1_bias = mx.sym.Variable('conv3_1_bias')
    conv3_2_weight = mx.sym.Variable('conv3_2_weight')
    conv3_2_bias = mx.sym.Variable('conv3_2_bias')
    conv3_3_weight = mx.sym.Variable('conv3_3_weight')
    conv3_3_bias = mx.sym.Variable('conv3_3_bias')
    conv4_1_weight = mx.sym.Variable('conv4_1_weight')
    conv4_1_bias = mx.sym.Variable('conv4_1_bias')
    conv4_2_weight = mx.sym.Variable('conv4_2_weight')
    conv4_2_bias = mx.sym.Variable('conv4_2_bias')
    conv4_3_weight = mx.sym.Variable('conv4_3_weight')
    conv4_3_bias = mx.sym.Variable('conv4_3_bias')
    conv5_1_weight = mx.sym.Variable('conv5_1_weight')
    conv5_1_bias = mx.sym.Variable('conv5_1_bias')
    conv5_2_weight = mx.sym.Variable('conv5_2_weight')
    conv5_2_bias = mx.sym.Variable('conv5_2_bias')
    conv5_3_weight = mx.sym.Variable('conv5_3_weight')
    conv5_3_bias = mx.sym.Variable('conv5_3_bias')
    fc6_weight = mx.sym.Variable('fc6_weight')
    fc6_bias = mx.sym.Variable('fc6_bias')
    rois = mx.symbol.Variable(name='rois')
    rois = mx.symbol.Reshape(data=rois, shape=(-1, 5))
    rois_siamese = mx.symbol.Variable(name='rois_siamese')
    rois_siamese = mx.symbol.Reshape(data=rois_siamese, shape=(-1, 5))
    b1 = get_single_branch(data, rois,
                           conv1_1_weight, conv1_1_bias, conv1_2_weight, conv1_2_bias,
                           conv2_1_weight, conv2_1_bias, conv2_2_weight, conv2_2_bias,
                           conv3_1_weight, conv3_1_bias, conv3_2_weight, conv3_2_bias, conv3_3_weight, conv3_3_bias,
                           conv4_1_weight, conv4_1_bias, conv4_2_weight, conv4_2_bias, conv4_3_weight, conv4_3_bias,
                           conv5_1_weight, conv5_1_bias, conv5_2_weight, conv5_2_bias, conv5_3_weight, conv5_3_bias,
                           fc6_weight, fc6_bias)
    b2 = get_single_branch(data_siamese, rois_siamese,
                           conv1_1_weight, conv1_1_bias, conv1_2_weight, conv1_2_bias,
                           conv2_1_weight, conv2_1_bias, conv2_2_weight, conv2_2_bias,
                           conv3_1_weight, conv3_1_bias, conv3_2_weight, conv3_2_bias, conv3_3_weight, conv3_3_bias,
                           conv4_1_weight, conv4_1_bias, conv4_2_weight, conv4_2_bias, conv4_3_weight, conv4_3_bias,
                           conv5_1_weight, conv5_1_bias, conv5_2_weight, conv5_2_bias, conv5_3_weight, conv5_3_bias,
                           fc6_weight, fc6_bias)
    diff = b1 - b2
    square_dist = mx.sym.sum(mx.symbol.square(diff), axis=1, keepdims=1)
    input_shapes = {"data": (10, 3, 480, 640), "data_siamese": (10, 3, 480, 640), "rois": (10, 16, 5),
                    "rois_siamese": (10, 16, 5)}
    arg_shape, output_shape, aux_shape = square_dist.infer_shape(**input_shapes)
    print arg_shape
    print output_shape
    print aux_shape
    # input_shapes = {"data": (10, 3, 480, 640), "data_siamese": (10, 3, 480, 640), "rois": (10, 16, 5), "rois_siamese": (10, 16, 5),
    #                 "label": (10, 16)}
    loss = 0.5 * label * square_dist + 0.5 * (1 - label) * mx.symbol.maximum(0, 0.5 - square_dist)
    loss = mx.symbol.Reshape(data=loss, shape=(batch_size, -1))
    # arg_shape, output_shape, aux_shape = loss.infer_shape(**input_shapes)
    # print arg_shape
    # print output_shape
    # print aux_shape
    return mx.sym.MakeLoss(loss)


def get_single_branch_test():
    data = mx.sym.Variable('data')
    conv1_1_weight = mx.sym.Variable('conv1_1_weight')
    conv1_1_bias = mx.sym.Variable('conv1_1_bias')
    conv1_2_weight = mx.sym.Variable('conv1_2_weight')
    conv1_2_bias = mx.sym.Variable('conv1_2_bias')
    conv2_1_weight = mx.sym.Variable('conv2_1_weight')
    conv2_1_bias = mx.sym.Variable('conv2_1_bias')
    conv2_2_weight = mx.sym.Variable('conv2_2_weight')
    conv2_2_bias = mx.sym.Variable('conv2_2_bias')
    conv3_1_weight = mx.sym.Variable('conv3_1_weight')
    conv3_1_bias = mx.sym.Variable('conv3_1_bias')
    conv3_2_weight = mx.sym.Variable('conv3_2_weight')
    conv3_2_bias = mx.sym.Variable('conv3_2_bias')
    conv3_3_weight = mx.sym.Variable('conv3_3_weight')
    conv3_3_bias = mx.sym.Variable('conv3_3_bias')
    conv4_1_weight = mx.sym.Variable('conv4_1_weight')
    conv4_1_bias = mx.sym.Variable('conv4_1_bias')
    conv4_2_weight = mx.sym.Variable('conv4_2_weight')
    conv4_2_bias = mx.sym.Variable('conv4_2_bias')
    conv4_3_weight = mx.sym.Variable('conv4_3_weight')
    conv4_3_bias = mx.sym.Variable('conv4_3_bias')
    conv5_1_weight = mx.sym.Variable('conv5_1_weight')
    conv5_1_bias = mx.sym.Variable('conv5_1_bias')
    conv5_2_weight = mx.sym.Variable('conv5_2_weight')
    conv5_2_bias = mx.sym.Variable('conv5_2_bias')
    conv5_3_weight = mx.sym.Variable('conv5_3_weight')
    conv5_3_bias = mx.sym.Variable('conv5_3_bias')
    fc6_weight = mx.sym.Variable('fc6_weight')
    fc6_bias = mx.sym.Variable('fc6_bias')
    rois = mx.symbol.Variable(name='rois')
    rois = mx.symbol.Reshape(data=rois, shape=(-1, 5))
    b = get_single_branch(data, rois,
                           conv1_1_weight, conv1_1_bias, conv1_2_weight, conv1_2_bias,
                           conv2_1_weight, conv2_1_bias, conv2_2_weight, conv2_2_bias,
                           conv3_1_weight, conv3_1_bias, conv3_2_weight, conv3_2_bias, conv3_3_weight, conv3_3_bias,
                           conv4_1_weight, conv4_1_bias, conv4_2_weight, conv4_2_bias, conv4_3_weight, conv4_3_bias,
                           conv5_1_weight, conv5_1_bias, conv5_2_weight, conv5_2_bias, conv5_3_weight, conv5_3_bias,
                           fc6_weight, fc6_bias)
    return b


def get_siamese_test():
    data = mx.sym.Variable('data')
    data_siamese = mx.sym.Variable('data_siamese')
    # label = mx.symbol.Variable(name='label')
    # label = mx.symbol.Reshape(data=label, shape=(-1, 1), name='label_reshape')
    conv1_1_weight = mx.sym.Variable('conv1_1_weight')
    conv1_1_bias = mx.sym.Variable('conv1_1_bias')
    conv1_2_weight = mx.sym.Variable('conv1_2_weight')
    conv1_2_bias = mx.sym.Variable('conv1_2_bias')
    conv2_1_weight = mx.sym.Variable('conv2_1_weight')
    conv2_1_bias = mx.sym.Variable('conv2_1_bias')
    conv2_2_weight = mx.sym.Variable('conv2_2_weight')
    conv2_2_bias = mx.sym.Variable('conv2_2_bias')
    conv3_1_weight = mx.sym.Variable('conv3_1_weight')
    conv3_1_bias = mx.sym.Variable('conv3_1_bias')
    conv3_2_weight = mx.sym.Variable('conv3_2_weight')
    conv3_2_bias = mx.sym.Variable('conv3_2_bias')
    conv3_3_weight = mx.sym.Variable('conv3_3_weight')
    conv3_3_bias = mx.sym.Variable('conv3_3_bias')
    conv4_1_weight = mx.sym.Variable('conv4_1_weight')
    conv4_1_bias = mx.sym.Variable('conv4_1_bias')
    conv4_2_weight = mx.sym.Variable('conv4_2_weight')
    conv4_2_bias = mx.sym.Variable('conv4_2_bias')
    conv4_3_weight = mx.sym.Variable('conv4_3_weight')
    conv4_3_bias = mx.sym.Variable('conv4_3_bias')
    conv5_1_weight = mx.sym.Variable('conv5_1_weight')
    conv5_1_bias = mx.sym.Variable('conv5_1_bias')
    conv5_2_weight = mx.sym.Variable('conv5_2_weight')
    conv5_2_bias = mx.sym.Variable('conv5_2_bias')
    conv5_3_weight = mx.sym.Variable('conv5_3_weight')
    conv5_3_bias = mx.sym.Variable('conv5_3_bias')
    fc6_weight = mx.sym.Variable('fc6_weight')
    fc6_bias = mx.sym.Variable('fc6_bias')
    rois = mx.symbol.Variable(name='rois')
    rois = mx.symbol.Reshape(data=rois, shape=(-1, 5))
    rois_siamese = mx.symbol.Variable(name='rois_siamese')
    rois_siamese = mx.symbol.Reshape(data=rois_siamese, shape=(-1, 5))
    b1 = get_single_branch(data, rois,
                           conv1_1_weight, conv1_1_bias, conv1_2_weight, conv1_2_bias,
                           conv2_1_weight, conv2_1_bias, conv2_2_weight, conv2_2_bias,
                           conv3_1_weight, conv3_1_bias, conv3_2_weight, conv3_2_bias, conv3_3_weight, conv3_3_bias,
                           conv4_1_weight, conv4_1_bias, conv4_2_weight, conv4_2_bias, conv4_3_weight, conv4_3_bias,
                           conv5_1_weight, conv5_1_bias, conv5_2_weight, conv5_2_bias, conv5_3_weight, conv5_3_bias,
                           fc6_weight, fc6_bias)
    b2 = get_single_branch(data_siamese, rois_siamese,
                           conv1_1_weight, conv1_1_bias, conv1_2_weight, conv1_2_bias,
                           conv2_1_weight, conv2_1_bias, conv2_2_weight, conv2_2_bias,
                           conv3_1_weight, conv3_1_bias, conv3_2_weight, conv3_2_bias, conv3_3_weight, conv3_3_bias,
                           conv4_1_weight, conv4_1_bias, conv4_2_weight, conv4_2_bias, conv4_3_weight, conv4_3_bias,
                           conv5_1_weight, conv5_1_bias, conv5_2_weight, conv5_2_bias, conv5_3_weight, conv5_3_bias,
                           fc6_weight, fc6_bias)
    # diff = b1 - b2
    # square_dist = mx.sym.sum(mx.symbol.square(diff), axis=1, keepdims=1)
    #
    # return mx.symbol.Group([square_dist, label])
    return mx.symbol.Group([b1, b2])

if __name__ == '__main__':
    net = get_siamese_test()
    input_shapes = {"data": (1, 3, 480, 640), "data_siamese": (1, 3, 480, 640), "rois": (1, 8, 5),
                    "rois_siamese": (1, 8, 5)}
    dot = mx.viz.plot_network(symbol=net,
                              shape=input_shapes,
                              node_attrs={"shape": 'rect', "fixedsize": 'false'})
    dot.render('output.gv', view=True)