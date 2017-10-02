# Metrics
def dice_coef(y_true, y_pred):
    smooth = 1e-5
    
    y_true = tf.round(tf.reshape((y_true), [-1]))
    y_pred = tf.round(tf.reshape((y_pred), [-1]))
    
    isct = tf.reduce_sum(y_true * y_pred)
    
    return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

# Loss
def tf_detect_edge(y_true, kernel_size=3, verbose=0):
    """
    Use a small 3x3 kernel to detect edge in the mask (y_true)
    
    # Args:
     y_true:      (batch_size, high, width, 1)
    """
    y_true_round = tf.round(y_true)
    edge_detector = tf.reshape(shape=(kernel_size, kernel_size, 1, 1), 
                               tensor=tf.ones(kernel_size**2))
    y_true_conv = tf.nn.conv2d(y_true_round, edge_detector, (1,1,1,1), 
                               'SAME', name='edge_detector')
    y_true_conv = tf.round(y_true_conv)
    condition1 = tf.equal(y_true_conv, 0.)
    condition2 = tf.equal(y_true_conv, kernel_size**2.)
    condition = tf.logical_or(condition1, condition2)
    edge = tf.where(condition, 
                    tf.zeros_like(y_true_conv), tf.ones_like(y_true_conv))
    if verbose:
        return y_true_conv ,edge
    else:
        return edge

def tf_gaussian_blur(edge, sigma=5):
    """
    Blur the edge to weight the boundry loss in gaussian
    """
    from scipy.ndimage import gaussian_filter
    kernel = np.zeros((sigma*8+1, sigma*8+1))
    kernel[sigma*4, sigma*4] = 1
    kernel = gaussian_filter(kernel, sigma)
    kernel = tf.constant(dtype=tf.float32, value=kernel)
    kernel = tf.reshape(shape=(sigma*8+1, sigma*8+1, 1, 1), tensor=kernel)
    blurred = tf.nn.conv2d(edge, kernel, (1,1,1,1), 'SAME', name='blur')
    return blurred

def test_tf_blur(mask):
    """
    jointly test the tf_dectect_edge and tf_gaussian_blur
    """
    ph_mask = tf.placeholder(dtype=tf.float32, shape=(None, None, None, None))
    y_true_conv, edge = tf_detect_edge(ph_mask, verbose=1)
    blurred = tf_gaussian_blur(edge)
    with tf.Session() as sess:
        out = sess.run([y_true_conv, edge, blurred], feed_dict={ph_mask: mask})
    np.savetxt('edge.txt', out[0][0][:,:,0])
    plt.figure(figsize=(20, 16))
    plt.imshow(out[0][0][:,:,0])
    plt.show()
    plt.figure(figsize=(20, 16))
    plt.imshow(out[1][0][:,:,0])
    plt.show()
    plt.figure(figsize=(20, 16))
    plt.imshow(out[2][0][:,:,0])
    plt.show()    
    
def weighted_binary_crossentropy(y_true, y_pred):
    """
    The boundry penalised loss
    
    # Args:
    y_true:      (batch_size, high, width, channel)
    """
    class_weight = 1
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1-epsilon)
    # Calculate weight
    edge = tf_detect_edge(y_true)
    blurred = tf_gaussian_blur(edge)
    boundry_weight = 10
    weight_map = 1 + (class_weight - 1) * y_true + boundry_weight * blurred
    crossentropy = - y_true * tf.log(y_pred) - (1 - y_true) * tf.log(1 - y_pred)
    weighted_loss = K.mean(crossentropy * weight_map, axis=-1)
    
    return weighted_loss

def test_weighted_loss():
    y_true = np.random.choice([0,1], size=(1,4,4,1))
    y_pred = np.random.uniform(size=(1,4,4,1))
    ph_true = tf.placeholder(dtype=tf.float32, shape=(None, None, None, None))
    ph_pred = tf.placeholder(dtype=tf.float32, shape=(None, None, None, None))
    op_loss = weighted_binary_crossentropy(ph_true, ph_pred)
    
    from keras.losses import binary_crossentropy
    op_naive_loss = binary_crossentropy(ph_true, ph_pred)
    with tf.Session() as sess:
        loss = sess.run([op_loss, op_naive_loss], feed_dict={ph_true: y_true, ph_pred: y_pred})
        print ('weigted loss')
        print (loss[0])
        print ('naive loss')
        print (loss[1])
        loss = sess.run(K.mean(K.binary_crossentropy(ph_true, ph_pred), axis=-1), 
                        feed_dict={ph_true: y_true, ph_pred: y_pred})
        print ('naive loss 2')
        print (loss)
        loss = sess.run(K.mean(-ph_true * tf.log(ph_pred) - (1 - ph_true) * tf.log(1 - ph_pred), axis=-1), 
                        feed_dict={ph_true: y_true, ph_pred: y_pred})
        print ('loss with prob 2')
        print (loss)