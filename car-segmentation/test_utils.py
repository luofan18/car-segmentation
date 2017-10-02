"""
Test utils for valib padding
"""

def pad_image(image, pad_width, mode='symmetric'):
    """
    Padding the image for prediction, the mode is 
    same as the one in numpy.pad
    
    # Args 
    image:      PIL Image or array
    pad_width:  the width to pad around the image, integer or tuple represent
                ((h, h), (w, w), (channel, channel))
    
    # Return
                PIL Image padded
    """
    array = img_to_array(image)
    if type(pad_width) is int:
        padding = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))
    else:
        assert type(pad_width) in [tuple, list]
        padding = pad_width
    padded_array = np.pad(array, padding, mode=mode)
    padded_image = array_to_img(padded_array)
    return padded_image

def predict_on_patch(model, image, pad_width, mode, image_size, input_size, output_size, verbose=0):
    """
    Predict the mask patch by patch since the valid padding is used in train
    
    # Args:
    image:      PIL Image
    mode:       Method to pading the image. The same as np.pad
    image_size: The size to perform prediction, (w, h). e.g (906, 640)
    input_size: The size of image to feed into the network
    output_size:The output mask size, which is smaller than input because the edge of the 
                input is discard gradually
    
    """
    image = imresize(image, image_size[::-1])
    if pad_width != 0:
        image = pad_image(image, pad_width, mode)
    array = img_to_array(image) / 255
    mask = np.zeros(image_size[::-1])
    h, w = array.shape[0:2]
    mask_h, mask_w = mask.shape
    
    patch_width = output_size[0]
    patch_high = output_size[1]
    assert patch_width == patch_high, 'Output_size must be square'
    
    input_patch_width = input_size[0]
    input_patch_high = input_size[1]
    assert input_patch_width == input_patch_high, 'Input_size must be square'
    
    if verbose:
        plt.figure()
        plt.imshow(image)
        plt.show()
    
    # Function for judge the position of the patch
    def _is_last_col():
        return patch_width * i + input_patch_width > w
    def _is_last_row():
        return patch_high * j + input_patch_high > h
    
    # Predict patch by patch
    for i in range(int(np.ceil((w - input_patch_width) / patch_width)) + 1):
        for j in range(int(np.ceil((h - input_patch_high) / patch_high)) + 1):
            
            xs = i * patch_width
            xe = xs + input_patch_width
            ys = j * patch_high
            ye = ys + input_patch_high
            
            if (_is_last_row()):
                ys = h - input_patch_high - 1
                ye = h - 1
            if (_is_last_col()):
                xs = w - input_patch_width - 1
                xe = w - 1
                
            if verbose:
                print ('Now processing ({}, {}), ({}, {}).'.
                       format(xs, ys, xe, ye))
            
            # Predict the patch
            patch = array[ys:ye, xs:xe]
            patch = patch[None,:,:,:]
            output_patch = model.predict(patch, batch_size=1)
            
            if verbose:
                show_image_and_mask(patch[0], output_patch[0,:,:,0])
            
            # Fill the mask
            xs = i * patch_width
            xe = (i + 1) * patch_width
            ys = j * patch_high
            ye = (j + 1) * patch_high
            if (_is_last_row()):
                ys = mask_h - patch_high - 1
                ye = mask_h - 1
            if (_is_last_col()):
                xs = mask_w - patch_width - 1
                xe = mask_w - 1
            mask[ys:ye, xs:xe] = output_patch[0,:,:,0]
            
    return mask

def mirror_average_prediction(model, image, pad_width, mode, image_size, input_size, output_size, threshold=None):
    """
    Predict the mask, average the result with mirror of the image
    
    # Args:
    The same as predict on patch
    threshold:      used to decide whether the whether the pixel is 1 or 0. 
                    Default is None. Set to 0.5 or other value when deployed. 
    """
    image2 = image.transpose(Image.FLIP_LEFT_RIGHT)
    mask = predict_on_patch(model, image, pad_width, mode, image_size, input_size, output_size)
    mask2 = predict_on_patch(model, image2, pad_width, mode, image_size, input_size, output_size)
    mask = mask + np.fliplr(mask2)
    mask /= 2.
    if threshold:
        mask[mask>=threshold] = 1
        mask[mask<threshold] = 0
    return mask

def cal_dice_coef(mask1, mask2):
    """
    Calculate dice coefficient
    
    mask1:    Groundtruth lable, size + channel
    mask2:    Predicted lable, 
    """
    if mask1.max() > 1:
        mask1 = (mask1 / 255.)[:,:,0]
    mask1 = mask1.astype(int)
    threshold = 0.5
    if threshold:
        mask2[mask2>=threshold] = 1
        mask2[mask2<threshold] = 0
        mask2 = mask2.astype(int)
    intersection = mask1 * mask2
    return 2. * intersection.sum() / (mask1 + mask2).sum()

def val_dice_coef(model, image_pairs, 
                  pad_width, image_size, input_size, output_size, val_step=0):
    """
    Calculate dice coefficient on the generated validation data. This data is the
    same as the ones used in training. Hence no need to pad and can be predict 
    use model's original predict method.
    
    Will write the prediction to disk.
    
    # Args:
    image_size:      The size on which the image is cropped from
    input_size:      The input size of image, the cropped ones
    output_size:     The size of mask, smaller than input_size
    
    # Return:
    mean_dice
    mask_pred:       Lsit of predicted mask
    """
    dices = []
    if val_step == 0:
        assert len(image_pair) == 2, 'Image_pair should be a tuple.'
        images, masks = image_pairs
        for image, mask in tqdm(zip(images, masks)):
            mask_pred = model.predict(image[None,:,:,:])
            mask_pred = mask_pred[0,:,:,:]
            dice = cal_dice_coef(mask, mask_pred)
            dices.append(dice)
            # Show image, mask, prediction and difference
            show_4_images(image, mask[:,:,0], mask_pred[:,:,0], 
                          mask[:,:,0]-mask_pred[:,:,0])
    else:
        for i in tqdm(range(val_step)):
            val_images, val_masks = next(val_gen)
            mask_preds = model.predict(val_images)
            for image, mask, mask_pred in zip(val_images, val_masks, mask_preds):
                dice = cal_dice_coef(mask, mask_pred)
                dices.append(dice)
                show_4_images(image, mask[:,:,0], mask_pred[:,:,0], 
                              mask[:,:,0]-mask_pred[:,:,0])
    return dices

def sort_dices(dices, image_names):
    # Sort the dices coef and image-names according to the error
    return sorted(dices), [image_name for _, image_name in sorted(zip(dices, image_names))]

def show_top_error(model, imin, imax, dices, validation_images_sort, 
                   input_dims, transforms, save=True, show=False):
    # First, sort the dices coef and image name, then use this function to 
    # visualize the result
    for ix_ in range(imin, imax):
        print (validation_images_sort[ix_])
        print (dices_sort[ix_])

        val_gen_ = data_gen_small(data_dir, mask_dir, [validation_images_sort[ix_]], 1, 
                                 input_dims, transforms=transforms, in_order=True)
        image_, mask_ = next(val_gen_)

        show_image_and_mask(image_[0], mask_[0,:,:,0])

        mask_pred_ = model.predict(image_)

        print (cal_dice_coef(mask_[0], mask_pred_[0]))

        mask_pred_[mask_pred_>0.5] = 1
        mask_pred_[mask_pred_<0.5] = 0

        save_to = (tmp_dir + 'result/' + 'top-' + str(ix_ + 1).zfill(3) + 
                '_error_' + validation_images_sort[ix_])
        
        show_diff(image_[0], mask_[0,:,:,0], mask_pred_[0,:,:,0], 
                  save_to=save_to, show=show)

def val_dice_coef2(model, validation_images, data_dir, mask_dir, 
                   pad_width, mode, image_size, input_size, output_size):
    """
    Calculate dice coefficient on validation set on give size. The result will
    be slightly different form original size. But the variance will not be too
    much.
    
    Will save the prediction to disk 
    
    The prediction will be performed patch by patch on each images and combined 
    to 1. Mirror average is used to boost the performance. 
    
    # Args:
    Almost the same as val_dice_coef
    
    # Return:
    dice_mean
    mask_pred:        List of predicted masks
    """
    dices = []
    masks_pred = []
    for image in tqdm(validation_images):
        image, mask = read_image_and_mask(data_dir, mask_dir, image)
        mask = imresize(mask, image_size[::-1]) / 255.
        mask_pred = mirror_average_prediction(model, image, 
                                               pad_width, image_size, input_dims, output_dims)
        # Show image, mask, prediction and difference
        show_4_images(image, mask[:,:,0], mask_pred[:,:], mask[:,:,0]-mask_pred[:,:])
        dices.append(cal_dice_coef(mask[:,:,0], mask_pred))
    with open(tmp_dir + 'masks_pred.pickel') as f:
        pickle.dump(masks_pred, f)
    dices = np.array(dices)
    return dices, mask_pred

def rle_encode(mask):
    pixels = mask.flatten()
    
    # Set begining and end to 0 for simplicity
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs [1::2] = runs[1::2] - runs[:-1:2]
    
    return runs

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def generate_one_record(image_names, mask): 
    rle_code = rle_encode(mask)
    record = image_names + ',' + rle_to_string(rle_code)
    return record

def generate_test(model, test_images, test_dir, pad_width, 
                  mode, image_size, input_size, output_size):
    
    report_per = 100
    
    submission_path = tmp_dir + 'submission.csv'
    progress_path = tmp_dir + 'progress.log'
    
    if os.path.isfile(submission_path):
        os.remove(submission_path)
    if os.path.isfile(submission_path):
        os.remove(progress_path)
    
    with open(submission_path, 'w') as csvfile:
        csvfile.write('img,rle_mask\n')
        
    for i, image_name in enumerate(test_images):
        image = load_img(test_dir + image_name)
        mask = mirror_average_prediction(model, image, 
                                              pad_width, mode, image_size, input_dims, output_dims)
        mask = threshold_mask(mask)
        with open(submission_path, 'a') as csvfile:
            csvfile.write(generate_one_record(image_name, mask) + '\n')
            
        if i % report_per == 0:
            with open(progress_path, 'w') as progress_file:
                progress_file.write('progress is {}% \n'.format(i / len(test_images) * 100))