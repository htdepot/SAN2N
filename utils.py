import os
import os.path as osp
import torch
from time import sleep
import nibabel as nib
import numpy as np
from tqdm import tqdm
import math
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def restore_img(subject_id, prediction, im, image_file_path, mask_path, gt_path, is_generate_image):
    prediction = prediction[0]
    mask = nib.load(mask_path)
    dmri_data = nib.load(gt_path).get_fdata()

    mask = mask.get_fdata()
    back_pre = prediction

    if is_generate_image:
        affine = np.eye(4)
        if not os.path.isdir(image_file_path + '/' + subject_id):
            os.makedirs(image_file_path + '/' + subject_id)
        img_prediction = nib.Nifti1Image(back_pre, affine)
        nib.save(img_prediction, image_file_path + '/' + subject_id + '/back_denoised.nii.gz')

    x_size = mask.shape[0]
    y_size = mask.shape[1]
    z_size = mask.shape[2]

    metrics_prediction = []
    metrics_dmri_data = []

    for xx in range(0, x_size, 1):
        for yy in range(0, y_size, 1):
            for zz in range(0, z_size, 1):
                if mask[xx, yy, zz] > 0:
                    metrics_prediction.append(prediction[xx, yy, zz, :])
                    metrics_dmri_data.append(dmri_data[xx, yy, zz, :])

                else:
                    prediction[xx, yy, zz, :] = 0
                    dmri_data[xx, yy, zz, :] = 0

    metrics_dmri_data = np.array(metrics_dmri_data).reshape(-1)
    metrics_prediction = np.array(metrics_prediction).reshape(-1)


    if is_generate_image:
        if not os.path.isdir(image_file_path + '/' + subject_id):
            os.makedirs(image_file_path + '/' + subject_id)
        affine = np.eye(4)
        img_prediction = nib.Nifti1Image(prediction, affine)
        nib.save(img_prediction, image_file_path + '/' + subject_id + '/denoised.nii.gz')

        img_prediction = nib.Nifti1Image(dmri_data, affine)
        nib.save(img_prediction, image_file_path + '/' + subject_id + '/clean.nii.gz')

        img_prediction = nib.Nifti1Image(im, affine)
        nib.save(img_prediction, image_file_path + '/' + subject_id + '/noisy.nii.gz')

        resdiual = np.sqrt((prediction - dmri_data) ** 2)
        img_prediction = nib.Nifti1Image(resdiual, affine)
        nib.save(img_prediction, image_file_path + '/' + subject_id + '/resdiual.nii.gz')

        noise_resdiual = np.sqrt((prediction - im) ** 2)
        img_prediction = nib.Nifti1Image(noise_resdiual, affine)
        nib.save(img_prediction, image_file_path + '/' + subject_id + '/noise_resdiual.nii.gz')

        print('save img successful')

    max = int(metrics_dmri_data.max())
    psnr_value = psnr(metrics_dmri_data, metrics_prediction, data_range=max)
    ssim_value = ssim(metrics_dmri_data, metrics_prediction, data_range=max)
    rmse_value = np.sqrt(mse(metrics_dmri_data, metrics_prediction))


    return psnr_value, ssim_value, rmse_value
