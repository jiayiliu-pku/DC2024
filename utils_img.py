import os
import SimpleITK as sitk
import numpy as np
import cv2

def projection_with_angles(
        imagefile,
        savepath        = '',
        rotation_angles = [np.pi/2],
        rotation_axis   = [0,0,1],
        ptype           = sitk.MaximumProjection,
        paxis           = 0,
        pad_val         = 0,
    ):
    """
    Generate Maximum Projection of SUV image.
    """
    
    image = sitk.ReadImage(imagefile)

    rotation_center = image.TransformContinuousIndexToPhysicalPoint([(index-1)/2.0 for index in image.GetSize()])
    rotation_transform = sitk.VersorRigid3DTransform()
    rotation_transform.SetCenter(rotation_center)

    #Compute bounding box of rotating volume and the resampling grid structure
    image_indexes = list(zip([0,0,0], [sz-1 for sz in image.GetSize()]))
    image_bounds = []
    for i in image_indexes[0]:
        for j in image_indexes[1]:
            for k in image_indexes[2]:
                image_bounds.append(image.TransformIndexToPhysicalPoint([i,j,k]))

    all_points = []
    for angle in list(rotation_angles):
        rotation_transform.SetRotation(rotation_axis, angle)    
        all_points.extend([rotation_transform.TransformPoint(pnt) for pnt in image_bounds])
    all_points = np.array(all_points)
    min_bounds = all_points.min(0)
    max_bounds = all_points.max(0)
    #resampling grid will be isotropic so no matter which direction we project to
    #the images we save will always be isotropic (required for image formats that 
    #assume isotropy - jpg,png,tiff...)
    new_spc = [np.min(image.GetSpacing())]*3
    new_sz = [int(sz/spc + 0.5) for spc,sz in zip(new_spc, max_bounds-min_bounds)]

    proj_images = []
    for angle in list(rotation_angles):
        rotation_transform.SetRotation(rotation_axis, angle) 
        
        resampled_image = sitk.Resample(image1=image,
                                        size=new_sz,
                                        transform=rotation_transform,
                                        interpolator=sitk.sitkLinear,
                                        outputOrigin=min_bounds,
                                        outputSpacing=new_spc,
                                        outputDirection = [1,0,0,0,1,0,0,0,1],
                                        defaultPixelValue = pad_val, #HU unit for air in CT, possibly set to 0 in other cases
                                        outputPixelType = image.GetPixelID())
        
        proj_image = ptype(resampled_image, paxis)
        
        extract_size = list(proj_image.GetSize())
        extract_size[paxis]=0
        
        proj_images.append(sitk.Extract(proj_image, extract_size))
    
    count = 0
    mip_images = []
    filename = os.path.basename(imagefile).split('_0001.nii')[0]
    for d, ds in enumerate(proj_images):

        img = sitk.GetArrayFromImage(ds)
        rescale_img = norm_img(img)

        if savepath != '':
            cv2.imwrite(f'{savepath}/{filename}.png', rescale_img)
        mip_images.append(rescale_img)

        count += 1

    return mip_images

def norm_img(img, coeffi=10):
    rescale_img = img * coeffi
    rescale_img[rescale_img>255] = 255
    # rescale_img = (img - img.min()) / (img.max() - img.min()) * 255
    return rescale_img

import os
from nnunet.utilities.sitk_stuff import copy_geometry

def extract_lesion(input_file: str, output_file: str, lesion_ids: list):
    """
    Extract lesion
    """
    print(f'Extract lesion: {os.path.basename(input_file)}, {lesion_ids}')
    img_in  = sitk.ReadImage(input_file)
    img_npy = sitk.GetArrayFromImage(img_in)

    mask_npy = np.zeros(img_npy.shape)
    if len(lesion_ids) == 1:
        lesion_id = lesion_ids[0]
        mask_npy[img_npy != lesion_id] = 0
        mask_npy[img_npy == lesion_id] = 1
    else:
        for lesion_id in lesion_ids:
            mask_npy[img_npy == lesion_id] = 1

    img_out_itk = sitk.GetImageFromArray(mask_npy)
    img_out_itk = copy_geometry(img_out_itk, img_in)
    sitk.WriteImage(img_out_itk, output_file)