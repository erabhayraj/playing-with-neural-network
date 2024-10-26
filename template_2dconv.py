import numpy as np
from helper import *
def movePatchOverImg(image, filter_size, apply_filter_to_patch):

    # changing into grayscale
    if image.ndim==3:
        image = np.dot(image[...,:3],[ 0.2989,0.5870,0.1140])
    
    # padding image so that output be of same dimensions
    pad_size = (filter_size-1)//2                                               # as n+2p-f+1=n => 2p=f-1 => p=(f-1)/2
    padded_img = np.pad(image,pad_width=pad_size,mode="constant",constant_values=0)

    output_image=np.zeros(image.shape)
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image[i][j]=apply_filter_to_patch(padded_img[i:i+filter_size,j:j+filter_size])
    return output_image

def detect_horizontal_edge(image_patch):
    kernel=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    outputval=np.sum(image_patch*kernel)
    return outputval

def detect_vertical_edge(image_patch):
    kernel=np.array([[-1,0,-1],[-2,0,2],[1,0,1]])
    outputval=np.sum(image_patch*kernel)
    return outputval

def detect_all_edges(image_patch):
    kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    outputval=np.sum(image_patch*kernel)
    return outputval

def remove_noise(image_patch):
    kernel=np.array([[1,1,1],[1,1,1],[1,1,1]])
    outputval=np.median(image_patch*kernel)
    return outputval

def create_gaussian_kernel(size, sigma):
    output_kernel=np.array(np.zeros((size,size)))
    for i in range(size):
        for j in range(size):
            offset=(size-1)/2
            iterm=(i-offset)**2
            jterm=(j-offset)**2
            denom=2*(sigma**2)
            expterm=(iterm+jterm)/denom
            output_kernel[i-1][j-1]=(1/(2*np.pi*(sigma**2)))*np.exp(-expterm)
    return output_kernel

def gaussian_blur(image_patch):
    kernel=create_gaussian_kernel(25,1)
    outputval=np.sum(image_patch*kernel)
    print(str(cnt)+" done")
    return outputval

def unsharp_masking(image, scale):
    if image.ndim==3:
        image = np.dot(image[...,:3],[ 0.2989,0.5870,0.1140])
    gaussianblurred=movePatchOverImg(image,25,gaussian_blur)
    subtracted=image - gaussianblurred
    out=image + scale * subtracted
    out=np.clip(out,0,255)
    return out

#TASK 1  
img=load_image("cutebird.png")
filter_size=3 #You may change this to any appropriate odd number
hori_edges = movePatchOverImg(img, filter_size, detect_horizontal_edge)
save_image("hori.png",hori_edges)
filter_size=3 #You may change this to any appropriate odd number
vert_edges = movePatchOverImg(img, filter_size, detect_vertical_edge)
save_image("vert.png",vert_edges)
filter_size=3 #You may change this to any appropriate odd number
all_edges = movePatchOverImg(img, filter_size, detect_all_edges)
save_image("alledge.png",all_edges)

# #TASK 2
noisyimg=load_image("noisycutebird.png")
filter_size=3 #You may change this to any appropriate odd number
denoised = movePatchOverImg(noisyimg, filter_size, remove_noise)
save_image("denoised.png",denoised)


#TASK 3
scale=2 #You may use any appropriate positive number (ideally between 1 and 3)
save_image("unsharpmask.png",unsharp_masking(img,scale))
