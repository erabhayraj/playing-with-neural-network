import numpy as np
from helper import *

def movePatchOverImg(image, filter_size, apply_filter_to_patch):
    #ADD CODE HERE
    return output_image

def detect_horizontal_edge(image_patch):
    #ADD CODE HERE
    return outputval

def detect_vertical_edge(image_patch):
    #ADD CODE HERE
    return outputval

def detect_all_edges(image_patch):
    #ADD CODE HERE
    return outputval

def remove_noise(image_patch):
    #ADD CODE HERE
    return outval

def create_gaussian_kernel(size, sigma):
    #ADD CODE HERE
    return output_kernel

def gaussian_blur(image_patch):
    #ADD CODE HERE
    return outputval

def unsharp_masking(image, scale):
    #ADD CODE HERE
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

#TASK 2
noisyimg=load_image("noisycutebird.png")
filter_size=3 #You may change this to any appropriate odd number
denoised = movePatchOverImg(noisyimg, filter_size, remove_noise)
save_image("denoised.png",denoised)

#TASK 3
scale= #You may use any appropriate positive number (ideally between 1 and 3)
save_image("unsharpmask.png",unsharp_masking(img,scale))
