#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from objRemovalDrawing import ObjectRemove
from models.deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from lama_cleaner.model_manager import ModelManager
##################################
#get image path from command line#
##################################
parser = argparse.ArgumentParser()
parser.add_argument("image")
args = parser.parse_args()
image_path = args.image

######################################################
#creating Mask-RCNN model and load pretrained weights#
######################################################
for f in os.listdir('ObjectRemoval/src/models'):
    if f.endswith('.pth'):
        deepfill_weights_path = os.path.join('ObjectRemoval/src/models', f)
print("Creating rcnn model")
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()
rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False)
rcnn = rcnn.eval()

#########################
#create inaptining model#
#########################
print('Creating inpaint model')
deepfill = Generator(checkpoint=deepfill_weights_path, return_flow=True)
######################
#create ObjectRemoval#
######################
model = ObjectRemove(segmentModel=rcnn,
                        rcnn_transforms=transforms, 
                        inpaintModel=deepfill, 
                        image_path=image_path )
#####
#run#
#####
output = model.run()

#################
#display results#
#################
tensor_img = model.image_orig[0]

# If it's on GPU, move it to CPU
if tensor_img.is_cuda:
    tensor_img = tensor_img.cpu()

# Convert to NumPy array
np_img = tensor_img.detach().numpy()

# Transpose to HWC format
np_img = np.transpose(np_img, (1, 2, 0))

# Convert RGB to BGR
img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
boxed = cv2.rectangle(img, (model.box[0], model.box[1]),(model.box[2], model.box[3]), (0,255,0),2)
boxed = cv2.cvtColor(boxed,cv2.COLOR_BGR2RGB)

fig,axs = plt.subplots(1,3,layout='constrained')
axs[0].imshow(boxed)
axs[0].set_title('Original Image Bounding Box')
axs[1].imshow(model.image_masked.permute(1,2,0).detach().numpy())
axs[1].set_title('Masked Image')
axs[2].imshow(output)
axs[2].set_title('Inpainted Image')
plt.show()