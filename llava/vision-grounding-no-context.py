import cv2
import os
import torch
import argparse
import string
import numpy as np
import accelerate
import bitsandbytes
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import pairwise_distances
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

stop_line_prompt = """
[INST] <image>\nA stop line indicates to drivers where to stop the car when approaching the intersection. It is a straight line and painted in white color. Which images contain a stop line? Return the labeled image number.[/INST]
"""
raised_table_prompt = """
[INST] <image>\nA raised table in the road is usually painted in white color with traingular arrows. Which images contain a raised table? Return the labeled image number.[/INST]
"""

def miou_cal(gt, pred):
    return np.logical_and(gt, pred).sum()/np.logical_or(gt, pred).sum()

def mask_visualization(img, masks,output_path,save_name):
    ## visualize potential candidates
    col = 5
    row = len(masks)//col+1*(len(masks)%col>0)
    fig, axs = plt.subplots(row, col, figsize=(col*2, row*2))
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.subplots_adjust(wspace=0.1, hspace=0)
    for r in range(row):
        for c in range(col):

            if row == 1:
                ## outside of range
                if r*5+c >=len(masks):
                    axs[r*5+c].axis('off')
                    continue

                ## extract bounding box
                bbox = masks[r*5+c]['bbox']
                xtl, ytl = int(bbox[0]), int(bbox[1])
                xbr, ybr = int(xtl+bbox[2]), int(ytl+bbox[3])
                axs[c].imshow(img[ytl:ybr, xtl:xbr])
                axs[c].set_title(f'{r*5+c}')
                axs[c].axis('off')

            else:
                ## outside of range
                if r*5+c >=len(masks):
                    axs[r,c].axis('off')
                    continue

                ## extract bounding box
                bbox = masks[r*5+c]['bbox']
                xtl, ytl = int(bbox[0]), int(bbox[1])
                xbr, ybr = int(xtl+bbox[2]), int(ytl+bbox[3])
                axs[r,c].imshow(img[ytl:ybr, xtl:xbr])
                axs[r,c].set_title(f'{r*5+c}')
                axs[r,c].axis('off')
    plt.savefig(output_path+save_name+'_candidates.png', bbox_inches='tight', dpi=600, pad_inches=0)
    plt.close()

def final_visualization(img, bboxes, output_path,image_name):
    plt.figure(figsize=(3.36,3.36))
    plt.imshow(img)
    for bbox in bboxes:
        xtl, ytl = int(bbox[0]), int(bbox[1])
        xbr, ybr = int(xtl+bbox[2]), int(ytl+bbox[3])
        plt.hlines(ytl, xmin=xtl, xmax=xbr, color='red')
        plt.hlines(ybr, xmin=xtl, xmax=xbr, color='red')
        plt.vlines(xtl, ymin=ytl, ymax=ybr, color='red')
        plt.vlines(xbr, ymin=ytl, ymax=ybr, color='red')
        plt.axis('off')
    plt.savefig(output_path+image_name, bbox_inches='tight', dpi=600, pad_inches=0)
    plt.close()
    
def main():
        
    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', required=True, help='stop_line or raised_table') 
    args = parser.parse_args()
    obj = args.object
    
    method='vg-no-context'
    vl_model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    image_size = (336,336)
    image_path = f'images/{obj}/'
    image_names = os.listdir(image_path)
    image_names = [name for name in image_names if ((name.endswith('.png') & ('masking' not in name)))]
    output_path = f'outputs/{method}/{obj}/'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')        
    if obj == "stop_line":prompt = stop_line_prompt
    else:prompt = raised_table_prompt
    
    colors_to_remove=np.array([
        # green colors
        [124,252,0],[127,255,0],[50,205,50],[0,255,0],[34,139,34],[0,128,0],[0,100,0],
        [173,255,47],[154,205,50],[0,255,127],[0,250,154],[144,238,144],[152,251,152],
        [143,188,143],[60,179,113],[32,178,170],[46,139,87],[128,128,0],[85,107,47],[107,142,35],        
        # yellow colors
        [255,228,181],[255,218,185],[238,232,170],[240,230,140],[189,183,107],[255,255,0],
        [128,128,0],[173,255,47],[154,205,50],[255,255,153],[255,255,102],[255,255,51],[255,255,0],
        [204,204,0],[153,153,0],[102,102,0],[51,51,0],        
        # brown colors
        [222,184,135],[210,180,140],[188,143,143],[244,164,96],[218,165,32],[205,133,63],
        [210,105,30],[139,69,19],[160,82,45],[165,42,42],[128,0,0],
        
    ])
    if obj=='stop_line':
        colors_to_stay=np.array([        
            # white colors
            [255,255,255],[255,250,250],[245,255,250],[240,255,255],[248,248,255],[245,245,245],        
            # silver colors
            [220,220,220],[211,211,211],[192,192,192],
        ])
    else:
        colors_to_stay=np.array([        
            # white colors
            [255,255,255],[255,250,250],[245,255,250],[240,255,255],[248,248,255],[245,245,245],        
            # silver colors
            [220,220,220],[211,211,211],[192,192,192],[169,169,169],[128,128,128],[105,105,105],
        ])
    
    
    # ---------------
    # load model
    # ---------------
    processor = LlavaNextProcessor.from_pretrained(vl_model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        vl_model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True
    )    
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=32)

    # ---------------
    # iterate image
    # --------------- 
    for image_name in tqdm(image_names):
    
        # ---------------
        # generate masks
        # ---------------
        save_name = image_name.split('.')[0]
        img = np.array(Image.open(image_path+image_name).convert('RGB').resize((336,336)))
        masks = mask_generator.generate(img)

        # ---------------
        # filter masks
        # ---------------
        filtered_masks = []
        # filter based on area
        if obj=='raised_table':
            masks = [m for m in masks if m['area']>400]

        # filter based on color
        for mask in masks:
            masked_color = img[mask['segmentation']].mean(axis=0).reshape(1,-1)
            remove_dist = pairwise_distances(colors_to_remove, masked_color).min()
            stay_dist = pairwise_distances(colors_to_stay, masked_color).min()
            if remove_dist>stay_dist:
                filtered_masks.append(mask)
            
        # --------------------
        # visualize candidates
        # --------------------
        mask_visualization(img, filtered_masks, output_path, save_name)
        candidates_img = Image.open(output_path+save_name+'_candidates.png').convert('RGB')
        
        # ---------------
        # annotate
        # ---------------
        inputs = processor(prompt, candidates_img, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=50)
        
        # --------------------
        # process bounding box
        # --------------------
        try:
            labels = processor.decode(output[0], skip_special_tokens=True).split('[/INST]')[1]
            labels = labels.translate(str.maketrans('', '', string.punctuation)).split()
            labels = [int(i) for i in labels if i.isnumeric()]
            bboxes = [filtered_masks[l]['bbox'] for l in labels]
            masking = [filtered_masks[l]['segmentation'] for l in labels]
            x1 = masking[0]
            for i in range(1, len(masking)):
                x2 = masking[i]
                x1 = np.logical_or(x1, x2)
            masking=x1
        except:
            labels = [-1]
            bboxes = [[0,0,0,0]]
            masking = np.zeros(image_size)
            
        np.save(output_path+save_name+'_labels.npy', labels)
        np.save(output_path+save_name+'_bbox.npy', bboxes)        
        np.save(output_path+save_name+'_masking.npy', masking)                
    
        # --------------------
        # visualize bounding box
        # --------------------
        final_visualization(img, bboxes, output_path, image_name)        

    # --------------------
    # evaluation
    # --------------------        
    files = os.listdir(image_path)        
    files = [file for file in files if 'masking.npy' in file]
    miou = []
    for file in files:
        gt = np.load(image_path+file)
        pred = np.load(output_path+file)
        miou.append(miou_cal(gt,pred))
    print(f'{obj.upper()}, Zero Shot, mIoU: {np.mean(miou)}')
    
if __name__ == "__main__":
    main()
