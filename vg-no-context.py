import os
import cv2
import torch
import string
import base64
import requests
import argparse
import numpy as np
import accelerate
import bitsandbytes
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from matplotlib.patches import Rectangle
from sklearn.metrics import pairwise_distances
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

stop_line_prompt = """
[INST] <image>\n
A stop line is a single white line painted on the road at intersections where traffic must stop.
It shows drivers where to halt their vehicles. 
Which labeled images represent stop line?[/INST]
"""
raised_table_prompt="""
A raised table usually covers the entire width of the crosswalk. 
It is typically painted with triangular arrows in white color.
Which labeled images represent raised table?[/INST]
"""

def iou_cal(gt, pred):
    """
    Calculate IoU metric.
    """    
    return np.logical_and(gt, pred).sum()/np.logical_or(gt, pred).sum() 

def mask_filtering(img, masks, obj, colors_to_remove, colors_to_stay):
    """
    Filter segmented masks.
    """
    filtered_masks = []
    # filter based on area
    if obj=='raised_table':
        masks = [m for m in masks if m['area']>400]
    else:
        masks = [m for m in masks if m['area']>200]
        
    # filter based on color
    for mask in masks:
        masked_color = img[mask['segmentation']].mean(axis=0).reshape(1,-1)
        remove_dist = pairwise_distances(colors_to_remove, masked_color).min()
        stay_dist = pairwise_distances(colors_to_stay, masked_color).min()
        if remove_dist>stay_dist:
            filtered_masks.append(mask)
            
    return filtered_masks

def mask_visualization(img, masks, output_path,save_name):
    """
    Generate visualization containing segments.
    """
    
    ## visualize potential candidates
    col = 5
    row = len(masks)//col+1*(len(masks)%col>0)
    fig, axs = plt.subplots(row, col, figsize=(col*3, row*3))
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def write_completion_request(prompt, base64_image):
    """
    Compose completion request.
    """
    
    completion = {
      "model": "gpt-4-turbo-2024-04-09",
      "messages": [
          {"role": "user",
           "content": [
               {"type": "text", "text": prompt},
               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
           ]}
      ],
      "max_tokens": 200
    }
    return completion

def post_processing(response, masks, output_path, save_name):
    """
    Extract bounding box.
    """
    try:
        response = response.translate(str.maketrans('', '', string.punctuation))
        labels = [int(l) for l in response.split() if l.isnumeric()]        
        bboxes = [masks[l]['bbox'] for l in labels]
        masking = [masks[l]['segmentation'] for l in labels]        
        x1 = masking[0]
        for i in range(1, len(masking)):
            x2 = masking[i]
            x1 = np.logical_or(x1, x2)
        masking=x1
    except:
        labels = [-1]
        bboxes = [[0,0,0,0]]
        masking = np.zeros(masks[0]['segmentation'].shape)    

    np.save(output_path+save_name+'_bbox.npy', bboxes)
    np.save(output_path+save_name+'_masking.npy', masking)
    
    return masking

def final_visualization(img, masking, output_path,save_name):
    plt.figure(figsize=(3.36,3.36))
    plt.imshow(img)
    ax = plt.gca()
    ax.set_autoscale_on(False)    
    img_mask = np.ones((masking.shape[0], masking.shape[1], 4))
    img_mask[:,:,3] = 0
    img_mask[masking==1] = np.concatenate([[1,0,0], [0.5]])
    ax.imshow(img_mask)
    plt.axis('off')
    plt.savefig(output_path+save_name+'_masking.png', bbox_inches='tight', dpi=600, pad_inches=0)
    plt.close()     
    
def main():        
        
    #-------------------------
    # arguments
    #-------------------------
    print('-- Load Parameters...')    
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', required=True, help='stop_line or raised_table')
    parser.add_argument('--api_key', required=True, help='stop_line or raised_table')
    args = parser.parse_args()
    obj = args.object
    api_key = args.api_key    
    method='vg-no-context'
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    image_size = (336,336)
    image_path = f'images/{obj}/'
    image_names = os.listdir(image_path)
    image_names = [name for name in image_names if ((name.endswith('.png') & ('masking' not in name)))]
    output_path = f'outputs/{method}/{obj}/'
    api_web = "https://api.openai.com/v1/chat/completions"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if obj == "stop_line":
        prompt = stop_line_prompt
    else:
        prompt = raised_table_prompt        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
        
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
            [220,220,220],[211,211,211],[192,192,192],[169,169,169]
        ])
    else:
        colors_to_stay=np.array([
            # white colors
            [255,255,255],[255,250,250],[245,255,250],[240,255,255],[248,248,255],[245,245,245],        
            # silver colors
            [220,220,220],[211,211,211],[192,192,192],[169,169,169],[128,128,128]
        ])        
    
    print('-- Load SAM Model...')    
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=32)
    
    # ---------------
    # iterate image
    # --------------- 
    print('-- Start Annotating...')      
    for image_name in tqdm(image_names):
    
        save_name = image_name.split('.')[0]
        img = np.array(Image.open(image_path+image_name).convert('RGB').resize((336,336)))
        masks = mask_generator.generate(img)
        filtered_masks = mask_filtering(img, masks, obj, colors_to_remove, colors_to_stay)
        mask_visualization(img, filtered_masks, output_path, save_name)
        base64_image = encode_image(output_path+save_name+'_candidates.png')
        completion = write_completion_request(prompt, base64_image)
        response = requests.post(api_web, headers=headers, json=completion)
        response = response.json()['choices'][0]['message']['content']
        masking = post_processing(response, filtered_masks, output_path, save_name)
        final_visualization(img, masking, output_path, save_name)
    print('-- Annotation Done...') 
    
    # --------------------
    # evaluation
    # --------------------
    print('-- Start Evaluating...')         
    files = os.listdir(image_path)        
    files = [file for file in files if 'masking.npy' in file]
    iou = []
    for file in files:
        gt = np.load(image_path+file)
        pred = np.load(output_path+file)
        iou.append(iou_cal(gt,pred))
    print(f'-- {obj.upper()}, Zero Shot, IoU: {np.mean(iou)}')
    np.save(output_path+'IoU_metrix.npy', iou)
    
if __name__ == "__main__":
    main()
