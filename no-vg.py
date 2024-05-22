import os
import torch
import string
import base64
import requests
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from matplotlib.patches import Rectangle

stop_line_prompt = """
[INST] <image>\n
A stop line is a single white line painted on the road at intersections where traffic must stop.
It shows drivers where to halt their vehicles. 
Please identify the bounding box of the stop line in the image in the format of (xtl, ytl, xbr, ybr).[/INST]
"""
raised_table_prompt="""
A raised table usually covers the entire width of the crosswalk. 
It is typically painted with triangular arrows in white color.
Please identify the bounding box of the stop line in the image in the format of (xtl, ytl, xbr, ybr).[/INST]
"""

def iou_cal(gt, pred):
    """
    Calculate IoU metric.
    """    
    return np.logical_and(gt, pred).sum()/np.logical_or(gt, pred).sum() 

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

def post_processing(response, output_path, save_name):
    """
    Extract bounding box.
    """
    try:
        response = response.translate(str.maketrans('', '', string.punctuation))
        xtl, ytl, xbr, ybr = [int(l) for l in response.split() if l.isnumeric()]
    except:
        xtl, ytl, xbr, ybr = 0,0,0,0
    bbox = np.array([xtl, ytl, xbr, ybr])
    np.save(output_path+save_name+'_bbox.npy', bbox)
    return bbox

def generate_masking(img, bbox, output_path, save_name, image_size):
    
    """
    Generate binary masking.
    """
    
    save_dir = output_path+save_name
    xtl, ytl, xbr, ybr = bbox
    
    # generate black masking region
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    rect = Rectangle((xtl, ytl), xbr-xtl, ybr-ytl, color="black", angle=0, rotation_point='center')
    ax.add_patch(rect)
    plt.savefig(save_dir+'_masking.png', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # generate binary masking matrix
    img_masking = np.array(Image.open(save_dir+'_masking.png').convert('RGB').resize(image_size))
    masking = np.zeros(image_size)
    masking[np.where(img_masking.sum(axis=-1)==0.0)]=True
    np.save(save_dir+'_masking.npy', masking)
    
    # replot masking region in red
    plt.figure(figsize=(3.36,3.36))
    plt.imshow(img)
    ax = plt.gca()
    ax.set_autoscale_on(False)    
    img_mask = np.ones((masking.shape[0], masking.shape[1], 4))
    img_mask[:,:,3] = 0
    img_mask[masking==1] = np.concatenate([[1,0,0], [0.5]])
    ax.imshow(img_mask)
    plt.axis('off')
    plt.savefig(save_dir+'_masking.png', bbox_inches='tight', dpi=600, pad_inches=0)
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
    method='no-vg'  
    image_size = (336,336)
    image_path = f'images/{obj}/'
    image_names = os.listdir(image_path)
    image_names = [name for name in image_names if ((name.endswith('.png') & ('masking' not in name)))]
    output_path = f'outputs/{method}/{obj}/'
    api_web = "https://api.openai.com/v1/chat/completions"
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
        
    # ---------------
    # iterate image
    # --------------- 
    print('-- Start Annotating...')    
    for image_name in tqdm(image_names):
    
        save_name = image_name.split('.')[0]
        img = Image.open(image_path+image_name).convert('RGB').resize(image_size)
        base64_image = encode_image(image_path+image_name)
        completion = write_completion_request(prompt, base64_image)
        response = requests.post(api_web, headers=headers, json=completion)
        response = response.json()['choices'][0]['message']['content']
        bbox = post_processing(response,output_path, save_name)
        generate_masking(img, bbox, output_path, save_name, image_size)
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
