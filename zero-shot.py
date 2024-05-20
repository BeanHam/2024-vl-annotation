import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from matplotlib.patches import Rectangle
import accelerate
import bitsandbytes
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

stop_line_prompt = """
[INST] <image>\nA stop line indicates to drivers where to stop the car when approaching the intersection. It is usually painted in white color. Please identify the bounding box of the stop line in the image. [/INST]
"""
raised_table_prompt = """
[INST] <image>\nA raised table in the road is usually painted in white color with two arrows. Please identify the bounding box of the raised table in the image. [/INST]
"""

def miou_cal(gt, pred):
    return np.logical_and(gt, pred).sum()/np.logical_or(gt, pred).sum()

def main():
        
        
    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', required=True, help='stop_line or raised_table') 
    args = parser.parse_args()
    obj = args.object
        
    #--------------
    # arguments
    #--------------
    method='zero-shot'
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    image_size = (336,336)
    image_path = f'images/{obj}/'
    image_names = os.listdir(image_path)
    image_names = [name for name in image_names if ((name.endswith('.png') & ('masking' not in name)))]
    output_path = f'outputs/{method}/{obj}/'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if obj == "stop_line":prompt = stop_line_prompt
    else:prompt = raised_table_prompt
    
    # ---------------
    # load model
    # ---------------
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True
    )
    
    # ---------------
    # iterate image
    # --------------- 
    for image_name in tqdm(image_names):
    
        # ---------------
        # annotate image
        # ---------------
        save_name = image_name.split('.')[0]
        img = Image.open(image_path+image_name).convert('RGB').resize(image_size)
        inputs = processor(prompt, img, return_tensors="pt").to("cuda:0")
        output = model.generate(**inputs, max_new_tokens=50)
        
        # --------------------
        # process bounding box
        # --------------------
        try:
            xtl, ytl, xbr, ybr = processor.decode(
                output[0], 
                skip_special_tokens=True).split('[')[-1].split(']')[0].replace("'",'').split(',')
            xtl, ytl, xbr, ybr = size*float(xtl), size*float(ytl), size*float(xbr), size*float(ybr)
            np.save(output_path+save_name+'_bbox.npy', np.array([xtl, ytl, xbr, ybr]))
        except:
            xtl, ytl, xbr, ybr = 0,0,0,0
        np.save(output_path+save_name+'_bbox.npy', np.array([xtl, ytl, xbr, ybr]))
                        
        # --------------------
        # visualize bounding box
        # --------------------
        try:
            size=image_size[0]
            plt.imshow(img)
            plt.hlines(ytl, xmin=xtl, xmax=xbr, color='red')
            plt.hlines(ybr, xmin=xtl, xmax=xbr, color='red')
            plt.vlines(xtl, ymin=ytl, ymax=ybr, color='red')
            plt.vlines(xbr, ymin=ytl, ymax=ybr, color='red')
            plt.axis('off')
            plt.savefig(output_path+image_name, dpi=600, bbox_inches='tight', pad_inches=0)
            plt.close()
        except:
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(output_path+image_name, dpi=600, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        
        # --------------------
        # binary masking
        # --------------------
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        rect = Rectangle((xtl, ytl), xbr-xtl, ybr-ytl, color="black", angle=0, rotation_point='center')
        ax.add_patch(rect)
        plt.savefig(output_path+save_name+'_masking.png', dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        ## generate binary masking
        img_masking = np.array(Image.open(output_path+save_name+'_masking.png').convert('RGB').resize(image_size))
        masking = np.zeros(image_size)
        masking[np.where(img_masking.sum(axis=-1)==0.0)]=True
        
        ## save
        np.save(output_path+save_name+'_masking.npy', masking)
                
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
