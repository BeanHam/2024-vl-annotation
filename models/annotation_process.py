import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm
from matplotlib.patches import Rectangle


def main():
    
    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', required=True, help='stop_line or raised_table') 
    args = parser.parse_args()
    obj = args.object
    img_size = (336,336)
    
    #-------------------------
    # load annotations
    #-------------------------
    with open(f'images/{obj}.xml', 'r') as f:
        data = f.read()
    annos = BeautifulSoup(data, "xml").find_all('image')
    
    #-------------------------
    # iterate annotations
    #-------------------------    
    for anno in tqdm(annos):
    
        ## load image
        img_name = anno['name'].split('.')[0]
        img = Image.open(f'images/stop_line/{img_name}.png').convert('RGB').resize(img_size)
        
        ## prepare bounding boxes -- one image could have multiple objects
        bboxes= anno.find_all('box')
        img_bboxes = []
        for bbox in bboxes:
            try:
                img_bboxes.append([float(bbox['xtl']), 
                                   float(bbox['ytl']), 
                                   float(bbox['xbr']), 
                                   float(bbox['ybr']), 
                                   float(bbox['rotation'])])
            except:
                img_bboxes.append([float(bbox['xtl']), 
                                   float(bbox['ytl']), 
                                   float(bbox['xbr']), 
                                   float(bbox['ybr']), 
                                   float(0.0)])
        img_bboxes = np.stack(img_bboxes)
        
        ## overlay bounding box on image
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        for bbox in img_bboxes:
            xtl, ytl, xbr, ybr, rotation = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
            rect = Rectangle((xtl, ytl), xbr-xtl, ybr-ytl, color="black", angle=rotation, rotation_point='center')
            ax.add_patch(rect)
        plt.savefig(f'images/{obj}/{img_name}_masking.png', dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        ## generate binary masking
        img_masking = np.array(Image.open(f'images/{obj}/{img_name}_masking.png').convert('RGB').resize(img_size))
        masking = np.zeros(img_size)
        masking[np.where(img_masking.sum(axis=-1)==0.0)]=True
        
        ## save
        np.save(f'images/{obj}/{img_name}_masking.npy', masking)
        np.save(f'images/{obj}/{img_name}_bbox.npy', img_bboxes)
                               
if __name__ == "__main__":
    main()