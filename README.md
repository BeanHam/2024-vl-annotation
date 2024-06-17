# Towards Zero-Shot Annotation of the Built Environment with Vision-Language Models (Vision Paper)

### Introduction
Equitable urban transportation applications require high-fidelity digital representations of the built environment: not just streets and sidewalks, but bike lanes, marked and unmarked crossings, curb ramps and cuts, obstructions, traffic signals, signage, street markings, potholes, and more. Direct inspections and manual annotations are prohibitively expensive at scale. Conventional machine learning methods require substantial annotated training data for adequate performance. In this paper, we consider vision language models as a mechanism for annotating diverse urban features from satellite images, reducing the dependence on human annotation to produce large training sets.  

### Contribution
We demonstrate proof-of-concept combining a state-of-the-art vision language model and variants of a prompting strategy that asks the model to consider segmented elements independently of the original image. 

![alt text](https://github.com/BeanHam/2024-vl-annotation/blob/main/visualizations/pipeline.png)
###### Pipeline of our proposed automated annotation process. Users input a pair of (satellite image, annotation guidance). The image will go through a set of processes including segmentation, filtering, and set-of-mark generation. Then the image and guidance will go through a vision-language model, the output of which is post-processed to produce the final annotation results. The procedure requires no fine-tuning, and can be applied on different features with minimal adjust on the guidance.

### Examples
Experiments on two urban features --- stop lines and raised tables --- show that while direct zero-shot prompting correctly annotates nearly zero images, the pre-segmentation strategies can annotate images with near 40% intersection-over-union accuracy. 

|----------------|--------| ------ | ------ | ------ |
| Features | Direct Prompting | SoM - No Context | SoM - In Context | SoM - Combination |
|-- Stop Lines --| 0.0000 | 0.2483 | 0.3354 | 0.3657 |
|-- Raised Tables --| 0.0190 | 0.3315 | 0.4069 | 0.4189 |

Green and yellow outlines indicate perfect and approximate annotations, respectively. A Red outline indicate inaccurate annotations.

#### Stop Lines
![alt text](https://github.com/BeanHam/2024-vl-annotation/blob/main/visualizations/stop_lines.png)

2. Raised Table.
![alt text](https://github.com/BeanHam/2024-vl-annotation/blob/main/visualizations/raised_tables.png)
