# WEBUI API version of Visual ChatGPT

Because putting all those models in VRAM was just too heavy,I decide to make a API version for using AUTOMATIC1111 stable-diffusion-webui.  
Sharing this version is for helping who want to try it out.  

# WORK IN PROCESS!

Windows Installation:
```
git clone https://github.com/sanmeow/a1111-sd-webui-visual-chatgpt.git
cd a1111-sd-webui-visual-chatgpt

python -m venv venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade -r requirements.txt

mkdir image

python visual_chatgpt.py
```

# What was changed  
Use webui API https://github.com/mix1009/sdwebuiapi  
Working parts T2I ,CAPTION, VQA  
Change to BLIP2 COCO for VQA and caption    
Fix windows version preview

# Todo list 
Get all other functions back through API  
Emebed PREFIX and INSTRUCTIONS and SUFFIX , beware heavy token usage while debugging! 
Need support WD Tagger for describe PROMPT,for now use "I want a cat" instead of "draw me a cat".  
Make it an Extension inside AUTOMATIC1111 SD webui  
SELF NEED FUNCTION example:generate line-drawing version of art which loads LORA,generate pano,generate 3D ...etc  
DISCORD chatbot version  
CODE clean-up  
Other LLM version

# Post-script
First time using GITHUB fork, if any part doing wrong I'll try to fix it.  
　  
　  
　  
　  
# Visual ChatGPT 

**Visual ChatGPT** connects ChatGPT and a series of Visual Foundation Models to enable **sending** and **receiving** images during chatting.

See our paper: [<font size=5>Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models</font>](https://arxiv.org/abs/2303.04671)

## Demo 
<img src="./assets/demo_short.gif" width="750">

##  System Architecture 

 
<p align="center"><img src="./assets/figure.jpg" alt="Logo"></p>


## Quick Start

```
# create a new environment
conda create -n visgpt python=3.8

# activate the new environment
conda activate visgpt

#  prepare the basic environments
pip install -r requirement.txt

# download the visual foundation models
bash download.sh

# prepare your private openAI private key
export OPENAI_API_KEY={Your_Private_Openai_Key}

# create a folder to save images
mkdir ./image

# Start Visual ChatGPT !
python visual_chatgpt.py
```

## GPU memory usage
Here we list the GPU memory usage of each visual foundation model, one can modify ``self.tools`` with fewer visual foundation models to save your GPU memory:

| Foundation Model        | Memory Usage (MB) |
|------------------------|-------------------|
| ImageEditing           | 6667              |
| ImageCaption           | 1755              |
| T2I                    | 6677              |
| canny2image            | 5540              |
| line2image             | 6679              |
| hed2image              | 6679              |
| scribble2image         | 6679              |
| pose2image             | 6681              |
| BLIPVQA                | 2709              |
| seg2image              | 5540              |
| depth2image            | 6677              |
| normal2image           | 3974              |
| InstructPix2Pix        | 2795              |



## Acknowledgement
We appreciate the open source of the following projects:

[Hugging Face](https://github.com/huggingface) &#8194;
[LangChain](https://github.com/hwchase17/langchain) &#8194;
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) &#8194; 
[ControlNet](https://github.com/lllyasviel/ControlNet) &#8194; 
[InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) &#8194; 
[CLIPSeg](https://github.com/timojl/clipseg) &#8194;
[BLIP](https://github.com/salesforce/BLIP) &#8194;


