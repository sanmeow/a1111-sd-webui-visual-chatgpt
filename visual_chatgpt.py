#WORK IN PROCESS!
#Use webui api https://github.com/mix1009/sdwebuiapi
#Change to BLIP2 COCO for VQA and caption
#Working parts T2I ,CAPATION, VQA 
#EMBED PREFIX and INSTRUCTIONS and SUFFIX , beware heavy token usage while debugging!
#fix windows version preview
#TODO at the end of code
import sys
import os
os.environ["OPENAI_API_KEY"] = 'API KEY HERE'
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPSegProcessor, CLIPSegForImageSegmentation
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
import re
import uuid
from PIL import Image
from omegaconf import OmegaConf
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import einops
import requests
from lavis.models import load_model_and_preprocess
import webuiapi

# YOUR WEBUI SERVER HERE
api = webuiapi.WebUIApi(host='YOUR LOCAL IP HERE',
                        port=7860,
                        sampler='Euler a',
                        steps=20)




VISUAL_CHATGPT_PREFIX = """Visual ChatGPT is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. Visual ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Visual ChatGPT is able to process and understand large amounts of text and images. As a language model, Visual ChatGPT can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and Visual ChatGPT can invoke different tools to indirectly understand pictures. When talking about images, Visual ChatGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, Visual ChatGPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. Visual ChatGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.
Human may provide new figures to Visual ChatGPT with a description. The description helps Visual ChatGPT to understand this image, but Visual ChatGPT should use tools to finish following tasks, rather than directly imagine from the description.
Overall, Visual ChatGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 
TOOLS:
------
Visual ChatGPT  has access to the following tools:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VISUAL_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.
Begin!
Previous conversation history:
{chat_history}
New input: {input}
Since Visual ChatGPT is a text language model, Visual ChatGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for Visual ChatGPT, Visual ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad}"""

def cut_dialogue_history(history_memory, keep_last_n_words=500):
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"hitory_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    else:
        paragraphs = history_memory.split('\n')
        last_n_tokens = n_tokens
        while last_n_tokens >= keep_last_n_words:
            last_n_tokens = last_n_tokens - len(paragraphs[0].split(' '))
            paragraphs = paragraphs[1:]
        return '\n' + '\n'.join(paragraphs)

def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[0:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
        recent_prev_file_name = name_split[0]
        new_file_name = '{}_{}_{}_{}.png'.format(this_new_uuid, func_name, recent_prev_file_name, most_org_file_name)
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
        recent_prev_file_name = name_split[0]
        new_file_name = '{}_{}_{}_{}.png'.format(this_new_uuid, func_name, recent_prev_file_name, most_org_file_name)
    return os.path.join(head, new_file_name)

class ConversationBot:
    def __init__(self):
        print("Initializing VisualChatGPT")
        self.llm = OpenAI(temperature=0)
        #self.edit = ImageEditing()
        self.i2t = ImageCaptioning(device="cuda:0") #combine VQA
        self.t2i = T2I(device="cuda:1")
        #self.image2canny = image2canny()
        #self.canny2image = canny2image()
        #self.image2line = image2line()
        #self.line2image = line2image()
        #self.image2hed = image2hed()
        #self.hed2image = hed2image()
        #self.image2scribble = image2scribble()
        #self.scribble2image = scribble2image()
        #self.image2pose = image2pose()
        #self.pose2image = pose2image()
        #self.VQA = BLIPVQA(device="cuda:1")
        #self.image2seg = image2seg()
        #self.seg2image = seg2image()
        #self.image2depth = image2depth()
        #self.depth2image = depth2image()
        #self.image2normal = image2normal()
        #self.normal2image = normal2image()
        #self.pix2pix = Pix2Pix()
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.tools = [
            Tool(name="Get Photo Description", func=self.i2t.inference,
                 description="useful when you want to know what is inside the photo. receives image_path as input. "
                             "The input to this tool should be a string, representing the image_path. "),
            Tool(name="Generate Image Condition On Canny Image", func=self.canny2image.inference,
                 description="useful when you want to generate a new real image from both the user desciption and a canny image. like: generate a real image of a object or something from this canny image, or generate a new real image of a object or something from this edge image. "
                             "The input to this tool should be a comma seperated string of two, representing the image_path and the user description. "),
            Tool(name="Generate Image From User Input Text", func=self.t2i.inference,
                 description="useful when you want to generate an image from a user input text and save it to a file. like: generate an image of an object or something, or generate an image that includes some objects. "
                             "The input to this tool should be a string, representing the text used to generate image. "),
            Tool(name="Answer Question About The Image", func=self.i2t.get_answer_from_question_and_image,
                 description="useful when you need an answer for a question based on an image. like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
                             "The input to this tool should be a comma seperated string of two, representing the image_path and the question"),
            ]
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': VISUAL_CHATGPT_PREFIX, 'format_instructions': VISUAL_CHATGPT_FORMAT_INSTRUCTIONS, 'suffix': VISUAL_CHATGPT_SUFFIX}, )

    def run_text(self, text, state):
        print("===============Running run_text =============")
        print("Inputs:", text, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text})
        print("======>Current memory:\n %s" % self.agent.memory)
        #response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'].replace("\\", "/"))
        
        #FIX WINDOWS PREVIEW, see ISSUE108 https://github.com/microsoft/visual-chatgpt/issues/108#issuecomment-1465247501
        
        state = state + [(text, response[:-1]+" ")]
        print("Outputs:", state)
        return state, state

    def run_image(self, image, state, txt):
        print("===============Running run_image =============")
        print("Inputs:", image, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.i2t.inference(image_filename)
        Human_prompt = "\nHuman: provide a figure named {}. The description is: {}. This information helps you to understand this image, but you should use tools to finish following tasks, " \
                       "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(image_filename, description)
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        print("======>Current memory:\n %s" % self.agent.memory)
        state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt[:-1])]
        print("Outputs:", state)
        return state, state, txt + ' ' + image_filename + ' '


# In[10]:


class T2I:
    def __init__(self, device):
        print("Initializing T2I to %s" % device)
        self.device = device
        #self.text_refine_tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        #self.text_refine_model = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        #self.text_refine_gpt2_pipe = pipeline("text-generation", model=self.text_refine_model, tokenizer=self.text_refine_tokenizer, device=self.device)

    def inference(self, text):
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        #refined_text = self.text_refine_gpt2_pipe(text)[0]["generated_text"]
        #print(f'{text} refined to {refined_text}')
        image = api.txt2img(prompt=text,
                    negative_prompt="ugly, out of frame",
                    seed=-1,
#                    styles=["anime"],
                    cfg_scale=7,
#                      sampler_index='DDIM',
#                      steps=30,
                    ).images[0]
        image.save(image_filename)
        print(f"Processed T2I.run, text: {text}, image_filename: {image_filename}")
        return image_filename

#class ImageCaptioning:
#    def __init__(self, device):
#        print("Initializing ImageCaptioning to %s" % device)
#        self.device = device
#        model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)

#    def inference(self, image_path):
#        inputs = self.vis_processorsimage = vis_processors["eval"](Image.open(image_path)).unsqueeze(0).to(device)
#        out = self.model.generate(**inputs,min_length=30,max_length=150)
#        captions = self. vis_processors.decode(out[0], skip_special_tokens=True)
#        return captions

class ImageCaptioning:
    def __init__(self, device):
        print("Initializing Blip2 to %s" % device)
        self.device = device
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip2", model_type="coco", is_eval=True, device=device)

    def inference(self, image_path):
        image = self.vis_processors["eval"](Image.open(image_path)).unsqueeze(0).to("cuda:1")
        captions = self.model.generate({"image": image})
        return captions
        
    def get_answer_from_question_and_image(self, inputs):
        image_path, question = inputs.split(",")
        raw_image = Image.open(image_path).convert('RGB')
        print(F'BLIPVQA :question :{question}')
        image=self.vis_processors["eval"](Image.open(image_path)).unsqueeze(0).to("cuda:1")
        answer=self.model.generate({"image": image,"prompt": "Question: {question}"})
        return answer

class canny2image:
    def __init__(self):
        print("Initialize the canny2image model.")
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'EasyNegative, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting canny2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        img = Image.open(image_path)
        pmpt = instruct_text
        unit1 = webuiapi.ControlNetUnit(input_image=img, module='canny', model='control_sd15_canny [9d312881]')
#        r = api.txt2img(prompt=(pmpt + ', ' + self.a_prompt), controlnet_units=[unit1])
        r = api.txt2img(prompt=pmpt, controlnet_units=[unit1])
        updated_image_path = get_new_image_name(image_path, func_name="canny2image")
        real_image = r.image  # get default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path
class image2scribble:
    def __init__(self):
        print("Direct detect scribble.")
        self.detector = HEDdetector()
        self.resolution = 512

    def inference(self, inputs):
        print("===>Starting image2scribble Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        detected_map = self.detector(resize_image(image, self.resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0
        detected_map = 255 - detected_map
        updated_image_path = get_new_image_name(inputs, func_name="scribble")
        image = Image.fromarray(detected_map)
        image.save(updated_image_path)
        return updated_image_path

class scribble2image:
    def __init__(self, device):
        print("Initialize the scribble2image model...")
        model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
        model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_scribble.pth', location='cpu'))
        self.model = model.to(device)
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting scribble2image Inference")
        print(f'sketch device {self.device}')
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        prompt = instruct_text
        image = 255 - image
        img = resize_image(HWC3(image), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
        shape = (4, H // 8, W // 8)
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)
        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False, eta=0., unconditional_guidance_scale=self.scale, unconditional_conditioning=un_cond)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        updated_image_path = get_new_image_name(image_path, func_name="scribble2image")
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path

class image2pose:
    def __init__(self):
        print("Direct human pose.")
        self.detector = OpenposeDetector()
        self.resolution = 512

    def inference(self, inputs):
        print("===>Starting image2pose Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        detected_map, _ = self.detector(resize_image(image, self.resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        updated_image_path = get_new_image_name(inputs, func_name="human-pose")
        image = Image.fromarray(detected_map)
        image.save(updated_image_path)
        return updated_image_path

class pose2image:
    def __init__(self, device):
        print("Initialize the pose2image model...")
        model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
        model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_openpose.pth', location='cpu'))
        self.model = model.to(device)
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting pose2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        prompt = instruct_text
        img = resize_image(HWC3(image), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [ self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
        shape = (4, H // 8, W // 8)
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)
        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False, eta=0., unconditional_guidance_scale=self.scale, unconditional_conditioning=un_cond)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        updated_image_path = get_new_image_name(image_path, func_name="pose2image")
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path

class image2seg:
    def __init__(self):
        print("Direct segmentations.")
        self.detector = UniformerDetector()
        self.resolution = 512

    def inference(self, inputs):
        print("===>Starting image2seg Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        detected_map = self.detector(resize_image(image, self.resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        updated_image_path = get_new_image_name(inputs, func_name="segmentation")
        image = Image.fromarray(detected_map)
        image.save(updated_image_path)
        return updated_image_path

class seg2image:
    def __init__(self, device):
        print("Initialize the seg2image model...")
        model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
        model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_seg.pth', location='cpu'))
        self.model = model.to(device)
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting seg2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        prompt = instruct_text
        img = resize_image(HWC3(image), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
        shape = (4, H // 8, W // 8)
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)
        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False, eta=0., unconditional_guidance_scale=self.scale, unconditional_conditioning=un_cond)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        updated_image_path = get_new_image_name(image_path, func_name="segment2image")
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path

class image2depth:
    def __init__(self):
        print("Direct depth estimation.")
        self.detector = MidasDetector()
        self.resolution = 512

    def inference(self, inputs):
        print("===>Starting image2depth Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        detected_map, _ = self.detector(resize_image(image, self.resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        updated_image_path = get_new_image_name(inputs, func_name="depth")
        image = Image.fromarray(detected_map)
        image.save(updated_image_path)
        return updated_image_path

class depth2image:
    def __init__(self, device):
        print("Initialize depth2image model...")
        model = create_model('ControlNet/models/cldm_v15.yaml', device=device).to(device)
        model.load_state_dict(load_state_dict('ControlNet/models/control_sd15_depth.pth', location='cpu'))
        self.model = model.to(device)
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def inference(self, inputs):
        print("===>Starting depth2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        image = np.array(image)
        prompt = instruct_text
        img = resize_image(HWC3(image), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [ self.model.get_learned_conditioning([prompt + ', ' + self.a_prompt] * self.num_samples)]}
        un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)]}
        shape = (4, H // 8, W // 8)
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples, shape, cond, verbose=False, eta=0., unconditional_guidance_scale=self.scale, unconditional_conditioning=un_cond)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        updated_image_path = get_new_image_name(image_path, func_name="depth2image")
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path


# In[11]:


if __name__ == '__main__':
    bot = ConversationBot()
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="Visual ChatGPT")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear️")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("Upload", file_types=["image"])

        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        print(state)
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        demo.launch(server_name="0.0.0.0", server_port=7865) #YOUR Visual-ChatGPT SERVER HERE

      
#TODO list 
#get all other functions back through API
#support WD Tagger for describe PROMPT
#Make it a Extension inside AUTOMATIC1111 SD webui
#SELF NEED FUNCTION example:generate line-drawing version of art which loads LORA,generate pano,generate 3D etc
#DISCORD chatbot version
#CODE clean-up
