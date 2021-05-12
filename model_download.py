from transformers import (
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

download_type = sys.argv[1]

if download_type == 'dialogpt':
    #------dialogpt samll------#
    model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small")
    tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-small")
    #------dialogpt medium------#
    model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")

    print("dialogpt is done!")

elif download_type == 'gptneo':
    #------gptneo small------#
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    #------gptneo large------#
    #model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    #tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    print("gptneo is done!")    

elif  download_type == 'blender':
    #------blender small------#
    model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
    tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
    #------blender medium------#
    model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

    print("blender is done!")
