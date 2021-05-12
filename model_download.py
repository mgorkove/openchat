from transformers import (
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


#------dialogpt samll------#
model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small")
tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-small")
#------dialogpt medium------#
model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")

print("dialogpt is done!")


#------gptneo small------#
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
#------gptneo large------#
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

print("gptneo is done!")

#------blender small------#
model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
#------blender medium------#
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

print("blender is done!")
