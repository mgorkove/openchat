from transformers import (
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


#------dialogpt samll------#
GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small")
GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-small")
#------dialogpt medium------#
GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")


#------gptneo small------#
AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
#------gptneo large------#
AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")


#------blender small------#
BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
#------blender medium------#
BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
