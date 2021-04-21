import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch

from openchat import OpenChat, OpenChats

if __name__ == '__main__':
    '''
    Model list
    'blender.small', 'blender.medium', 'blender.large', 'blender.xlarge', 'blender.xxlarge', 
    'dialogpt.small', 'dialogpt.medium', 'dialogpt.large', 
    'gptneo.small', 'gptneo.medium', 'gptneo.large', 'gptneo.xlarge', 
    'dodecathlon.all_tasks_mt', 'dodecathlon.convai2', 'dodecathlon.wizard_of_wikipedia', 
    'dodecathlon.empathetic_dialoguesdodecathlon.eli5', 'dodecathlon.reddit', 'dodecathlon.twitter', 
    'dodecathlon.ubuntu', 'dodecathlon.image_chat', 'dodecathlon.cornell_movie', 
    'dodecathlon.light_dialog', 'dodecathlon.daily_dialog', 
    'reddit.xlarge', 'reddit.xxlarge', 
    'safety.sensitive', 'safety.offensive', 
    'unlikelihood.wizard_of_wikipedia.context_and_label', 'unlikelihood.wizard_of_wikipedia.context', 
    'unlikelihood.wizard_of_wikipedia.label', 'unlikelihood.convai2.context_and_label', 'unlikelihood.convai2.context', 
    'unlikelihood.convai2.label', 'unlikelihood.convai2.vocab.alpha.1e-0', 'unlikelihood.convai2.vocab.alpha.1e-1', 
    'unlikelihood.convai2.vocab.alpha.1e-2', 'unlikelihood.convai2.vocab.alpha.1e-3', 
    'unlikelihood.eli5.context_and_label', 'unlikelihood.eli5.context', 'unlikelihood.eli5.label', 
    'wizard_of_wikipedia.end2end_generator'
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #OpenChat(model="blender.small", device="cpu", environment='webserver')
    OpenChats(models=["blender.small", 'dialogpt.small', 'gptneo.small'], device=device, environment='webserver')
    #OpenChats(models=["blender.small", 'dialogpt.small', 'gptneo.small'], device="cpu", environment='webserver')
    #OpenChat(model="dialogpt.small", device="cpu")
