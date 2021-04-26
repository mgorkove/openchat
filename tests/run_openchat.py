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
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(device)

    if device != 'cuda':
        exit(-1)

    OpenChats(models=["blender.small", "blender.medium", 'dialogpt.small',
                      'dialogpt.medium', 'gptneo.small', 'gptneo.large'],
              device=device,
              environment='webserver',
              method="top_k",
              top_k=25,
              top_p=0.85,
              no_repeat_ngram_size=3,
              )
