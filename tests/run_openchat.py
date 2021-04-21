from openchat import OpenChat

if __name__ == '__main__':
    '''
    Model list
    'blender.sall', 'blender.medium', 'blender.large', 'blender.xlarge', 'blender.xxlarge', 
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
    OpenChat(model="gptneo.medium", device="cpu", environment='webserver')
    #OpenChat(model="dialogpt.small", device="cpu")