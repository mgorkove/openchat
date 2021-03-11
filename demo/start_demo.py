import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from openchat import OpenChat
from web_demo_env import WebDemoEnv

'''
    gpt2: 12
    
    gpt2-medium: 24
    
    gpt2-large: 36
    
    gpt2-xl: 48
'''

# CPU Running
#print('cpu')
#OpenChat(model="blenderbot", size="large", env=WebDemoEnv(), max_context_length=36)
# GPU Running
print('gpu')
OpenChat(model="blenderbot", size="large", env=WebDemoEnv(), device="cuda", max_context_length=36)
