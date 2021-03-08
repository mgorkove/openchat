from openchat import OpenChat
from web_demo_env import WebDemoEnv

# CPU Running
#print('cpu')
#OpenChat(model="blenderbot", size="large", env=WebDemoEnv(), max_context_length=65536)
# GPU Running
print('gpu')
OpenChat(model="blenderbot", size="large", env=WebDemoEnv(), device="cuda", max_context_length=65536)
