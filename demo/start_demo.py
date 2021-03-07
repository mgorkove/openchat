from openchat import OpenChat
from web_demo_env import WebDemoEnv

# CPU Running
#print('cpu')
#OpenChat(model="blenderbot", size="large", env=WebDemoEnv())
# GPU Running
print('gpu')
OpenChat(model="blenderbot", size="large", env=WebDemoEnv(), device="cuda")
