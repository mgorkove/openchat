from openchat import OpenChat
from web_demo_env import WebDemoEnv

# CPU Running
OpenChat(model="blenderbot", size="large", env=WebDemoEnv())
# GPU Running
#OpenChat(model="blenderbot", size="large", env=WebDemoEnv(), device="cuda")
