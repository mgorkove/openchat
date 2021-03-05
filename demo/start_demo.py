from openchat import OpenChat
from web_demo_env import WebDemoEnv

# Running
OpenChat(model="blenderbot", size="large", env=WebDemoEnv(), device="cuda")
