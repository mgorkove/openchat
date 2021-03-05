from openchat import OpenChat
from web_demo_env import WebDemoEnv

OpenChat(model="blenderbot", size="large", env=WebDemoEnv(), device="cuda")
