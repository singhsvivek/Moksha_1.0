import requests
import json
import sys
from Moksha_1.config import settings

class Messenger:
    def __init__(self):
        # Map channels to settings
        self.channels = {
            "alert": settings.DISCORD_WEBHOOK_ALERTS or settings.DISCORD_WEBHOOK_URL,
            "heartbeat": settings.DISCORD_WEBHOOK_HEARTBEAT or settings.DISCORD_WEBHOOK_URL,
            "error": settings.DISCORD_WEBHOOK_ERROR or settings.DISCORD_WEBHOOK_URL
        }
        
        # DEBUG PRINT AT STARTUP
        print(f"DEBUG: Messenger Init. Alert URL set? {'Yes' if self.channels['alert'] else 'No'}")

    def send_message(self, message, title="Moksha Notification", channel="alert"):
        url = self.channels.get(channel)
        
        # 1. CHECK FOR MISSING URL
        if not url:
            print(f"❌ ERROR: No URL found for channel '{channel}'. Check .env file.")
            return

        # 2. CHECK FOR PLACEHOLDER TEXT
        if "YOUR_" in url:
            print(f"❌ ERROR: URL for '{channel}' still contains placeholder text. Edit .env file.")
            return

        # 3. CONSTRUCT PAYLOAD
        color_map = {
            "alert": 65280,      # Green
            "heartbeat": 3447003,# Blue
            "error": 16711680    # Red
        }

        data = {
            "username": "Moksha Prime",
            "embeds": [{
                "title": title,
                "description": message,
                "color": color_map.get(channel, 3447003),
                "footer": {"text": f"Channel: #{channel.upper()}"}
            }]
        }
        
        # 4. SEND WITH VERBOSE ERROR HANDLING
        try:
            response = requests.post(url, json=data, timeout=5)
            if response.status_code == 204:
                print(f"✅ Discord Sent ({channel})")
            else:
                print(f"❌ Discord Failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Messenger Connection Error: {e}")

messenger = Messenger()
