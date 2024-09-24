import requests
import random

def download_random_audio_sample():
    audio_urls = [
        "http://www.moviesoundclips.net/movies1/darkknightrises/darkness.mp3"
    ]
    
    audio_url = random.choice(audio_urls)
    audio_path = "test_audio.wav"
    
    response = requests.get(audio_url)
    if response.status_code == 200:
        with open(audio_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded audio sample from {audio_url} to {audio_path}")
    else:
        raise Exception(f"Failed to download audio: {response.status_code}")
    
    return audio_path