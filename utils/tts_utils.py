"""
Text-to-Speech utilities
"""

import pyttsx3
import os
import tempfile
from typing import Optional

class TTS_System:
    def __init__(self, use_online: bool = False):
        """
        Initialize TTS system
        
        Args:
            use_online: If True, use gTTS (online), else use pyttsx3 (offline)
        """
        self.use_online = use_online
        self.engine = None
        
        if not use_online:
            try:
                self.engine = pyttsx3.init()
                # Configure voice properties
                voices = self.engine.getProperty('voices')
                if voices:
                    self.engine.setProperty('voice', voices[0].id)
                self.engine.setProperty('rate', 150)  # Speed of speech
                self.engine.setProperty('volume', 0.9)  # Volume level
            except Exception as e:
                print(f"Warning: Could not initialize pyttsx3: {e}")
                self.engine = None
    
    def speak(self, text: str) -> bool:
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            
        Returns:
            True if successful, False otherwise
        """
        if not text:
            return False
        
        try:
            if self.use_online:
                return self._speak_online(text)
            else:
                return self._speak_offline(text)
        except Exception as e:
            print(f"TTS Error: {e}")
            return False
    
    def _speak_offline(self, text: str) -> bool:
        """Speak using pyttsx3 (offline)"""
        if self.engine is None:
            return False
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"Offline TTS Error: {e}")
            return False
    
    def _speak_online(self, text: str) -> bool:
        """Speak using gTTS (online)"""
        try:
            from gtts import gTTS
            import pygame
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                # Generate speech
                tts = gTTS(text=text, lang='en')
                tts.save(tmp_file.name)
                
                # Play audio
                pygame.mixer.init()
                pygame.mixer.music.load(tmp_file.name)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                # Clean up
                pygame.mixer.quit()
                os.unlink(tmp_file.name)
                
            return True
        except ImportError:
            print("gTTS or pygame not available for online TTS")
            return False
        except Exception as e:
            print(f"Online TTS Error: {e}")
            return False
    
    def set_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        if self.engine and not self.use_online:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)"""
        if self.engine and not self.use_online:
            self.engine.setProperty('volume', volume)
    
    def close(self):
        """Close TTS engine"""
        if self.engine and not self.use_online:
            self.engine.stop()
