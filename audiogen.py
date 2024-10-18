"""
Audio Generation Module

This module provides functionality for text-to-speech (TTS) conversion and audio generation.
It includes abstract base classes and concrete implementations for TTS services and audio
generation, with a focus on OpenAI's TTS capabilities.

The module contains:
1. Abstract TTS class defining the interface for text-to-speech conversion.
2. OpenAITTS class implementing TTS using OpenAI's API.
3. AbstractAudioGen class defining the interface for audio generation from log files.
4. OpenAIAudioGen class implementing audio generation with advanced features like
   transcription and voice assignment.
5. Utility function to instantiate the appropriate audio generation class.

This module is designed to be extensible, allowing for easy integration of additional
TTS services and audio generation methods in the future.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os
import json
from openai import OpenAI
from pydub import AudioSegment

class TTS(ABC):
    @abstractmethod
    def generate_audio(self, text: str, voice_dict: Dict[str, Any]) -> str:
        """
        Generate audio from text using specified voice settings.

        Args:
            text (str): The text to convert to speech.
            voice_dict (Dict[str, Any]): Dictionary containing voice settings.

        Returns:
            str: Path to the generated audio file.
        """
        pass

class OpenAITTS(TTS):
    """
    Implementation of TTS using OpenAI's text-to-speech API.
    """
    def __init__(self) -> None:
        self.client = OpenAI()

    def generate_audio(self, text: str, voice_dict: Dict[str, Any]) -> str:
        """
        Generate audio using OpenAI's TTS service.

        Args:
            text (str): The text to convert to speech.
            voice_dict (Dict[str, Any]): Dictionary containing voice settings.

        Returns:
            str: Path to the generated audio file.
        """
        response = self.client.audio.speech.create(
            model=voice_dict.get('model', 'tts-1-hd'),
            voice=voice_dict.get('voice', 'alloy'),
            input=text
        )

        output_path = f"output_{voice_dict.get('voice', 'default')}.mp3"
        response.stream_to_file(output_path)
        return output_path

class AbstractAudioGen(ABC):
    """
    Abstract base class for audio generation from log files.
    """
    def __init__(self, tts_service: TTS) -> None:
        self.tts_service = tts_service

    def process_log_files(self, log_dir: str, voice_settings: Dict[str, Any]) -> None:
        """
        Process text files in a directory and generate audio for each.

        Args:
            log_dir (str): Directory containing log files.
            voice_settings (Dict[str, Any]): Voice settings for TTS.
        """
        for filename in os.listdir(log_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(log_dir, filename), 'r') as file:
                    text = file.read()
                    audio_path = self.tts_service.generate_audio(text, voice_settings)
                    print(f"Generated audio for {filename}: {audio_path}")

class OpenAIAudioGen:
    """
    Advanced audio generation class using OpenAI's TTS service.
    """
    def __init__(self, tts_service: TTS) -> None:
        self.tts_service = tts_service
        self.agents: List[Any] = []

    def add_agent(self, agent: Any) -> None:
        """
        Add an agent to the audio generation system.

        Args:
            agent (Any): Agent object to be added.
        """
        self.agents.append(agent)

    def process_log_files(self, log_dir: str, voice_settings: Dict[str, Any]) -> None:
        """
        Process text files in a directory and generate audio for each.

        Args:
            log_dir (str): Directory containing log files.
            voice_settings (Dict[str, Any]): Voice settings for TTS.
        """
        for filename in os.listdir(log_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(log_dir, filename), 'r') as file:
                    text = file.read()
                    audio_path = self.tts_service.generate_audio(text, voice_settings)
                    print(f"Generated audio for {filename}: {audio_path}")

    def transcribe_and_save(self, json_file_path: str, output_audio_path: str) -> None:
        """
        Transcribe a JSON file containing dialogue and save as an audio file.

        Args:
            json_file_path (str): Path to the JSON file containing the transcript.
            output_audio_path (str): Path where the output audio file will be saved.
        """
        if os.path.exists(output_audio_path):
            print(f"Audio file {output_audio_path} already exists. Skipping generation.")
            return

        with open(json_file_path, 'r') as file:
            transcript = json.load(file)

        audio_segments = []

        for entry in transcript:
            for speaker, text in entry.items():
                voice_settings = self.get_voice_settings(speaker)
                audio_path = self.tts_service.generate_audio(text, voice_settings)
                audio_segments.append(AudioSegment.from_mp3(audio_path))
                os.remove(audio_path)  # Remove temporary audio file

        combined_audio = sum(audio_segments)
        combined_audio.export(output_audio_path, format="mp3")
        print(f"Generated and saved audio to {output_audio_path}")

    def get_voice_settings(self, speaker: str) -> Dict[str, str]:
        """
        Get voice settings for a specific speaker.

        Args:
            speaker (str): Name of the speaker.

        Returns:
            Dict[str, str]: Voice settings for the speaker.
        """
        for agent in self.agents:
            if speaker == agent.config.name:
                return {"voice": agent.config.voice, "model": "tts-1-hd"}
            elif speaker == f"{agent.config.name}'s consciousness":
                return {"voice": agent.config.consciousness_voice, "model": "tts-1-hd"}
        
        # Default voice if no match found
        return {"voice": "alloy", "model": "tts-1-hd"}

    def assign_voices(self) -> None:
        """
        Assign voices to agents.
        """
        self.agents[0].config.voice = "echo"
        self.agents[0].config.consciousness_voice = "nova"

        self.agents[1].config.voice = "onyx"
        self.agents[1].config.consciousness_voice = "shimmer"

def get_audiogen(audio_class_name: str) -> AbstractAudioGen:
    """
    Factory function to get an instance of the specified audio generation class.

    Args:
        audio_class_name (str): The name of the audio generation class.

    Returns:
        AbstractAudioGen: An instance of the specified audio generation class.

    Raises:
        NotImplementedError: If an unsupported audio class name is provided.
    """
    if audio_class_name == "OpenAIAudioGen":
        tts_service = OpenAITTS()
        return OpenAIAudioGen(tts_service)
    else:
        raise NotImplementedError(f"Audio generation class '{audio_class_name}' is not implemented.")









#