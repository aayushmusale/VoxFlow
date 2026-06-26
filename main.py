"""
VoxFlow: Zero-Latency Cross-Lingual Communication Pipeline
Architecture: Multi-threaded Producer-Consumer with Virtual Audio Routing.
Features: Hardware VAD, Silero Software VAD, Dynamic Language Locking, and Hallucination Blacklisting.
"""

import os
import queue
import asyncio
import threading
import numpy as np
from scipy.io.wavfile import write

import customtkinter as ctk
import sounddevice as sd
import pygame
import edge_tts
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

# --- CONFIGURATION & ROUTING ---
INDIC_LANGUAGES = {
    "English": {"whisper": "en", "google": "en", "tts": "en-IN-PrabhatNeural"},
    "Hindi": {"whisper": "hi", "google": "hi", "tts": "hi-IN-MadhurNeural"},
    "Marathi": {"whisper": "mr", "google": "mr", "tts": "mr-IN-AarohiNeural"},
    "Bengali": {"whisper": "bn", "google": "bn", "tts": "bn-IN-BashkarNeural"},
    "Telugu": {"whisper": "te", "google": "te", "tts": "te-IN-MohanNeural"},
    "Tamil": {"whisper": "ta", "google": "ta", "tts": "ta-IN-ValluvarNeural"},
    "Gujarati": {"whisper": "gu", "google": "gu", "tts": "gu-IN-NiranjanNeural"},
    "Kannada": {"whisper": "kn", "google": "kn", "tts": "kn-IN-GaganNeural"},
    "Malayalam": {"whisper": "ml", "google": "ml", "tts": "ml-IN-MidhunNeural"},
    "Punjabi": {"whisper": "pa", "google": "pa", "tts": "pa-IN-NavneetNeural"}
}

# Set the theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class VoxFlowApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # --- GUI SETUP ---
        self.title("VoxFlow")
        self.geometry("850x500")

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.english_textbox = ctk.CTkTextbox(self, font=("Arial", 16), wrap="word")
        self.english_textbox.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.english_textbox.insert("0.0", "Listening...\n\nYour speech will appear here.")
        self.english_textbox.configure(state="disabled")

        self.translated_textbox = ctk.CTkTextbox(self, font=("Arial", 16), wrap="word")
        self.translated_textbox.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.translated_textbox.insert("0.0", "Translation...\n\nThe translated text will appear here.")
        self.translated_textbox.configure(state="disabled")

        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.grid(row=1, column=0, columnspan=2, pady=(0, 20))

        # --- DUAL LANGUAGE ROUTING ---
        language_list = list(INDIC_LANGUAGES.keys())

        # Source Language Dropdown
        self.input_lang_var = ctk.StringVar(value="English")
        self.input_dropdown = ctk.CTkOptionMenu(
            self.control_frame, 
            values=language_list,
            variable=self.input_lang_var,
            font=("Arial", 14)
        )
        self.input_dropdown.pack(side="left", padx=(20, 10))

        # Visual Separator
        self.arrow_label = ctk.CTkLabel(self.control_frame, text="➔", font=("Arial", 20, "bold"))
        self.arrow_label.pack(side="left", padx=5)

        # Target Language Dropdown
        self.output_lang_var = ctk.StringVar(value="Hindi")
        self.output_dropdown = ctk.CTkOptionMenu(
            self.control_frame, 
            values=language_list,
            variable=self.output_lang_var,
            font=("Arial", 14)
        )
        self.output_dropdown.pack(side="left", padx=(10, 20))

        self.is_recording = False
        self.record_btn = ctk.CTkButton(
            self.control_frame, 
            text="Start VoxFlow", 
            command=self.toggle_recording,
            font=("Arial", 14, "bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.record_btn.pack(side="left", padx=20)

        # --- AI MODEL INITIALIZATION ---
        print("Loading Whisper model into memory... (Please wait a few seconds)")
        self.whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
        
        # The Bridge between the Microphone and the AI
        self.audio_queue = queue.Queue() 

        # Start the invisible audio player
        pygame.mixer.init()
        print("VoxFlow Ready!")

    # --- UI THREAD-SAFE HELPERS ---
    def update_text_safe(self, textbox, new_text):
        """Safely updates a disabled text box from the background thread."""
        textbox.configure(state="normal")
        textbox.delete("0.0", "end")
        textbox.insert("0.0", new_text)
        textbox.configure(state="disabled")

    def toggle_recording(self):
        if not self.is_recording:
            # TURN ON
            self.is_recording = True
            self.record_btn.configure(text="Stop VoxFlow", fg_color="red", hover_color="darkred")
            
            # Spawn BOTH background worker threads
            self.mic_thread = threading.Thread(target=self.microphone_producer, daemon=True)
            self.ai_thread = threading.Thread(target=self.start_ai_consumer, daemon=True)
            
            self.mic_thread.start()
            self.ai_thread.start()
        else:
            # TURN OFF
            self.is_recording = False
            self.record_btn.configure(text="Start VoxFlow", fg_color="green", hover_color="darkgreen")
            self.update_text_safe(self.english_textbox, "Stopped listening.\nPress Start to resume.")
            self.update_text_safe(self.translated_textbox, "System paused.")

    # ==========================================
    # THREAD 1: THE PRODUCER (Always Listening)
    # ==========================================
    def microphone_producer(self):
        samplerate = 16000
        silence_threshold = 800
        max_silence_chunks = int(1.2 * (samplerate / 1024))
        min_speech_chunks = int(0.5 * (samplerate / 1024))

        q = queue.Queue()
        def callback(indata, frames, time, status):
            q.put(indata.copy())

        self.after(0, lambda: self.update_text_safe(self.english_textbox, "🟢 Mic open. Listening continuously..."))

        with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', blocksize=1024, callback=callback):
            while self.is_recording:
                silence_counter = 0
                audio_data = []
                has_spoken = False
                
                # Listen until silence is detected
                while self.is_recording:
                    try:
                        chunk = q.get(timeout=0.5) 
                    except queue.Empty:
                        continue

                    audio_data.append(chunk)
                    volume = np.linalg.norm(chunk)

                    if volume > silence_threshold:
                        has_spoken = True
                        silence_counter = 0
                    elif has_spoken:
                        silence_counter += 1

                    if has_spoken and silence_counter > max_silence_chunks:
                        break

                if not self.is_recording:
                    break
                    
                # Package the audio slice and drop it onto the Bridge
                if len(audio_data) > min_speech_chunks and has_spoken:
                    slice_filename = f"temp_mic_{np.random.randint(1000, 9999)}.wav"
                    final_recording = np.concatenate(audio_data, axis=0)
                    write(slice_filename, samplerate, final_recording)
                    
                    self.audio_queue.put(slice_filename)
                    print(f"-> [PRODUCER] Sent chunk to AI: {slice_filename}")

    # ==========================================
    # THREAD 2: THE CONSUMER (Always Processing)
    # ==========================================
    def start_ai_consumer(self):
        asyncio.run(self.ai_consumer_loop())

    async def ai_consumer_loop(self):
        loop = asyncio.get_event_loop()
        
        while self.is_recording:
            try:
                mic_file = await loop.run_in_executor(None, self.audio_queue.get)
            except Exception:
                await asyncio.sleep(0.1)
                continue

            try:
                # Pull current language settings
                input_lang_name = self.input_lang_var.get()
                output_lang_name = self.output_lang_var.get()
                
                input_codes = INDIC_LANGUAGES[input_lang_name]
                output_codes = INDIC_LANGUAGES[output_lang_name]

                # --- 1. Transcription (Dynamic Language Lock & VAD) ---
                segments, _ = await loop.run_in_executor(
                    None, lambda: list(self.whisper_model.transcribe(
                        mic_file, 
                        beam_size=5,
                        language=input_codes["whisper"],
                        vad_filter=True, 
                        vad_parameters=dict(min_silence_duration_ms=500),
                        condition_on_previous_text=False
                    ))
                )
                
                valid_text = [seg.text for seg in segments if seg.no_speech_prob < 0.6]
                source_text = " ".join(valid_text).strip()

                # --- 1.5 The Hallucination Blacklist ---
                clean_text = source_text.replace(".", "").replace(",", "").replace("?", "").strip()
                ghost_phrases = [
                    "Thank you", "Bye", "Bye bye", "I like it", "Yeah", "Okay", "Oh", "You",
                    "Hey Priya do you know something", "there's nothing inside", "AKANAU", "what happened"
                ]
                
                if len(clean_text) < 4 or clean_text in ghost_phrases:
                    try: os.remove(mic_file)
                    except: pass
                    continue
                
                self.after(0, lambda t=source_text: self.update_text_safe(self.english_textbox, t))

                # --- 2. Translation (Google API via Executor) ---
                translated_text = await loop.run_in_executor(
                    None, 
                    lambda: GoogleTranslator(
                        source=input_codes["google"], 
                        target=output_codes["google"]
                    ).translate(source_text)
                )
                
                self.after(0, lambda c=translated_text: self.update_text_safe(self.translated_textbox, c))

                # --- 3. Text-to-Speech Generation ---
                if not translated_text or not translated_text.strip():
                    continue

                output_audio = f"temp_out_{np.random.randint(1000, 9999)}.mp3"
                selected_voice = output_codes["tts"]
                
                try:
                    communicate = edge_tts.Communicate(translated_text, selected_voice)
                    await communicate.save(output_audio)
                except Exception as tts_error:
                    print(f"⚠️ [TTS Error] Skipped audio generation: {tts_error}")
                    continue

                # --- 4. Play Audio (Invisibly via Pygame) ---
                if os.path.exists(output_audio):
                    pygame.mixer.music.load(output_audio)
                    pygame.mixer.music.play()
                    
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.05)
                    
                    pygame.mixer.music.unload()

                # --- 5. Cleanup Files ---
                for file_path in [mic_file, output_audio]:
                    if os.path.exists(file_path):
                        try: os.remove(file_path)
                        except: pass

            except Exception as e:
                print(f"Error in consumer execution pipeline: {e}")
            finally:
                self.audio_queue.task_done()


if __name__ == "__main__":
    app = VoxFlowApp()
    app.mainloop()