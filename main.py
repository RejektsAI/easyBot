import os, time, json
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from faster_whisper import WhisperModel
from collections import deque
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI
from IPython.display import display, Audio, clear_output

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
config_file = "config.json"

with open(config_file, 'r') as f:
        config = json.load(f)

def load_config(config_file):
    global system_prompt, default_gpt, GPT_MODEL, TOP_P, TEMP, TOKENS, F_PEN, P_PEN
    with open(config_file, 'r') as f:
        config = json.load(f)
    system_prompt = {"role": "system", "content": config['system_prompt']}
    default_gpt = config['default_gpt']
    GPT_MODEL = config['gpt_model']
    TOP_P = config['top_p']
    TEMP = config['temp']
    TOKENS = config['tokens']
    F_PEN = config['f_pen']
    P_PEN = config['p_pen']

load_config(config_file)

conversation = deque(maxlen=10)

whisper_model = WhisperModel(
    config['whisper_model']['name'],
    compute_type=config['whisper_model']['compute_type'],
    device=config['whisper_model']['device']
)

SampleRate = config['sample_rate']
BlockSize = config['block_size']
Threshold = config['threshold']
Vocals = config['vocals']
EndBlocks = config['end_blocks']
os.environ['TEMP'] = os.path.join(os.getcwd(), 'temp_files')
os.makedirs(os.environ['TEMP'], exist_ok=True)
input_audio = os.path.join(os.environ['TEMP'], config['file_paths']['stt'])
output_audio = os.path.join(os.environ['TEMP'], config['file_paths']['tts'])
audio_data = None
running = True
padding = 0
prevblock = buffer = np.zeros((0,1))
fileready = False

def llm_response(list):
    api_response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=list,
            top_p = TOP_P,
            temperature=TEMP,
            max_tokens=TOKENS,
            frequency_penalty=F_PEN,
            presence_penalty=P_PEN,
            response_format={"type": "text"}
        )
    return api_response.choices[0].message.content

def speak_up(string, language):
    try:
        tts = gTTS(string, lang=language)
        tts.save(output_audio)
        time.sleep(0.2)
        audio = AudioSegment.from_file(output_audio)
        play(audio)
    except Exception as e:
        print(f"Error in speak_up: {e}")

def type_up(string, speed=0.05):
    new_string = string.encode('utf-8').decode('utf-8')
    for character in "Her: "+new_string:
        print(character, end='', flush=True)
        time.sleep(speed)
    print("",end='\n')
    return

def temp_audio(audio_data):
    if audio_data.ndim == 2 and audio_data.shape[1] == 1:
        audio_data = audio_data.flatten()  # Convert to 1D array

    try:
        wavfile.write(input_audio, int(SampleRate), audio_data)
        print(f"\033[90mAudio successfully saved to {input_audio}.\033[0m")
        if os.path.getsize(input_audio) == 0:  # Check if the file is empty
            print(f"Warning: {input_audio} is empty after saving.")
    except Exception as e:
        print(f"Error caching audio: {e}")

def transcribe_audio(audio_path):
    transcription = []
    if not os.path.isfile(audio_path):
        print(f"Audio file {audio_path} does not exist.")
        return "", ""  # Return empty strings for both transcription and language

    try:
        # Call the model to transcribe the audio
        result = whisper_model.transcribe(audio_path)
        
        # Unpack the results correctly: the first element is a generator
        segments = list(result[0])  # Convert the generator to a list
        info = result[1]  # The second element is the transcription info

        # Append the text of each segment to the transcription list
        for segment in segments:
            transcription.append(segment.text)

        return "".join(transcription), info.language  # Return transcription and language
    except Exception as e:
        print(f"Error in transcribing audio: {e}")
        return "", ""  # Return empty strings on error

def callback(indata, frames, time, status):
    global audio_data, padding, prevblock, buffer, fileready

    if not any(indata):
        print('\033[31m.\033[0m', end='', flush=True)
        return  # No audio captured, do nothing.

    freq = np.argmax(np.abs(np.fft.rfft(indata[:, 0]))) * SampleRate / frames
    is_speech = np.sqrt(np.mean(indata**2)) > Threshold and Vocals[0] <= freq <= Vocals[1]

    if is_speech:
        print('\033[90m.\033[0m', end='', flush=True)  # Feedback while speaking
        if padding < 1:
            buffer = prevblock.copy()
        buffer = np.concatenate((buffer, indata))
        padding = EndBlocks
    else:
        padding -= 1
        if padding > 1:
            buffer = np.concatenate((buffer, indata))
        elif padding < 1:
            if buffer.shape[0] > SampleRate:
                fileready = True
                audio_data = buffer
                temp_audio(audio_data)
            buffer = np.zeros((0, 1))  # Reset buffer
            print("\033[2K\033[0G", end='', flush=True)
        else:
            prevblock = indata.copy()

def audio_call():
    global fileready, conversation

    if not fileready:
        return

    print("\n\033[90mTranscribing..\033[0m")
    try:
        time.sleep(0.1)
        transcription, language = transcribe_audio(input_audio)
        print(f"You: {transcription}")
        user_content = {"role": "user", "content": transcription}
        conversation.append(user_content)
        messages = [system_prompt] + list(conversation)
        ai_message = llm_response(messages)
        ai_content = {"role": "assistant", "content": ai_message}
        conversation.append(ai_content)
        print(f"Her: {ai_message}")
        speak_up(ai_message, language)
        

    except Exception as e:
        print(f"Error during processing: {e}")

    print('\n')
    fileready = False

def listen():
    global running
    print("\033[32mListening.. \033[32m(Press Ctrl+C to end the call)\033[0m \n")

    try:
        with sd.InputStream(
            channels=1,
            callback=callback,
            blocksize=int(SampleRate * BlockSize / 1000),
            samplerate=SampleRate
        ):
            while running:
                audio_call()

    except KeyboardInterrupt:
        print("\n\033[31mStopped listening.\033[0m")
    except Exception as e:
        print(f"\n\033[31mError: {e}\033[0m")

def check_for_command(user_input):
    global conversation
    if user_input.lower() in ('/end', '/exit', '/close'):
        print("\033[31mEnding chat.\033[0m")
        exit()
    elif user_input.lower() == '/call':
        print("\033[32mStarting audio call...\033[0m")
        listen()
        print("\033[32mReturning to text chat... \033[32m(Type '/call' to start audio call)\033[0m")
        return True
    elif "/load" in user_input.split() and len(user_input.split()) == 2:
        try:
            config_file = user_input.split()[1]
            print(f"\033[32mLoading config file: {config_file}\033[0m")
            load_config(config_file)
        except Exception as e:
            print(f"\033[31mError: {e}\033[0m")
        return True
    elif user_input.lower() == '/clear':
        conversation = []
        print("\033[32mClearing conversation history\033[0m",end="")
        type_up(".....")
        os.system('cls' if os.name == 'nt' else 'clear')    
        print("\033[32mStarting text chat... \033[32m(Type '/call' to start audio call)\033[0m")   
        return True
    else:
        return False
    
def text_chat():
    global conversation
    print("\033[32mStarting text chat... \033[32m(Type '/call' to start audio call)\033[0m")
    
    while True:
        user_input = input("You: ")
        if not check_for_command(user_input):
            user_content = {"role": "user", "content": user_input}
            conversation.append(user_content)
            messages = [system_prompt] + list(conversation)
            time.sleep(1)
            ai_message = llm_response(messages)
            ai_content = {"role": "assistant", "content": ai_message}
            conversation.append(ai_content)
            print("Her: ",end="")
            type_up(ai_message)
            

# Start with text chat
text_chat()
