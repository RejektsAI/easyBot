# easyBot

easyBot: A versatile AI chatbot with voice interaction capabilities, powered by OpenAI's GPT and Whisper models.

## Key Features

- Text-based chat interface
- Voice-based interaction with speech recognition and text-to-speech capabilities
- Configurable AI model settings
- Ability to switch between text and voice modes
- Conversation history management
- Custom system prompts

## Project Structure

- `main.py`: The main Python script that runs the easyBot application
- `config.json`: Configuration file for customizing easyBot settings
- `run.bat`: Batch file for easy execution on Windows systems
- `TEMP/`: Directory created by the application for temporary audio files

## Requirements

- Python 3.x
- OpenAI API key
- Required Python packages (install via `pip install -r requirements.txt`):
  - numpy
  - sounddevice
  - scipy
  - faster_whisper
  - gtts
  - pydub
  - openai

## Setup and Usage

1. Clone the repository or download the project files.
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   - On Windows: Double-click the `run.bat` file or run it from the command line.
   - On other systems: Run `python main.py` in your terminal.

5. The application will start in text chat mode. Type your messages and press Enter to interact with the AI.
6. Use the following commands during chat:
   - `/call`: Switch to voice mode (type '/hang_up' to return to text chat)
   - `/clear`: Clear conversation history
   - `/load [filename]`: Load a different configuration file
   - `/end`, `/exit`, or `/close`: End the chat session

## Configuration

The `config.json` file allows you to customize various aspects of the easyBot:

- AI model settings (GPT model, temperature, tokens, etc.)
- Speech recognition model settings
- Audio processing parameters
- File paths for temporary audio files
- System prompt for setting the AI's behavior and context

Modify the `config.json` file to adjust these settings according to your preferences.

## Note

- This project requires an active internet connection for API calls to OpenAI services. Ensure you have a valid OpenAI API key and sufficient credits for using the GPT and Whisper models.
- The application creates a `TEMP` folder in the project directory to store temporary audio files for speech-to-text and text-to-speech operations.

## Credits

This project utilizes several open-source libraries and services:

- [OpenAI API](https://openai.com/): For GPT language models and Whisper speech recognition.
- [NumPy](https://numpy.org/): For numerical computing.
- [SoundDevice](https://python-sounddevice.readthedocs.io/): For audio input and output.
- [SciPy](https://www.scipy.org/): For signal processing.
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper): For efficient speech recognition.
- [gTTS (Google Text-to-Speech)](https://gtts.readthedocs.io/): For text-to-speech conversion.
- [PyDub](https://github.com/jiaaro/pydub): For audio file manipulation.
- [LiveWhisper](https://github.com/Nikorasu/LiveWhisper): For real-time speech recognition.

Special thanks to the developers and contributors of these projects for making this application possible.
