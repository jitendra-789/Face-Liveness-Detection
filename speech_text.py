import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Use the microphone as the audio source
with sr.Microphone() as source:
    print("Please say something...")
    # Listen for the first phrase and extract it into audio data
    audio_data = recognizer.listen(source)
    print("Recognizing...")

    try:
        # Use Google Web Speech API to transcribe audio data to text
        text = recognizer.recognize_google(audio_data)
        print("You said: " + text)
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
