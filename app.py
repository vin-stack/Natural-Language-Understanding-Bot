import streamlit as st
import speech_recognition as sr
import spacy
python -m spacy download en_core_web_sm

from textblob import TextBlob
import nltk
nltk.download('brown')
# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Define keyword for triggering action
KEYWORD = "forward"

# Create a function to record and transcribe audio
def transcribe_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say something!")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        st.write("You said: ", text)

        # Check for keyword
        if KEYWORD in text.lower():
            st.write("Keyword detected: ", KEYWORD)

        # Perform NLU with spaCy and TextBlob
        doc = nlp(text)
        blob = TextBlob(text)

        # Analyze grammar with spaCy
        grammar = []
        for token in doc:
            if token.pos_ != "SPACE":
                grammar.append((token.text, token.pos_, token.tag_))
        if grammar:
            st.write("Grammar analysis:")
            for token in grammar:
                st.write(f"{token[0]} ({token[1]}, {token[2]})")

        # Extract noun phrases with TextBlob
        noun_phrases = blob.noun_phrases
        if noun_phrases:
            st.write("Noun phrases:", ", ".join(noun_phrases))

        # Perform sentiment analysis with TextBlob
        sentiment = blob.sentiment
        st.write("Sentiment polarity:", sentiment.polarity)
        st.write("Sentiment subjectivity:", sentiment.subjectivity)

        # Perform part-of-speech tagging with TextBlob
        pos_tags = blob.tags
        if pos_tags:
            st.write("Part-of-speech tags:")
            for tag in pos_tags:
                st.write(f"{tag[0]} ({tag[1]})")

    except:
        st.write("Sorry, could not recognize your voice.")

# Create a Streamlit app
def app():
    st.title("Speech Recognition App with NLU and Keyword Triggering")
    st.write("Click the button below to start recording.")
    if st.button("Record"):
        transcribe_audio()

if __name__ == '__main__':
    app()
