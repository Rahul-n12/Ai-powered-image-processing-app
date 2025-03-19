import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
from dotenv import load_dotenv
from io import BytesIO
import os
import tempfile
import imghdr

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Function to generate an image using DALL-E 3
def generate_edited_image(image_path, prompt):
    try:
        with open(image_path, "rb") as image_file:
            response = client.images.edit(
                image=image_file,  # Provide the image file directly
                prompt=prompt,      # Your concise edit prompt
                n=1,
                size="1024x1024"  # Or your desired size
            )
            print(response)
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        st.error(f"Image edit failed: {e}")
        return None

# Function to transcribe audio using OpenAI Whisper API
def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

# Main Streamlit app
def main():
    st.title("Speech-to-Image Generator")

    # Audio input/upload
    st.write("Provide your audio input describing the image:")
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])  

    uploaded_image = st.file_uploader("Upload an initial image (<4MB)", type=["png", "jpg", "jpeg"])  # Allow JPEG/JPG as well
    initial_image = None
    
    if uploaded_image:
        try:
            image_size_bytes = len(uploaded_image.getvalue())
            image_size_mb = image_size_bytes / (1024 * 1024)

            if image_size_mb > 4:
                st.error(f"Image size is too large ({image_size_mb:.2f} MB). Please upload an image less than 4MB.")
            else:
                file_type = imghdr.what(uploaded_image)  # Check actual file type

                if file_type == 'png':
                    initial_image = Image.open(uploaded_image)
                elif file_type in ('jpeg', 'jpg'): # Handle JPEG/JPG
                    initial_image = Image.open(uploaded_image).convert("RGBA") # Convert to RGBA first
                    initial_image = initial_image.convert("PNG")  # Then to PNG
                else:
                    st.error(f"Unsupported image format: {file_type}. Please upload a PNG or JPEG image.")

                if initial_image: # Only display if successfully opened/converted
                    st.image(initial_image, caption="Uploaded Initial Image", use_container_width=True) 

        except Exception as e:
            st.error(f"Error processing image: {e}")
            initial_image = None

    if audio_file and initial_image:  # Check for both audio and image
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as temp_audio_file:
            temp_audio_file.write(audio_file.getvalue())
            temp_audio_path = temp_audio_file.name

        transcription = transcribe_audio(temp_audio_path)

        if transcription:
            st.success(f"Transcription: {transcription}")

            prompt = f"{transcription}"  # Edit prompt from speech

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file:  # Use .png suffix
                initial_image.save(temp_image_file, format="PNG")  # Save as PNG
                temp_image_path = temp_image_file.name

            image_url = generate_edited_image(temp_image_path, prompt)

            if image_url:
                try:
                    response = requests.get(image_url, stream=True)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption="Generated Image", use_container_width=True)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching image: {e}")
            else:
                st.error("Image edit failed.")

            os.remove(temp_image_path)  # Clean up the temporary image file

        os.remove(temp_audio_path)  # Clean up the temporary audio file

    elif audio_file is None and uploaded_image is None:
        st.info("Upload an audio file and an initial image to get started.")
    elif audio_file is None:
        st.info("Upload an audio file to get started.")
    elif uploaded_image is None:
        st.info("Upload an initial image to get started.")

if __name__ == "__main__":
    main()