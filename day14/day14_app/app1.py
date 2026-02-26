import streamlit as st
import ollama
import base64
from io import BytesIO

st.title("Text Extraction from Image")

# Upload an image file
uploaded_file = st.file_uploader(
	"Choose an image...",
	type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
	# Display the uploaded image (use `width` instead of deprecated `use_column_width`)
	st.image(uploaded_file, caption="Uploaded Image", width=700)

	# When the button is pressed
	if st.button("Extract Text"):
		# Convert uploaded file to base64
		uploaded_file.seek(0)
		image_data = base64.standard_b64encode(uploaded_file.read()).decode("utf-8")
        
		# Call the ollama chat function
		res = ollama.chat(
			model="llama3.2-vision:latest",
			messages=[
				{
					"role": "user",
					"content": "extract text from this image",
					"images": [image_data]
				}
			]
		)

		# Display the extracted text
		st.subheader("Extracted Text:")
		st.write(res["message"]["content"])

