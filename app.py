import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Import the architecture
from model import MalariaCNN

# --- CONFIGURATION ---
MODEL_PATH = 'malaria_cnn_final.pth'
CLASSES = ['Parasitized 🔴', 'Uninfected 🟢'] 

# --- 1. MODEL LOADER ---
@st.cache_resource
def load_model():
    """
    Loads the trained model weights. 
    Cached so it doesn't reload every time a user clicks a button.
    """
    # Initialize the blank architecture
    model = MalariaCNN()
    
    # Load the saved weights
    # map_location='cpu' ensures it runs on the web server (which likely has no GPU)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval() # Turn off training specific layers like Dropout
        return model
    except FileNotFoundError:
        st.error(f"Error: Could not find '{MODEL_PATH}'. Please make sure the file is in the same folder as app.py")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 2. IMAGE PREPROCESSOR ---
def process_image(image):
    """
    Transforms the user's image to match exactly what the model saw during training.
    """
    # Define the standard transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)), # Resize to 128x128
        transforms.ToTensor(),         # Convert to numbers (0-1)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize
    ])
    
    # Apply transform
    tensor = transform(image)
    
    # Add Batch Dimension: [3, 128, 128] -> [1, 3, 128, 128]
    # The model expects a batch, even if it's just one image.
    tensor = tensor.unsqueeze(0)
    return tensor

# --- 3. THE USER INTERFACE ---
st.set_page_config(page_title="Malaria AI Diagnosis", page_icon="🦟")

st.title("🦟 Malaria AI Diagnostician")
st.markdown("""
This tool uses a **Deep Convolutional Neural Network (CNN)** to detect malaria parasites in microscopic blood smear images.
* **Accuracy:** ~97%
* **Model:** Custom PyTorch Architecture (4-Layer CNN)
* **Training Data:** NIH Malaria Dataset
""")

st.divider()

# File Uploader
uploaded_file = st.file_uploader("Upload a Microscopic Cell Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the user's image
    image = Image.open(uploaded_file).convert('RGB') # Ensure it's RGB (3 channels)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Sample", use_column_width=True)
    
    with col2:
        st.write("### Diagnostic Result")
        
        # Analyze Button
        if st.button("Run AI Analysis", type="primary"):
            with st.spinner("Scanning cell patterns..."):
                # A. Load Model
                model = load_model()
                
                if model:
                    # B. Process Image
                    input_tensor = process_image(image)
                    
                    # C. Get Prediction
                    with torch.no_grad():
                        output = model(input_tensor)
                        # Softmax turns raw scores into probabilities (0 to 1)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    # D. Display Result
                    result_class = CLASSES[predicted_idx.item()]
                    confidence_score = confidence.item() * 100
                    
                    # Logic for displaying Red (Sick) or Green (Healthy)
                    if predicted_idx.item() == 0: # Parasitized
                        st.error(f"**Diagnosis:** {result_class}")
                        st.write("⚠️ Parasitic structures detected.")
                    else: # Uninfected
                        st.success(f"**Diagnosis:** {result_class}")
                        st.write("✅ Cell appears healthy.")
                        
                    st.metric("Model Confidence", f"{confidence_score:.2f}%")