import streamlit as st
import os
from PIL import Image
from dotenv import load_dotenv

# Import your backend modules
# (Assuming you saved the previous scripts as generator.py and search_agent.py in the backend/ folder)
from backend.search_agent import TrendResearcherAgent
from backend.generator import DesignCreatorIPAdapter

# Load environment variables
load_dotenv()

# ==========================================
# 1. Page Configuration & State Management
# ==========================================
st.set_page_config(page_title="AI Design Studio", page_icon="🎨", layout="wide")

# Initialize session state to remember search results across button clicks
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_images" not in st.session_state:
    st.session_state.selected_images = []

# ==========================================
# 2. Heavy Resource Caching (FAANG Standard)
# ==========================================
# We cache these so they only load into RAM/VRAM once when the app starts.
@st.cache_resource
def load_search_agent():
    return TrendResearcherAgent()

@st.cache_resource
def load_vision_engine():
    return DesignCreatorIPAdapter()

try:
    search_agent = load_search_agent()
    vision_engine = load_vision_engine()
except Exception as e:
    st.error(f"Failed to load backend models: {e}")
    st.stop()

# ==========================================
# 3. UI Layout
# ==========================================
st.title("🎨 AI Custom Design Studio")
st.markdown("Search for trending fashion, cultural attire, or custom card designs, select your favorites, and let the IP-Adapter blend them into a brand new design.")

# Split screen into Left (Inspiration) and Right (Generation)
col_left, col_right = st.columns([1, 1], gap="large")

# --- LEFT COLUMN: Search & Inspiration ---
with col_left:
    st.subheader("1. Gather Inspiration")
    
    # Automated Agent Search
    search_query = st.text_input(
        "Ask the AI to find trends:", 
        placeholder="e.g. Trending minimalist wedding cards, or royal blue velvet cultural attire"
    )
    
    if st.button("🔍 Search Pinterest/Web"):
        with st.spinner("Agent is reasoning and searching..."):
            results = search_agent.run(search_query)
            st.session_state.search_results = results
            st.session_state.selected_images = [] # Reset selections on new search

    # Display Search Results in a Grid
    if st.session_state.search_results:
        st.markdown("**Select up to 4 images to blend:**")
        
        # Create a dynamic 2-column grid for images
        cols = st.columns(2)
        for idx, url in enumerate(st.session_state.search_results):
            with cols[idx % 2]:
                st.image(url,width="stretch")
                # Checkbox to select image
                if st.checkbox(f"Select Image {idx+1}", key=f"chk_{idx}"):
                    if url not in st.session_state.selected_images:
                        st.session_state.selected_images.append(url)
                else:
                    if url in st.session_state.selected_images:
                        st.session_state.selected_images.remove(url)

    # Optional: Manual Upload
    st.markdown("---")
    uploaded_files = st.file_uploader("Or upload client references (JPG/PNG)", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    if uploaded_files:
        st.success(f"Received {len(uploaded_files)} local files! (Will be processed in Vision Engine)")

# --- RIGHT COLUMN: Blending & Generation ---
with col_right:
    st.subheader("2. AI Design Blender")
    
    # Show what we are about to blend
    total_selected = len(st.session_state.selected_images) + (len(uploaded_files) if uploaded_files else 0)
    st.markdown(f"**Selected Moodboard: {total_selected} images**")
    
    if total_selected == 0:
        st.info("👈 Run a search and select images from the left panel, or upload files to begin.")
    else:
        # Mini gallery of selected images
        sel_cols = st.columns(total_selected)
        col_idx = 0
        
        for img_url in st.session_state.selected_images:
            with sel_cols[col_idx]:
                st.image(img_url, caption=f"Web Style {col_idx+1}")
            col_idx += 1
            
        if uploaded_files:
            for file in uploaded_files:
                with sel_cols[col_idx]:
                    st.image(file, caption="Client Upload")
                col_idx += 1
                
        st.markdown("---")
        
        # Guidance Prompt
        guidance = st.text_area(
            "Additional Instructions (Optional):", 
            value="A highly detailed, intricate custom design, vivid colors, rich patterns, professional studio photography, high resolution.",
            height=100
        )
        
        # Generate Button
        if st.button("✨ Generate New Design", type="primary"):
            if total_selected > 4:
                st.warning("Please select a maximum of 4 images total for best results.")
            else:
                with st.spinner("Vision Engine is mathematically blending features... This takes 10-30 seconds."):
                    try:
                        # Convert uploaded files to PIL Images for the backend
                        client_pil_images = []
                        if uploaded_files:
                            for file in uploaded_files:
                                client_pil_images.append(Image.open(file))
                                
                        # Call the backend vision generator
                        generated_images = vision_engine.generate_blended_design(
                            image_urls=st.session_state.selected_images,
                            local_images=client_pil_images,
                            text_prompt=guidance,
                            num_outputs=1
                        )
                        
                        st.success("Design Generated Successfully!")
                        # Display the final output
                        st.image(generated_images[0], caption="Final AI Blended Design")
                        
                        # Add a download button for the client
                        st.download_button(
                            label="Download Design",
                            data=generated_images[0].tobytes(),
                            file_name="ai_custom_design.png",
                            mime="image/png"
                        )
                        
                    except Exception as e:
                        st.error(f"Generation Failed: {e}")