import streamlit as st
import os
import zipfile
import gdown
import shutil

st.title("🔍 DEBUG MODE - Dataset Structure Check")

@st.cache_resource
def debug_download_dataset():
    """Download and show what's inside"""
   
    st.info("📥 Downloading dataset...")
   
    # Download
    file_id = "1E341LM0PcxGo9vG1FQAguahPHuz5pB89"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "dataset.zip"
   
    try:
        gdown.download(url, output, quiet=False, fuzzy=True)
       
        st.success("✅ Download complete!")
       
        # Show zip contents BEFORE extraction
        st.subheader("📦 Contents of dataset.zip:")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            for f in file_list[:20]:  # Show first 20 files
                st.text(f)
            if len(file_list) > 20:
                st.text(f"... and {len(file_list) - 20} more files")
       
        # Extract
        st.info("📂 Extracting...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(".")
       
        os.remove(output)
       
        # Show current directory structure
        st.subheader("📁 Current Directory After Extraction:")
       
        def show_tree(path, prefix="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return
           
            try:
                items = os.listdir(path)
                items = [i for i in items if not i.startswith('.')]  # Skip hidden files
               
                for i, item in enumerate(items[:10]):  # Limit to 10 items per level
                    item_path = os.path.join(path, item)
                    is_last = i == len(items) - 1
                   
                    if os.path.isdir(item_path):
                        st.text(f"{prefix}{'└── ' if is_last else '├── '}📁 {item}/")
                        new_prefix = prefix + ("    " if is_last else "│   ")
                        show_tree(item_path, new_prefix, max_depth, current_depth + 1)
                    else:
                        st.text(f"{prefix}{'└── ' if is_last else '├── '}📄 {item}")
               
                if len(items) > 10:
                    st.text(f"{prefix}... and {len(items) - 10} more items")
            except:
                pass
       
        show_tree(".")
       
        # Check all possible locations
        st.subheader("🔍 Checking Possible Dataset Locations:")
       
        possible_paths = [
            "./train",
            "./test",
            "./dataset/train",
            "./dataset/test",
            "./Dataset/train",
            "./Dataset/test",
            "./DATASET/train",
            "./DATASET/test"
        ]
       
        for path in possible_paths:
            exists = os.path.exists(path)
            if exists:
                st.success(f"✅ FOUND: {path}")
                # Show what's inside
                try:
                    contents = os.listdir(path)
                    st.text(f"   Contains: {contents}")
                except:
                    pass
            else:
                st.error(f"❌ NOT FOUND: {path}")
       
        return True
       
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False

if st.button("🔍 Download & Debug Dataset Structure"):
    debug_download_dataset()

st.markdown("---")
st.info("""
### What This Debug Tool Does:
1. Downloads your dataset.zip from Google Drive
2. Shows what's INSIDE the zip file
3. Extracts it
4. Shows the folder structure after extraction
5. Checks all possible locations for train/test folders

**After running this, take a screenshot and I'll give you the EXACT fix!**
""")