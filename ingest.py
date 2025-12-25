from vectorstore import VectorStore
from vl_caption import generate_visual_caption
import os 

vs = VectorStore()

def chunk_text(text, max_len=300):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    current = []
    
    for word in words:
        if len(' '.join(current + [word])) > max_len:
            chunks.append(' '.join(current))
            current = [word]
        else:
            current.append(word)
    
    if current:
        chunks.append(' '.join(current))
    
    return chunks

texts, sources = [], []
documents_path = "documents"

for file in os.listdir(documents_path):
    if file.endswith('.txt'):
        with open(os.path.join(documents_path, file)) as f:
            content = f.read()
            chunks = chunk_text(content)
            texts.extend(chunks)
            sources.extend([file] * len(chunks))
    elif file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        video_path = f'documents/{file}'
        print(f'Extracting caption from {file}')
        visual_caption = generate_visual_caption(video_path)
        full_text = f"[VIDEO FIRST FRAME DESCRIPTION]\n{visual_caption}\n\n[VIDEO FILE]\n{file}"
        chunks = chunk_text(full_text)
        texts.extend(chunks)
        sources.extend([f"VIDEO_SMOLVLM: {file}"] * len(chunks))  

vs.add_documents(texts, sources)
print(f"✅ Vector database built: {len(texts)} chunks!")

