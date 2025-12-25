import cv2 as cv 
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

class VLMCaption:
    """Video caption model using SmolVLM for local GPU"""
    
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-256M-Instruct"):
        self.model_name = model_name
        
        self.vlm_model = AutoModelForVision2Seq.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16, 
            device_map='auto',
            cache_dir='./',
            trust_remote_code=True
        )
        
        self.vlm_processor = AutoProcessor.from_pretrained(self.model_name)

    def get_first_frame(self, video_path):
        """Get first frame from video"""
        cap = cv.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def generate_visual_caption(self, video_path, prompt="Describe this image in detail for a video summary."):
        """Generate caption using SmolVLM"""
        image = self.get_first_frame(video_path)
        if image is None:
            return "Error: Could not read video frame."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs_text = self.vlm_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        inputs = self.vlm_processor(
            text=[inputs_text],
            images=[image],
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            output = self.vlm_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.vlm_processor.tokenizer.eos_token_id
            )
        
        caption = self.vlm_processor.decode(output[0], skip_special_tokens=True)
        caption = caption.split(prompt)[-1].strip()
        return caption[:200]

if __name__ == "__main__":
    print(torch.cuda.is_available())
    vlm_caption = VLMCaption()
    # Test with a video file
    result = vlm_caption.generate_visual_caption("documents/received_1071581410560701.mp4")
    print(result)
