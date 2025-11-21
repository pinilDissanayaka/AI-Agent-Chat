import base64
import io
from typing import Optional, Dict, Any
import google.generativeai as genai
from config import config


class ImageGenerator:
    """Handle image generation using Gemini/Imagen API"""
    
    def __init__(self):
        """
        Initialize the image generator with the Gemini API key.
        
        Raises:
            ValueError: If GEMINI_API_KEY is not found in environment variables.
        """
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def detect_image_request(self, text: str) -> bool:
        """
        Detect if the user is requesting an image generation
        
        Args:
            text: User input text
            
        Returns:
            True if image generation is requested
        """
        image_keywords = [
            'generate image', 'create image', 'draw', 'make an image',
            'show me image', 'visualize', 'picture of', 'illustration of',
            'generate a picture', 'create a picture', 'image of', 'photo of',
            'generate imge', 'create imge', 'make imge', 'imge of',  # Common typos
            'gen image', 'gen imge', 'draw me', 'paint', 'sketch'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in image_keywords)
    
    def extract_image_prompt(self, text: str) -> str:
        """
        Extract the actual image prompt from user text
        
        Args:
            text: Full user message
            
        Returns:
            Cleaned prompt for image generation
        """
        # Remove common request phrases
        removal_phrases = [
            'generate image of', 'create image of', 'draw me', 
            'make an image of', 'show me image of', 'generate a picture of',
            'create a picture of', 'please', 'can you', 'could you'
        ]
        
        prompt = text.lower()
        for phrase in removal_phrases:
            prompt = prompt.replace(phrase, '')
        
        return prompt.strip()
    
    def generate_image_description(self, prompt: str) -> Dict[str, Any]:
        """
        Use Gemini to generate a detailed image description
        This is used because Gemini doesn't have direct image generation,
        but can describe what an image should look like
        
        Args:
            prompt: User's image request
            
        Returns:
            Dict with description and metadata
        """
        try:
            system_prompt = f"""You are an AI that helps create detailed image descriptions for image generation.
            
User request: {prompt}

Generate a detailed, vivid description that could be used to create this image. Include:
- Main subject and its characteristics
- Setting/background
- Colors and lighting
- Style (realistic, artistic, cartoon, etc.)
- Mood/atmosphere
- Important details

Keep it concise but descriptive (2-3 sentences)."""

            response = self.model.generate_content(system_prompt)
            description = response.text
            
            return {
                "success": True,
                "description": description,
                "original_prompt": prompt,
                "note": "Image description generated using Gemini. For actual image generation, you would need DALL-E, Stable Diffusion, or Imagen API."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "original_prompt": prompt
            }
    
    def generate_with_dalle(self, prompt: str, size: str = "1024x1024") -> Optional[str]:
        """
        Generate image using DALL-E (requires OpenAI API)
        
        Args:
            prompt: Image description
            size: Image size (256x256, 512x512, 1024x1024)
            
        Returns:
            Base64 encoded image or None
        """
        try:
            from openai import OpenAI
            
            if not config.OPENAI_API_KEY:
                return None
            
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality="standard",
                n=1,
            )
            
            # Get image URL
            image_url = response.data[0].url
            return image_url
            
        except Exception as e:
            print(f"DALL-E generation error: {e}")
            return None
    
    def create_placeholder_image(self, text: str) -> str:
        """
        Create a placeholder image with text
        
        Args:
            text: Text to display on image
            
        Returns:
            Base64 encoded image
        """
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (512, 512), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            line = ' '.join(current_line)
            if len(line) > 40:
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        y = 200
        for line in lines[:5]:  
            d.text((20, y), line, fill=(255, 255, 255), font=font)
            y += 30
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
