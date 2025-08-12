"""
Document processing module for Sheria Kiganjani
"""
import os
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from langdetect import detect
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and OCR operations"""
    
    SUPPORTED_FORMATS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    
    def __init__(self, upload_dir: str = "uploads"):
        """Initialize document processor"""
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        
    async def process_document(self, file_content: bytes, filename: str) -> Dict:
        """Process uploaded document and extract text"""
        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # Save original file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.upload_dir, f"{timestamp}_{filename}")
            with open(save_path, "wb") as f:
                f.write(file_content)
            
            # Extract text based on file type
            if ext == '.pdf':
                text = await self._process_pdf(file_content)
            else:
                text = await self._process_image(file_content)
            
            # Detect language
            try:
                language = detect(text)
            except:
                language = 'en'  # Default to English if detection fails
            
            # Get document metadata
            metadata = self._extract_metadata(text)
            
            return {
                "text": text,
                "language": language,
                "metadata": metadata,
                "file_path": save_path,
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
    
    async def _process_pdf(self, content: bytes) -> str:
        """Process PDF document and extract text"""
        try:
            # Convert PDF to images
            images = convert_from_bytes(content)
            
            # Process each page
            text_parts = []
            for img in images:
                # Convert PIL image to OpenCV format
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # Process image and extract text
                page_text = await self._process_image_cv2(img_cv)
                text_parts.append(page_text)
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    async def _process_image(self, content: bytes) -> str:
        """Process image file and extract text"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(content))
            
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process image and extract text
            return await self._process_image_cv2(img_cv)
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
    
    async def _process_image_cv2(self, image: np.ndarray) -> str:
        """Process image using OpenCV and extract text"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh)
            
            # Deskew if needed
            angle = self._get_skew_angle(denoised)
            if abs(angle) > 0.5:
                denoised = self._rotate_image(denoised, angle)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(denoised)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in CV2 processing: {str(e)}")
            raise
    
    def _get_skew_angle(self, image: np.ndarray) -> float:
        """Detect skew angle of text in image"""
        try:
            # Find all contours
            contours, _ = cv2.findContours(
                image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            
            angles = []
            for contour in contours:
                if len(contour) < 5:  # Need at least 5 points for ellipse
                    continue
                    
                # Fit ellipse and get angle
                try:
                    _, _, angle = cv2.fitEllipse(contour)
                    angles.append(angle)
                except:
                    continue
            
            if angles:
                # Use median angle
                return np.median(angles) - 90
            return 0
            
        except Exception:
            return 0
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        try:
            # Get image dimensions
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Perform rotation
            rotated = cv2.warpAffine(
                image, rotation_matrix, (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return rotated
            
        except Exception as e:
            logger.error(f"Error rotating image: {str(e)}")
            return image
    
    def _extract_metadata(self, text: str) -> Dict:
        """Extract metadata from document text"""
        metadata = {
            "document_type": self._detect_document_type(text),
            "dates": self._extract_dates(text),
            "parties": self._extract_parties(text),
            "case_numbers": self._extract_case_numbers(text)
        }
        return metadata
    
    def _detect_document_type(self, text: str) -> str:
        """Detect the type of legal document"""
        # Common legal document keywords
        document_types = {
            "contract": ["agreement", "contract", "deed", "mkataba"],
            "court_filing": ["petition", "pleading", "affidavit", "kiapo"],
            "judgment": ["judgment", "decree", "order", "hukumu"],
            "legislation": ["act", "regulation", "statute", "sheria"],
            "correspondence": ["letter", "notice", "memo", "barua"]
        }
        
        text_lower = text.lower()
        for doc_type, keywords in document_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return doc_type
        
        return "unknown"
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text"""
        import re
        
        # Date patterns (customize for your needs)
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # DD/MM/YYYY
            r'\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))
    
    def _extract_parties(self, text: str) -> List[str]:
        """Extract party names from text"""
        import re
        
        # Common party indicators
        indicators = [
            r'(?:plaintiff|defendant|applicant|respondent|petitioner):?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:between|vs\.?|v\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:mdai|mdaiwa|mlalamishi|mlalamikiwa):?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        parties = []
        for pattern in indicators:
            matches = re.findall(pattern, text)
            parties.extend(matches)
        
        return list(set(parties))
    
    def _extract_case_numbers(self, text: str) -> List[str]:
        """Extract case numbers from text"""
        import re
        
        # Case number patterns (customize for your jurisdiction)
        patterns = [
            r'Case\s+No\.?\s*\d+\s*of\s*\d{4}',
            r'Civil\s+Case\s+No\.?\s*\d+\s*of\s*\d{4}',
            r'Criminal\s+Case\s+No\.?\s*\d+\s*of\s*\d{4}',
            r'Kesi\s+Na\.?\s*\d+\s*ya\s*\d{4}'
        ]
        
        case_numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            case_numbers.extend(matches)
        
        return list(set(case_numbers))
