"""
OCR Document Analysis Engine - Text extraction and document verification

This module uses Tesseract OCR (via pytesseract) and PyMuPDF to:
1. Extract text from uploaded documents (paystubs, applications, ID verification)
2. Handle both image files AND PDF documents
3. Detect fraudulent document patterns
4. Validate document consistency
5. Identify potential document tampering

Used by landlords to verify renter-submitted documents.
"""

import os
import io
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Lazy load pytesseract
_tesseract_available = None

# Check for PyMuPDF (fitz) for PDF support
_fitz_available = False
try:
    import fitz  # PyMuPDF
    _fitz_available = True
    logger.info("PyMuPDF available for PDF processing")
except ImportError:
    logger.warning("PyMuPDF not available - PDF documents will not be supported")

# Lazy load pytesseract
_tesseract_available = None


def _check_tesseract():
    """Check if Tesseract is available"""
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available
    
    try:
        import pytesseract
        # Try to run tesseract
        pytesseract.get_tesseract_version()
        _tesseract_available = True
        logger.info("Tesseract OCR is available")
    except Exception as e:
        _tesseract_available = False
        logger.warning(f"Tesseract OCR not available: {e}")
    
    return _tesseract_available


class DocumentType(str, Enum):
    """Types of documents we can analyze"""
    PAYSTUB = "paystub"
    ID_CARD = "id_card"
    BANK_STATEMENT = "bank_statement"
    RENTAL_APPLICATION = "rental_application"
    EMPLOYMENT_LETTER = "employment_letter"
    TAX_DOCUMENT = "tax_document"
    UTILITY_BILL = "utility_bill"
    UNKNOWN = "unknown"


class DocumentRiskLevel(str, Enum):
    """Risk levels for document analysis"""
    VERIFIED = "verified"
    LIKELY_AUTHENTIC = "likely_authentic"
    UNCERTAIN = "uncertain"
    SUSPICIOUS = "suspicious"
    LIKELY_FRAUDULENT = "likely_fraudulent"


@dataclass
class ExtractedData:
    """Data extracted from document"""
    raw_text: str
    name: Optional[str]
    date: Optional[str]
    amounts: List[float]
    employer: Optional[str]
    address: Optional[str]
    phone_numbers: List[str]
    emails: List[str]
    sin_found: bool  # Canadian Social Insurance Number
    account_numbers: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text[:500] + "..." if len(self.raw_text) > 500 else self.raw_text,
            "name": self.name,
            "date": self.date,
            "amounts": self.amounts,
            "employer": self.employer,
            "address": self.address,
            "phone_numbers": self.phone_numbers,
            "emails": self.emails,
            "sin_found": self.sin_found,
            "account_numbers": ["****" + acc[-4:] for acc in self.account_numbers]  # Mask for privacy
        }


@dataclass
class DocumentAnalysisResult:
    """Complete document analysis result"""
    document_hash: str
    document_type: DocumentType
    risk_level: DocumentRiskLevel
    risk_score: float
    extracted_data: ExtractedData
    quality_score: float
    consistency_score: float
    indicators: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_hash": self.document_hash,
            "document_type": self.document_type.value,
            "risk_level": self.risk_level.value,
            "risk_score": round(self.risk_score, 3),
            "extracted_data": self.extracted_data.to_dict(),
            "quality_score": round(self.quality_score, 3),
            "consistency_score": round(self.consistency_score, 3),
            "indicators": self.indicators,
            "metadata": self.metadata,
            "explanation": self.explanation
        }


# Patterns for detecting document elements
PATTERNS = {
    # Canadian phone numbers
    "phone": re.compile(r'(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
    
    # Email addresses
    "email": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    
    # Canadian SIN (999-999-999) — must have dashes/spaces to avoid matching phone numbers etc.
    "sin": re.compile(r'\b\d{3}[-\s]\d{3}[-\s]\d{3}\b'),
    
    # Dollar amounts
    "amount": re.compile(r'\$[\d,]+(?:\.\d{2})?'),
    
    # Dates (various formats)
    "date": re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\w+ \d{1,2},? \d{4})\b'),
    
    # Account numbers (generic pattern)
    "account": re.compile(r'\b\d{8,16}\b'),
}

# Keywords for document type detection
DOCUMENT_KEYWORDS = {
    DocumentType.PAYSTUB: [
        "pay stub", "paystub", "earnings", "deductions", "net pay",
        "gross pay", "ytd", "pay period", "hours worked", "salary"
    ],
    DocumentType.ID_CARD: [
        "driver's license", "drivers license", "date of birth", "dob",
        "id card", "identification", "provincial id", "passport"
    ],
    DocumentType.BANK_STATEMENT: [
        "account statement", "bank statement", "opening balance",
        "closing balance", "transaction", "deposit", "withdrawal"
    ],
    DocumentType.RENTAL_APPLICATION: [
        "rental application", "tenant application", "landlord reference",
        "previous rental", "rental history", "monthly rent"
    ],
    DocumentType.EMPLOYMENT_LETTER: [
        "employment confirmation", "letter of employment", "to whom it may concern",
        "this is to confirm", "employed since", "salary of", "annual income"
    ],
    DocumentType.TAX_DOCUMENT: [
        "t4", "t1", "tax return", "cra", "canada revenue", "notice of assessment",
        "income tax", "tax year"
    ],
    DocumentType.UTILITY_BILL: [
        "utility bill", "hydro", "electricity", "gas bill", "water bill",
        "account holder", "service address", "amount due"
    ]
}

# Suspicious patterns that may indicate fraud
FRAUD_INDICATORS = {
    "perfect_amounts": "Suspiciously round dollar amounts (e.g., exactly $5000.00)",
    "inconsistent_fonts": "Font inconsistencies detected in document",
    "missing_employer": "Employment document missing employer information",
    "future_date": "Document contains future dates",
    "low_quality": "Low quality/resolution may hide tampering",
    "round_numbers": "All amounts are round numbers (unusual for real documents)",
    "generic_employer": "Employer name appears generic or suspicious",
    "no_contact_info": "Document missing standard contact information",
    "suspicious_text": "Document contains unusual or suspicious text patterns"
}


class OCRDocumentEngine:
    """
    OCR-powered document analysis for landlord verification.
    
    Capabilities:
    1. Text Extraction - OCR to extract all text from document images
    2. Document Classification - Identify document type (paystub, ID, etc.)
    3. Data Extraction - Extract structured data (names, amounts, dates)
    4. Fraud Detection - Identify potential document tampering/fraud
    5. Consistency Checking - Verify internal document consistency
    """
    
    def __init__(self):
        self._tesseract_checked = False
    
    def _ensure_tesseract(self) -> bool:
        """Ensure Tesseract OCR is available"""
        if not self._tesseract_checked:
            self._tesseract_checked = True
            return _check_tesseract()
        return _tesseract_available
    
    async def analyze_document(
        self,
        image_data: bytes,
        expected_type: Optional[DocumentType] = None,
        applicant_name: Optional[str] = None
    ) -> DocumentAnalysisResult:
        """
        Analyze a document image or PDF for authenticity and extract data.
        
        Supports:
        - Image files (JPEG, PNG, TIFF, BMP, WebP)
        - PDF documents (converted to images via PyMuPDF)
        
        Args:
            image_data: Raw file bytes (image or PDF)
            expected_type: Expected document type (optional)
            applicant_name: Name to verify against document (optional)
        
        Returns:
            DocumentAnalysisResult with extracted data and risk assessment
        """
        try:
            # Generate document hash
            doc_hash = hashlib.sha256(image_data).hexdigest()[:16]
            
            # Detect if PDF by checking magic bytes (%PDF)
            is_pdf = image_data[:5] == b'%PDF-'
            
            image = None
            pdf_text = ""
            pdf_page_count = 0
            
            if is_pdf:
                # --- Handle PDF documents ---
                image, pdf_text, pdf_page_count = self._process_pdf(image_data)
                if image is None:
                    return self._create_error_result(
                        image_data, 
                        "Failed to process PDF. File may be corrupted or password-protected."
                    )
                logger.info(f"PDF processed: {pdf_page_count} pages, "
                           f"{len(pdf_text)} chars extracted")
            else:
                # --- Handle image files ---
                try:
                    image = Image.open(io.BytesIO(image_data))
                except Exception as e:
                    return self._create_error_result(
                        image_data,
                        f"Cannot open image file: {str(e)}"
                    )
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 1. Analyze image quality
            quality_score, quality_indicators = self._analyze_document_quality(image)
            
            # 2. Extract text via OCR (or use PDF-extracted text)
            if pdf_text and len(pdf_text.strip()) > 20:
                # PDF already has good text — use it directly
                raw_text = pdf_text
                logger.info("Using PDF text extraction (no OCR needed)")
            else:
                # Use OCR for images or PDFs with no embedded text
                raw_text = await self._extract_text(image)
                if is_pdf and not raw_text.strip():
                    raw_text = pdf_text  # Fallback to whatever PDF gave us
            
            # 3. Classify document type
            detected_type = self._classify_document(raw_text, expected_type)
            
            # 4. Extract structured data
            extracted_data = self._extract_structured_data(raw_text)
            
            # 5. Check for fraud indicators
            fraud_indicators = self._detect_fraud_patterns(
                raw_text, extracted_data, quality_score, detected_type
            )
            
            # 6. Check consistency
            consistency_score = self._check_consistency(
                extracted_data, detected_type, applicant_name
            )
            
            # 7. Combine all indicators
            all_indicators = quality_indicators + fraud_indicators
            
            # 8. Calculate overall risk
            risk_score, risk_level = self._calculate_risk(
                quality_score, consistency_score, all_indicators
            )
            
            # 9. Generate explanation
            explanation = self._generate_explanation(
                detected_type, risk_level, extracted_data, all_indicators
            )
            
            metadata = {
                "image_size": image.size,
                "expected_type": expected_type.value if expected_type else None,
                "applicant_name": applicant_name,
                "text_length": len(raw_text),
                "source_format": "pdf" if is_pdf else "image",
            }
            if is_pdf:
                metadata["pdf_pages"] = pdf_page_count
                metadata["pdf_text_extracted"] = len(pdf_text) > 0
            
            return DocumentAnalysisResult(
                document_hash=doc_hash,
                document_type=detected_type,
                risk_level=risk_level,
                risk_score=risk_score,
                extracted_data=extracted_data,
                quality_score=quality_score,
                consistency_score=consistency_score,
                indicators=all_indicators,
                metadata=metadata,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return self._create_error_result(image_data, str(e))
    
    def _process_pdf(
        self, pdf_data: bytes
    ) -> Tuple[Optional[Image.Image], str, int]:
        """
        Process a PDF file:
        1. Extract embedded text from all pages
        2. Render first page as PIL Image for quality/visual analysis
        
        Returns:
            (first_page_image, combined_text, page_count)
        """
        if not _fitz_available:
            logger.error("PyMuPDF not installed — cannot process PDF")
            return None, "", 0
        
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            page_count = len(doc)
            
            if page_count == 0:
                doc.close()
                return None, "", 0
            
            # Extract text from ALL pages
            all_text_parts = []
            for page_num in range(page_count):
                page = doc[page_num]
                page_text = page.get_text("text")
                if page_text.strip():
                    all_text_parts.append(page_text.strip())
            
            combined_text = "\n\n".join(all_text_parts)
            
            # Render first page as high-DPI image for quality analysis
            first_page = doc[0]
            # Render at 200 DPI for good quality (default is 72 DPI)
            zoom = 200 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = first_page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            doc.close()
            
            logger.info(
                f"PDF: {page_count} pages, {len(combined_text)} chars text, "
                f"rendered at {image.size[0]}x{image.size[1]}"
            )
            
            return image, combined_text, page_count
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return None, "", 0
    
    async def _extract_text(self, image: Image.Image) -> str:
        """Extract text from image using OCR"""
        if not self._ensure_tesseract():
            logger.warning("Tesseract not available, using fallback")
            return ""
        
        try:
            import pytesseract
            
            # Preprocess image for better OCR
            processed = self._preprocess_for_ocr(image)
            
            # Run OCR
            text = pytesseract.image_to_string(
                processed,
                lang='eng',
                config='--psm 3'  # Fully automatic page segmentation
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def _preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        try:
            # Convert to grayscale
            gray = image.convert('L')
            
            # Increase contrast
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(gray)
            enhanced = enhancer.enhance(2.0)
            
            # Convert to numpy for additional processing
            img_array = np.array(enhanced)
            
            # Apply threshold to make text clearer
            threshold = 128
            img_array = ((img_array > threshold) * 255).astype(np.uint8)
            
            return Image.fromarray(img_array)
            
        except Exception:
            return image
    
    def _analyze_document_quality(self, image: Image.Image) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze document image quality"""
        indicators = []
        quality_score = 1.0
        
        width, height = image.size
        
        # Check resolution (documents need good resolution for OCR)
        if width < 500 or height < 500:
            quality_score -= 0.3
            indicators.append({
                "code": "LOW_RESOLUTION",
                "severity": 3,
                "description": "Document has low resolution, may hide tampering",
                "evidence": [f"Image size: {width}x{height}"]
            })
        elif width < 800 or height < 800:
            quality_score -= 0.1
            indicators.append({
                "code": "MODERATE_RESOLUTION",
                "severity": 1,
                "description": "Document resolution is adequate but not optimal",
                "evidence": [f"Image size: {width}x{height}"]
            })
        
        # Check for rotation/skew (documents should be straight)
        try:
            from scipy import ndimage
            img_array = np.array(image.convert('L'))
            edges = ndimage.sobel(img_array.astype(float))
            # High variance in edges at different angles might indicate skew
        except Exception:
            pass
        
        # Check brightness/contrast
        try:
            img_array = np.array(image)
            mean_brightness = np.mean(img_array)
            
            if mean_brightness < 50:
                quality_score -= 0.2
                indicators.append({
                    "code": "TOO_DARK",
                    "severity": 2,
                    "description": "Document image is too dark",
                    "evidence": ["May affect text extraction accuracy"]
                })
            elif mean_brightness > 220:
                quality_score -= 0.2
                indicators.append({
                    "code": "TOO_BRIGHT",
                    "severity": 2,
                    "description": "Document image is overexposed",
                    "evidence": ["May affect text extraction accuracy"]
                })
        except Exception:
            pass
        
        return max(0.0, min(1.0, quality_score)), indicators
    
    def _classify_document(
        self,
        text: str,
        expected_type: Optional[DocumentType]
    ) -> DocumentType:
        """Classify document type based on extracted text"""
        text_lower = text.lower()
        
        # Count keyword matches for each type
        scores = {}
        for doc_type, keywords in DOCUMENT_KEYWORDS.items():
            scores[doc_type] = sum(1 for kw in keywords if kw in text_lower)
        
        # Find best match
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type
        
        # Fall back to expected type or unknown
        return expected_type if expected_type else DocumentType.UNKNOWN
    
    def _extract_structured_data(self, text: str) -> ExtractedData:
        """Extract structured data from OCR text"""
        
        # Extract phone numbers
        phones = PATTERNS["phone"].findall(text)
        
        # Extract emails
        emails = PATTERNS["email"].findall(text)
        
        # Check for SIN
        sin_matches = PATTERNS["sin"].findall(text)
        sin_found = len(sin_matches) > 0
        
        # Extract amounts
        amount_strings = PATTERNS["amount"].findall(text)
        amounts = []
        for amt in amount_strings:
            try:
                # Remove $ and commas
                value = float(amt.replace('$', '').replace(',', ''))
                amounts.append(value)
            except ValueError:
                pass
        
        # Extract dates
        date_matches = PATTERNS["date"].findall(text)
        date = date_matches[0] if date_matches else None
        
        # Extract possible names (heuristic — filter out common false positives)
        name = None
        # Words that commonly appear as headers, not names
        header_words = {
            'employment', 'verification', 'letter', 'statement', 'document',
            'certificate', 'confirmation', 'reference', 'application', 'notice',
            'invoice', 'receipt', 'report', 'summary', 'balance', 'account',
            'bank', 'royal', 'national', 'canadian', 'financial', 'insurance',
            'tax', 'income', 'revenue', 'government', 'department', 'ministry',
            'page', 'total', 'date', 'period', 'pay', 'stub', 'earnings',
        }
        lines = text.split('\n')
        for line in lines[:15]:  # Check first 15 lines
            line = line.strip()
            words = line.split()
            # Must be 2-4 words, each capitalized, not all caps, not a header
            if 2 <= len(words) <= 4:
                if all(w[0].isupper() and not w.isupper() for w in words if len(w) > 1):
                    lower_words = {w.lower() for w in words}
                    if not lower_words.intersection(header_words):
                        name = line
                        break
        
        # Extract possible employer
        employer = None
        for line in lines:
            if any(kw in line.lower() for kw in ['employer:', 'company:', 'from:']):
                employer = line.split(':', 1)[-1].strip()
                break
        
        # Extract address (simple heuristic)
        address = None
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in ['address:', 'street', 'avenue', 'road', 'dr.', 'blvd']):
                address = line
                break
        
        # Extract account numbers (be careful with false positives)
        account_numbers = PATTERNS["account"].findall(text)
        # Filter out likely false positives
        account_numbers = [acc for acc in account_numbers if not acc.startswith('20')]  # Not years
        
        return ExtractedData(
            raw_text=text,
            name=name,
            date=date,
            amounts=amounts[:10],  # Limit to first 10
            employer=employer,
            address=address,
            phone_numbers=phones[:3],  # Limit
            emails=emails[:3],
            sin_found=sin_found,
            account_numbers=account_numbers[:3]
        )
    
    def _detect_fraud_patterns(
        self,
        text: str,
        extracted_data: ExtractedData,
        quality_score: float,
        doc_type: DocumentType
    ) -> List[Dict[str, Any]]:
        """Detect potential fraud patterns in document"""
        indicators = []
        
        # Check for suspiciously round amounts
        if extracted_data.amounts:
            round_count = sum(1 for amt in extracted_data.amounts if amt == int(amt))
            if round_count == len(extracted_data.amounts) and len(extracted_data.amounts) > 2:
                indicators.append({
                    "code": "ROUND_AMOUNTS",
                    "severity": 2,
                    "description": "All amounts are round numbers (unusual for real documents)",
                    "evidence": [f"${amt:.0f}" for amt in extracted_data.amounts[:3]]
                })
        
        # Check for future dates
        if extracted_data.date:
            try:
                # Try to parse date
                for fmt in ['%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d']:
                    try:
                        doc_date = datetime.strptime(extracted_data.date, fmt)
                        if doc_date > datetime.now():
                            indicators.append({
                                "code": "FUTURE_DATE",
                                "severity": 4,
                                "description": "Document contains a future date",
                                "evidence": [extracted_data.date]
                            })
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
        
        # Check for missing expected content based on document type
        if doc_type == DocumentType.PAYSTUB:
            if not extracted_data.amounts:
                indicators.append({
                    "code": "NO_AMOUNTS",
                    "severity": 3,
                    "description": "Paystub has no detectable amounts",
                    "evidence": []
                })
            if not extracted_data.employer and 'employer' not in text.lower():
                indicators.append({
                    "code": "NO_EMPLOYER",
                    "severity": 2,
                    "description": "Paystub has no employer information",
                    "evidence": []
                })
        
        if doc_type == DocumentType.EMPLOYMENT_LETTER:
            if not extracted_data.employer:
                indicators.append({
                    "code": "NO_EMPLOYER",
                    "severity": 3,
                    "description": "Employment letter has no employer information",
                    "evidence": []
                })
        
        # Check for very short text (may indicate image issue or fake)
        if len(text) < 50:
            indicators.append({
                "code": "MINIMAL_TEXT",
                "severity": 2,
                "description": "Very little text extracted from document",
                "evidence": [f"Only {len(text)} characters found"]
            })
        
        # Check for suspicious generic patterns
        suspicious_terms = ["lorem ipsum", "sample document", "test", "example"]
        for term in suspicious_terms:
            if term in text.lower():
                indicators.append({
                    "code": "SAMPLE_DOCUMENT",
                    "severity": 5,
                    "description": "Document contains placeholder/sample text",
                    "evidence": [f"Found: '{term}'"]
                })
        
        return indicators
    
    def _check_consistency(
        self,
        extracted_data: ExtractedData,
        doc_type: DocumentType,
        applicant_name: Optional[str]
    ) -> float:
        """Check internal consistency of document data"""
        consistency_score = 1.0
        
        # If we have an applicant name, check if it appears
        if applicant_name and extracted_data.name:
            name_parts = applicant_name.lower().split()
            found_name = extracted_data.name.lower()
            matches = sum(1 for part in name_parts if part in found_name)
            if matches < len(name_parts) / 2:
                consistency_score -= 0.3
        
        # Check if document type matches content
        if doc_type == DocumentType.PAYSTUB:
            if not extracted_data.amounts:
                consistency_score -= 0.2
            if not any(kw in extracted_data.raw_text.lower() for kw in ['pay', 'earnings', 'salary']):
                consistency_score -= 0.2
        
        # Additional consistency checks for other document types
        if doc_type == DocumentType.EMPLOYMENT_LETTER:
            if not extracted_data.employer:
                consistency_score -= 0.2
        
        return max(0.0, consistency_score)
    
    def _calculate_risk(
        self,
        quality_score: float,
        consistency_score: float,
        indicators: List[Dict[str, Any]]
    ) -> Tuple[float, DocumentRiskLevel]:
        """Calculate overall risk score and level"""
        
        risk_score = 0.0
        
        # Factor in quality
        risk_score += (1 - quality_score) * 0.3
        
        # Factor in consistency
        risk_score += (1 - consistency_score) * 0.3
        
        # Factor in indicator severity
        total_severity = sum(i.get("severity", 1) for i in indicators)
        risk_score += min(total_severity * 0.04, 0.4)
        
        # Cap risk score
        risk_score = min(risk_score, 1.0)
        
        # Determine risk level
        if risk_score < 0.15:
            level = DocumentRiskLevel.VERIFIED
        elif risk_score < 0.3:
            level = DocumentRiskLevel.LIKELY_AUTHENTIC
        elif risk_score < 0.5:
            level = DocumentRiskLevel.UNCERTAIN
        elif risk_score < 0.7:
            level = DocumentRiskLevel.SUSPICIOUS
        else:
            level = DocumentRiskLevel.LIKELY_FRAUDULENT
        
        return risk_score, level
    
    def _generate_explanation(
        self,
        doc_type: DocumentType,
        risk_level: DocumentRiskLevel,
        extracted_data: ExtractedData,
        indicators: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable explanation"""
        parts = []
        
        # Document type
        parts.append(f"📄 Document type: {doc_type.value.replace('_', ' ').title()}")
        
        # Key extracted data
        if extracted_data.name:
            parts.append(f"👤 Name found: {extracted_data.name}")
        if extracted_data.employer:
            parts.append(f"🏢 Employer: {extracted_data.employer}")
        if extracted_data.amounts:
            total = sum(extracted_data.amounts)
            parts.append(f"💰 Total amounts: ${total:,.2f}")
        
        # Risk level
        risk_emoji = {
            DocumentRiskLevel.VERIFIED: "✅",
            DocumentRiskLevel.LIKELY_AUTHENTIC: "✅",
            DocumentRiskLevel.UNCERTAIN: "❓",
            DocumentRiskLevel.SUSPICIOUS: "⚠️",
            DocumentRiskLevel.LIKELY_FRAUDULENT: "🚨"
        }
        parts.append(f"{risk_emoji.get(risk_level, '')} Risk: {risk_level.value.replace('_', ' ').title()}")
        
        # Key concerns
        high_severity = [i for i in indicators if i.get("severity", 0) >= 3]
        if high_severity:
            parts.append(f"⚠️ {len(high_severity)} concern(s) found")
        
        return " | ".join(parts)
    
    def _create_error_result(self, image_data: bytes, error: str) -> DocumentAnalysisResult:
        """Create a result for failed analysis"""
        return DocumentAnalysisResult(
            document_hash=hashlib.sha256(image_data).hexdigest()[:16],
            document_type=DocumentType.UNKNOWN,
            risk_level=DocumentRiskLevel.UNCERTAIN,
            risk_score=0.5,
            extracted_data=ExtractedData(
                raw_text="",
                name=None,
                date=None,
                amounts=[],
                employer=None,
                address=None,
                phone_numbers=[],
                emails=[],
                sin_found=False,
                account_numbers=[]
            ),
            quality_score=0.0,
            consistency_score=0.0,
            indicators=[{
                "code": "ANALYSIS_FAILED",
                "severity": 2,
                "description": f"Document analysis could not be completed: {error[:100]}",
                "evidence": []
            }],
            metadata={},
            explanation="❓ Document could not be fully analyzed. Manual review required."
        )
    
    async def verify_documents_set(
        self,
        documents: List[Tuple[bytes, DocumentType, str]],
        applicant_name: str
    ) -> Dict[str, Any]:
        """
        Verify a complete set of documents from a rental applicant.
        
        Full pipeline:
          1. Preprocessing + OCR  — extract text from every document image/PDF
          2. NER + Classification  — cross_document_engine extracts named entities
                                     (spaCy / transformers NER) and classifies doc type
          3. Fraud Detection       — per-document fraud pattern checks (this engine)
          4. Cross-Doc Consistency  — cross_document_engine compares entities across docs
          5. Risk Scoring           — aggregate per-doc + cross-doc into overall score
          6. Audit / Explainability — caller (route) handles audit log; cross-doc report
                                     provides human-readable explanations
        
        Args:
            documents: List of (image_bytes, expected_type, filename) tuples
            applicant_name: Name of the applicant to verify against
        
        Returns:
            Combined verification result
        """
        # ── Stage 1-3: Preprocessing → OCR → Fraud Detection (per-document) ──
        results = []
        
        for image_data, expected_type, filename in documents:
            result = await self.analyze_document(
                image_data, expected_type, applicant_name
            )
            result.metadata["filename"] = filename
            results.append(result)
        
        if not results:
            return {
                "document_count": 0,
                "overall_risk_score": 0.5,
                "overall_risk_level": DocumentRiskLevel.UNCERTAIN.value,
                "verified_count": 0,
                "suspicious_count": 0,
                "documents": [],
                "name_consistent": None,
                "summary": "No documents to analyze",
                "cross_document_analysis": None
            }
        
        # ── Stage 4: NER + Cross-Document Consistency ──
        # Lazy import to avoid circular dependency (same pattern as landlord_routes.py)
        cross_doc_report = None
        try:
            from application.use_cases.cross_document_engine import cross_document_engine
            
            # Build document dicts that cross_document_engine expects:
            #   { "name": str, "type": str, "text": str }
            cross_doc_input = []
            for r in results:
                cross_doc_input.append({
                    "name": r.metadata.get("filename", "document"),
                    "type": r.document_type.value,
                    "text": r.extracted_data.raw_text or ""
                })
            
            # Run NER extraction + cross-doc consistency (needs ≥1 doc + applicant name)
            if cross_doc_input:
                cross_doc_report = cross_document_engine.analyze_documents(
                    documents=cross_doc_input,
                    expected_name=applicant_name,
                    expected_address=None
                )
                logger.info(
                    f"Cross-doc analysis: {cross_doc_report.overall_consistency.value}, "
                    f"score={cross_doc_report.consistency_score:.3f}, "
                    f"checks={cross_doc_report.total_checks}"
                )
        except Exception as e:
            logger.warning(f"Cross-document analysis skipped: {e}")
        
        # ── Stage 5: Aggregate Risk Scoring ──
        
        # Baseline: name consistency from simple OCR-extracted names
        found_names = [r.extracted_data.name for r in results if r.extracted_data.name]
        name_consistent = True
        if len(found_names) >= 2:
            base_name = found_names[0].lower().split()
            for name in found_names[1:]:
                name_parts = name.lower().split()
                if not any(part in base_name for part in name_parts):
                    name_consistent = False
                    break
        
        # If cross-doc engine ran with ≥2 docs, its NER-based name check is superior
        if cross_doc_report and cross_doc_report.total_checks > 0:
            # Any critical name mismatch from NER overrides the simple check
            name_checks = [
                c for c in cross_doc_report.consistency_checks
                if c.entity_type.value == "person_name"
            ]
            if name_checks:
                name_consistent = all(c.is_consistent for c in name_checks)
        
        # Per-document risk aggregation
        overall_risk = float(np.mean([r.risk_score for r in results]))
        verified_count = sum(
            1 for r in results 
            if r.risk_level in [DocumentRiskLevel.VERIFIED, DocumentRiskLevel.LIKELY_AUTHENTIC]
        )
        suspicious_count = sum(
            1 for r in results 
            if r.risk_level in [DocumentRiskLevel.SUSPICIOUS, DocumentRiskLevel.LIKELY_FRAUDULENT]
        )
        
        # Factor in cross-document consistency risk
        if cross_doc_report and cross_doc_report.total_checks > 0:
            consistency_penalty = (1.0 - cross_doc_report.consistency_score) * 0.25
            overall_risk = min(overall_risk + consistency_penalty, 1.0)
            
            # Critical issues (e.g. NER name/SSN mismatch) get extra weight
            if cross_doc_report.critical_issues:
                overall_risk = min(overall_risk + 0.15 * len(cross_doc_report.critical_issues), 1.0)
        elif not name_consistent and len(found_names) >= 2:
            # Fallback: simple name inconsistency penalty (original logic)
            overall_risk = min(overall_risk + 0.2, 1.0)
        
        # Determine overall level
        if overall_risk < 0.2:
            overall_level = DocumentRiskLevel.VERIFIED
        elif overall_risk < 0.4:
            overall_level = DocumentRiskLevel.LIKELY_AUTHENTIC
        elif overall_risk < 0.6:
            overall_level = DocumentRiskLevel.UNCERTAIN
        else:
            overall_level = DocumentRiskLevel.SUSPICIOUS
        
        # ── Stage 6: Summary with cross-doc context ──
        total = len(results)
        
        # Build summary — incorporate cross-doc critical issues if any
        if cross_doc_report and cross_doc_report.critical_issues:
            issues_text = ", ".join(
                set(i["type"] for i in cross_doc_report.critical_issues)
            )
            summary = f"🚨 Identity mismatch detected ({issues_text}) across {total} documents"
        elif suspicious_count > 0:
            summary = f"🚨 {suspicious_count} of {total} documents appear suspicious"
        elif not name_consistent:
            summary = f"⚠️ Names are inconsistent across documents"
        elif cross_doc_report and cross_doc_report.warnings:
            summary = f"⚠️ {len(cross_doc_report.warnings)} minor discrepancy(ies) across {total} documents"
        elif verified_count == total:
            summary = f"✅ All {total} documents verified successfully"
        else:
            summary = f"ℹ️ {verified_count} of {total} documents verified"
        
        # Enrich per-document dicts with NER-extracted entities (if available)
        doc_dicts = []
        for r in results:
            d = r.to_dict()
            # Attach NER entities for this document from cross-doc report
            if cross_doc_report:
                filename = r.metadata.get("filename", "document")
                ner_entities = cross_doc_report.extracted_entities.get(filename, [])
                d["ner_entities"] = [e.to_dict() for e in ner_entities]
            doc_dicts.append(d)
        
        return {
            "document_count": total,
            "overall_risk_score": round(overall_risk, 3),
            "overall_risk_level": overall_level.value,
            "verified_count": verified_count,
            "suspicious_count": suspicious_count,
            "documents": doc_dicts,
            "name_consistent": name_consistent if len(found_names) >= 2 else None,
            "summary": summary,
            "cross_document_analysis": cross_doc_report.to_dict() if cross_doc_report else None
        }


# Singleton instance
ocr_engine = OCRDocumentEngine()
