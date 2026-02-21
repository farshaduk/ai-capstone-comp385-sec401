"""
Cross-Document Consistency Engine - AI-Powered Document Verification

This module implements cross-document consistency checking for landlord verification:
1. Compare information across multiple documents (ID, lease, pay stubs)
2. Detect inconsistencies in names, dates, addresses
3. Use fuzzy matching for OCR errors
4. AI-based entity extraction and comparison

For capstone: COMP385 AI Project
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

# Try to import NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ConsistencyLevel(str, Enum):
    """Consistency levels between documents"""
    CONSISTENT = "consistent"
    MINOR_DISCREPANCY = "minor_discrepancy"
    MAJOR_DISCREPANCY = "major_discrepancy"
    CRITICAL_MISMATCH = "critical_mismatch"
    INSUFFICIENT_DATA = "insufficient_data"


class EntityType(str, Enum):
    """Types of entities we extract and compare"""
    PERSON_NAME = "person_name"
    ADDRESS = "address"
    DATE = "date"
    PHONE = "phone"
    EMAIL = "email"
    SSN = "ssn"
    AMOUNT = "amount"
    EMPLOYER = "employer"
    ID_NUMBER = "id_number"


@dataclass
class ExtractedEntity:
    """An entity extracted from a document"""
    entity_type: EntityType
    value: str
    normalized_value: str
    confidence: float
    source_document: str
    position: Optional[Tuple[int, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.entity_type.value,
            "value": self.value,
            "normalized": self.normalized_value,
            "confidence": float(round(self.confidence, 3)),
            "source": self.source_document
        }


@dataclass
class ConsistencyCheck:
    """Result of comparing two entities"""
    entity_type: EntityType
    doc1_value: str
    doc2_value: str
    doc1_source: str
    doc2_source: str
    similarity: float
    is_consistent: bool
    discrepancy_type: str
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type.value,
            "documents": {
                "doc1": {"source": self.doc1_source, "value": self.doc1_value},
                "doc2": {"source": self.doc2_source, "value": self.doc2_value}
            },
            "similarity": round(self.similarity, 3),
            "is_consistent": self.is_consistent,
            "discrepancy_type": self.discrepancy_type,
            "explanation": self.explanation
        }


@dataclass
class ConsistencyReport:
    """Complete cross-document consistency report"""
    report_id: str
    documents_analyzed: List[str]
    total_checks: int
    
    overall_consistency: ConsistencyLevel
    consistency_score: float  # 0-1
    confidence: float
    
    extracted_entities: Dict[str, List[ExtractedEntity]]  # by document
    consistency_checks: List[ConsistencyCheck]
    
    critical_issues: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    
    summary: str
    recommendation: str
    
    verified_fields: List[str]
    unverified_fields: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "summary": self.summary,
            "documents_analyzed": self.documents_analyzed,
            "overall": {
                "consistency": self.overall_consistency.value,
                "score": round(self.consistency_score, 3),
                "confidence": round(self.confidence, 3)
            },
            "checks": {
                "total": self.total_checks,
                "passed": len([c for c in self.consistency_checks if c.is_consistent]),
                "failed": len([c for c in self.consistency_checks if not c.is_consistent])
            },
            "entities_by_document": {
                doc: [e.to_dict() for e in entities]
                for doc, entities in self.extracted_entities.items()
            },
            "consistency_checks": [c.to_dict() for c in self.consistency_checks],
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
            "verified_fields": self.verified_fields,
            "unverified_fields": self.unverified_fields
        }


class CrossDocumentEngine:
    """
    AI-powered cross-document consistency verification.
    
    This engine:
    1. Extracts entities from multiple documents using NLP
    2. Normalizes extracted data for comparison
    3. Applies fuzzy matching to account for OCR errors
    4. Detects inconsistencies that indicate fraud
    """
    
    # Thresholds for consistency
    EXACT_MATCH_THRESHOLD = 0.98
    HIGH_SIMILARITY_THRESHOLD = 0.85
    ACCEPTABLE_SIMILARITY_THRESHOLD = 0.70
    
    # Name normalization patterns
    NAME_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'phd', 'md', 'esq'}
    NAME_PREFIXES = {'mr', 'mrs', 'ms', 'miss', 'dr', 'prof'}
    
    def __init__(self):
        self.nlp = None
        self.ner_pipeline = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of NLP models"""
        if self._initialized:
            return
        
        # Try spaCy first
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self._initialized = True
                logger.info("Cross-document engine initialized with spaCy")
                return
            except OSError:
                logger.warning("spaCy model not downloaded")
        
        # Try transformers NER
        if TRANSFORMERS_AVAILABLE:
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple"
                )
                self._initialized = True
                logger.info("Cross-document engine initialized with transformers NER")
            except Exception as e:
                logger.warning(f"Failed to initialize NER pipeline: {e}")
    
    def analyze_documents(
        self,
        documents: List[Dict[str, Any]],
        expected_name: Optional[str] = None,
        expected_address: Optional[str] = None
    ) -> ConsistencyReport:
        """
        Analyze multiple documents for consistency.
        
        Args:
            documents: List of document dicts with 'name', 'type', 'text' (OCR output)
            expected_name: Expected tenant/landlord name for comparison
            expected_address: Expected property address for comparison
            
        Returns:
            ConsistencyReport with full analysis
        """
        self._ensure_initialized()
        
        report_id = f"cross_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract entities from each document
        extracted_entities: Dict[str, List[ExtractedEntity]] = {}
        
        for doc in documents:
            doc_name = doc.get('name', f"document_{len(extracted_entities)}")
            doc_type = doc.get('type', 'unknown')
            text = doc.get('text', '')
            
            entities = self._extract_entities(text, doc_name, doc_type)
            extracted_entities[doc_name] = entities
        
        # Add expected values as reference document
        if expected_name or expected_address:
            reference_entities = []
            if expected_name:
                reference_entities.append(ExtractedEntity(
                    entity_type=EntityType.PERSON_NAME,
                    value=expected_name,
                    normalized_value=self._normalize_name(expected_name),
                    confidence=1.0,
                    source_document="application_form"
                ))
            if expected_address:
                reference_entities.append(ExtractedEntity(
                    entity_type=EntityType.ADDRESS,
                    value=expected_address,
                    normalized_value=self._normalize_address(expected_address),
                    confidence=1.0,
                    source_document="application_form"
                ))
            if reference_entities:
                extracted_entities["application_form"] = reference_entities
        
        # Perform consistency checks
        consistency_checks = self._perform_consistency_checks(extracted_entities)
        
        # Analyze results
        critical_issues = []
        warnings = []
        
        for check in consistency_checks:
            if not check.is_consistent:
                if check.entity_type in [EntityType.PERSON_NAME, EntityType.SSN, EntityType.ID_NUMBER]:
                    critical_issues.append({
                        "type": check.entity_type.value,
                        "description": check.explanation,
                        "severity": "critical",
                        "documents": [check.doc1_source, check.doc2_source]
                    })
                elif check.similarity < self.ACCEPTABLE_SIMILARITY_THRESHOLD:
                    warnings.append({
                        "type": check.entity_type.value,
                        "description": check.explanation,
                        "severity": "warning",
                        "documents": [check.doc1_source, check.doc2_source]
                    })
        
        # Calculate overall consistency
        if not consistency_checks:
            overall_level = ConsistencyLevel.INSUFFICIENT_DATA
            consistency_score = 0.5
        else:
            avg_similarity = sum(c.similarity for c in consistency_checks) / len(consistency_checks)
            passed_checks = sum(1 for c in consistency_checks if c.is_consistent)
            pass_rate = passed_checks / len(consistency_checks)
            
            consistency_score = (avg_similarity + pass_rate) / 2
            
            if critical_issues:
                overall_level = ConsistencyLevel.CRITICAL_MISMATCH
            elif consistency_score >= 0.9:
                overall_level = ConsistencyLevel.CONSISTENT
            elif consistency_score >= 0.7:
                overall_level = ConsistencyLevel.MINOR_DISCREPANCY
            else:
                overall_level = ConsistencyLevel.MAJOR_DISCREPANCY
        
        # Determine verified vs unverified fields
        verified_fields = list(set(
            c.entity_type.value for c in consistency_checks if c.is_consistent
        ))
        all_entity_types = set(e.entity_type.value for entities in extracted_entities.values() for e in entities)
        unverified_fields = list(all_entity_types - set(verified_fields))
        
        # Generate summary and recommendation
        summary, recommendation = self._generate_summary(
            overall_level, critical_issues, warnings, consistency_checks
        )
        
        return ConsistencyReport(
            report_id=report_id,
            documents_analyzed=list(extracted_entities.keys()),
            total_checks=len(consistency_checks),
            overall_consistency=overall_level,
            consistency_score=consistency_score,
            confidence=0.85 if self._initialized else 0.65,
            extracted_entities=extracted_entities,
            consistency_checks=consistency_checks,
            critical_issues=critical_issues,
            warnings=warnings,
            summary=summary,
            recommendation=recommendation,
            verified_fields=verified_fields,
            unverified_fields=unverified_fields
        )
    
    def _extract_entities(
        self,
        text: str,
        doc_name: str,
        doc_type: str
    ) -> List[ExtractedEntity]:
        """Extract entities from document text using NLP"""
        entities = []
        
        # Pattern-based extraction (works without NLP models)
        entities.extend(self._extract_patterns(text, doc_name))
        
        # NLP-based extraction if available
        if self.nlp:
            entities.extend(self._extract_with_spacy(text, doc_name))
        elif self.ner_pipeline:
            entities.extend(self._extract_with_transformers(text, doc_name))
        
        # Document-type specific extraction
        if 'id' in doc_type.lower() or 'license' in doc_type.lower():
            entities.extend(self._extract_id_specific(text, doc_name))
        elif 'pay' in doc_type.lower() or 'stub' in doc_type.lower():
            entities.extend(self._extract_paystub_specific(text, doc_name))
        elif 'lease' in doc_type.lower() or 'contract' in doc_type.lower():
            entities.extend(self._extract_lease_specific(text, doc_name))
        
        # Deduplicate
        return self._deduplicate_entities(entities)
    
    def _extract_patterns(
        self,
        text: str,
        doc_name: str
    ) -> List[ExtractedEntity]:
        """Extract entities using regex patterns"""
        entities = []
        
        # Phone numbers
        phone_pattern = r'\b(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b'
        for match in re.finditer(phone_pattern, text):
            normalized = re.sub(r'[^\d]', '', match.group(1))
            if len(normalized) >= 10:
                entities.append(ExtractedEntity(
                    entity_type=EntityType.PHONE,
                    value=match.group(1),
                    normalized_value=normalized[-10:],
                    confidence=0.9,
                    source_document=doc_name,
                    position=match.span()
                ))
        
        # Email addresses
        email_pattern = r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
        for match in re.finditer(email_pattern, text):
            entities.append(ExtractedEntity(
                entity_type=EntityType.EMAIL,
                value=match.group(1),
                normalized_value=match.group(1).lower(),
                confidence=0.95,
                source_document=doc_name,
                position=match.span()
            ))
        
        # Dates
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b',
        ]
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                normalized = self._normalize_date(match.group(1))
                if normalized:
                    entities.append(ExtractedEntity(
                        entity_type=EntityType.DATE,
                        value=match.group(1),
                        normalized_value=normalized,
                        confidence=0.85,
                        source_document=doc_name,
                        position=match.span()
                    ))
        
        # Money amounts
        amount_pattern = r'\$\s*([\d,]+\.?\d*)\b'
        for match in re.finditer(amount_pattern, text):
            value = match.group(1).replace(',', '')
            try:
                float(value)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.AMOUNT,
                    value=f"${match.group(1)}",
                    normalized_value=value,
                    confidence=0.9,
                    source_document=doc_name,
                    position=match.span()
                ))
            except ValueError:
                pass
        
        # SSN (masked or full)
        ssn_patterns = [
            r'\b(\d{3}[-\s]?\d{2}[-\s]?\d{4})\b',
            r'\b(XXX[-\s]?XX[-\s]?\d{4})\b',
            r'\b(\*{3}[-\s]?\*{2}[-\s]?\d{4})\b',
        ]
        for pattern in ssn_patterns:
            for match in re.finditer(pattern, text):
                normalized = re.sub(r'[^\dX*]', '', match.group(1))
                entities.append(ExtractedEntity(
                    entity_type=EntityType.SSN,
                    value=match.group(1),
                    normalized_value=normalized[-4:],  # Last 4 for comparison
                    confidence=0.7,
                    source_document=doc_name,
                    position=match.span()
                ))
        
        return entities
    
    def _extract_with_spacy(
        self,
        text: str,
        doc_name: str
    ) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER"""
        entities = []
        
        if not self.nlp:
            return entities
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities.append(ExtractedEntity(
                    entity_type=EntityType.PERSON_NAME,
                    value=ent.text,
                    normalized_value=self._normalize_name(ent.text),
                    confidence=0.85,
                    source_document=doc_name
                ))
            elif ent.label_ == "ORG":
                entities.append(ExtractedEntity(
                    entity_type=EntityType.EMPLOYER,
                    value=ent.text,
                    normalized_value=ent.text.lower().strip(),
                    confidence=0.80,
                    source_document=doc_name
                ))
            elif ent.label_ in ["GPE", "LOC"]:
                # Could be part of an address
                pass
        
        return entities
    
    def _extract_with_transformers(
        self,
        text: str,
        doc_name: str
    ) -> List[ExtractedEntity]:
        """Extract entities using transformers NER"""
        entities = []
        
        if not self.ner_pipeline:
            return entities
        
        try:
            results = self.ner_pipeline(text[:512])  # Limit text length
            
            for ent in results:
                if ent['entity_group'] == 'PER':
                    entities.append(ExtractedEntity(
                        entity_type=EntityType.PERSON_NAME,
                        value=ent['word'],
                        normalized_value=self._normalize_name(ent['word']),
                        confidence=ent['score'],
                        source_document=doc_name
                    ))
                elif ent['entity_group'] == 'ORG':
                    entities.append(ExtractedEntity(
                        entity_type=EntityType.EMPLOYER,
                        value=ent['word'],
                        normalized_value=ent['word'].lower().strip(),
                        confidence=ent['score'],
                        source_document=doc_name
                    ))
        except Exception as e:
            logger.warning(f"Transformers NER failed: {e}")
        
        return entities
    
    def _extract_id_specific(
        self,
        text: str,
        doc_name: str
    ) -> List[ExtractedEntity]:
        """Extract ID document specific entities"""
        entities = []
        
        # Driver's license / ID number patterns
        id_patterns = [
            r'\b([A-Z]\d{7})\b',  # Common format
            r'\b(\d{8,9})\b',  # Numeric only
            r'\b([A-Z]{2}\d{6})\b',  # State prefix
        ]
        
        for pattern in id_patterns:
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    entity_type=EntityType.ID_NUMBER,
                    value=match.group(1),
                    normalized_value=match.group(1).upper(),
                    confidence=0.75,
                    source_document=doc_name
                ))
        
        # Look for name after common labels
        name_labels = r'(?:NAME|FULL\s*NAME|APPLICANT)\s*[:]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        for match in re.finditer(name_labels, text, re.IGNORECASE):
            entities.append(ExtractedEntity(
                entity_type=EntityType.PERSON_NAME,
                value=match.group(1),
                normalized_value=self._normalize_name(match.group(1)),
                confidence=0.9,
                source_document=doc_name
            ))
        
        return entities
    
    def _extract_paystub_specific(
        self,
        text: str,
        doc_name: str
    ) -> List[ExtractedEntity]:
        """Extract pay stub specific entities"""
        entities = []
        
        # Employee name patterns
        emp_patterns = [
            r'(?:EMPLOYEE|EMP(?:LOYEE)?|PAYEE)\s*(?:NAME)?[:]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]
        for pattern in emp_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    entity_type=EntityType.PERSON_NAME,
                    value=match.group(1),
                    normalized_value=self._normalize_name(match.group(1)),
                    confidence=0.9,
                    source_document=doc_name
                ))
        
        # Employer patterns
        employer_patterns = [
            r'(?:EMPLOYER|COMPANY)\s*(?:NAME)?[:]\s*([A-Z][\w\s&,.]+?)(?:\n|$)',
        ]
        for pattern in employer_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    entity_type=EntityType.EMPLOYER,
                    value=match.group(1).strip(),
                    normalized_value=match.group(1).lower().strip(),
                    confidence=0.85,
                    source_document=doc_name
                ))
        
        return entities
    
    def _extract_lease_specific(
        self,
        text: str,
        doc_name: str
    ) -> List[ExtractedEntity]:
        """Extract lease/contract specific entities"""
        entities = []
        
        # Tenant name patterns
        tenant_patterns = [
            r'(?:TENANT|LESSEE|RENTER)\s*(?:NAME)?[:]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'(?:between|with)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        ]
        for pattern in tenant_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    entity_type=EntityType.PERSON_NAME,
                    value=match.group(1),
                    normalized_value=self._normalize_name(match.group(1)),
                    confidence=0.85,
                    source_document=doc_name
                ))
        
        # Property address patterns
        address_pattern = r'(?:PROPERTY|PREMISES|ADDRESS)\s*[:]\s*(.+?)(?:\n|$)'
        for match in re.finditer(address_pattern, text, re.IGNORECASE):
            entities.append(ExtractedEntity(
                entity_type=EntityType.ADDRESS,
                value=match.group(1).strip(),
                normalized_value=self._normalize_address(match.group(1)),
                confidence=0.85,
                source_document=doc_name
            ))
        
        return entities
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a person's name for comparison"""
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Lowercase
        name = name.lower()
        
        # Remove common prefixes/suffixes
        words = name.split()
        words = [w for w in words if w.rstrip('.') not in self.NAME_PREFIXES | self.NAME_SUFFIXES]
        
        return ' '.join(words)
    
    def _normalize_address(self, address: str) -> str:
        """Normalize an address for comparison"""
        address = address.lower()
        
        # Standardize common abbreviations
        replacements = {
            r'\bstreet\b': 'st',
            r'\bave(?:nue)?\b': 'ave',
            r'\broad\b': 'rd',
            r'\bdrive\b': 'dr',
            r'\bboulevard\b': 'blvd',
            r'\bapartment\b': 'apt',
            r'\bunit\b': 'unit',
            r'\b#\b': 'apt',
            r'\bnorth\b': 'n',
            r'\bsouth\b': 's',
            r'\beast\b': 'e',
            r'\bwest\b': 'w',
        }
        
        for pattern, replacement in replacements.items():
            address = re.sub(pattern, replacement, address)
        
        # Remove punctuation except apostrophes
        address = re.sub(r"[^\w\s']", '', address)
        
        # Remove extra whitespace
        address = ' '.join(address.split())
        
        return address
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date to YYYY-MM-DD format"""
        formats = [
            '%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%Y/%m/%d',
            '%m/%d/%y', '%m-%d-%y', '%d/%m/%Y', '%d-%m-%Y',
            '%B %d, %Y', '%b %d, %Y', '%B %d %Y', '%b %d %Y',
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def _deduplicate_entities(
        self,
        entities: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """Remove duplicate entities, keeping highest confidence"""
        unique = {}
        
        for ent in entities:
            key = (ent.entity_type, ent.normalized_value)
            if key not in unique or ent.confidence > unique[key].confidence:
                unique[key] = ent
        
        return list(unique.values())
    
    def _perform_consistency_checks(
        self,
        extracted_entities: Dict[str, List[ExtractedEntity]]
    ) -> List[ConsistencyCheck]:
        """Compare entities across documents"""
        checks = []
        
        # Group entities by type
        entities_by_type: Dict[EntityType, List[ExtractedEntity]] = {}
        for doc_name, entities in extracted_entities.items():
            for ent in entities:
                if ent.entity_type not in entities_by_type:
                    entities_by_type[ent.entity_type] = []
                entities_by_type[ent.entity_type].append(ent)
        
        # Compare entities of the same type from different documents
        for entity_type, entities in entities_by_type.items():
            # Group by source document
            by_document = {}
            for ent in entities:
                if ent.source_document not in by_document:
                    by_document[ent.source_document] = []
                by_document[ent.source_document].append(ent)
            
            # Compare across documents
            doc_names = list(by_document.keys())
            for i in range(len(doc_names)):
                for j in range(i + 1, len(doc_names)):
                    doc1, doc2 = doc_names[i], doc_names[j]
                    
                    # Compare best match from each document
                    for ent1 in by_document[doc1]:
                        for ent2 in by_document[doc2]:
                            check = self._compare_entities(ent1, ent2)
                            if check:
                                checks.append(check)
        
        return checks
    
    def _compare_entities(
        self,
        ent1: ExtractedEntity,
        ent2: ExtractedEntity
    ) -> Optional[ConsistencyCheck]:
        """Compare two entities of the same type"""
        if ent1.entity_type != ent2.entity_type:
            return None
        
        # Calculate similarity
        similarity = SequenceMatcher(
            None, ent1.normalized_value, ent2.normalized_value
        ).ratio()
        
        # Determine consistency based on entity type
        if ent1.entity_type == EntityType.PERSON_NAME:
            is_consistent = similarity >= self.ACCEPTABLE_SIMILARITY_THRESHOLD
            if similarity >= self.EXACT_MATCH_THRESHOLD:
                discrepancy_type = "none"
                explanation = "Names match exactly across documents"
            elif similarity >= self.HIGH_SIMILARITY_THRESHOLD:
                discrepancy_type = "minor"
                explanation = f"Names are highly similar (possible OCR error or nickname)"
            elif similarity >= self.ACCEPTABLE_SIMILARITY_THRESHOLD:
                discrepancy_type = "moderate"
                explanation = f"Names differ slightly - verify manually"
            else:
                discrepancy_type = "critical"
                explanation = f"Names do not match: '{ent1.value}' vs '{ent2.value}' - POSSIBLE FRAUD"
        
        elif ent1.entity_type == EntityType.SSN:
            # Last 4 digits must match exactly
            is_consistent = ent1.normalized_value == ent2.normalized_value
            if is_consistent:
                discrepancy_type = "none"
                explanation = "SSN last 4 digits match"
            else:
                discrepancy_type = "critical"
                explanation = f"SSN MISMATCH detected - different identities"
        
        elif ent1.entity_type == EntityType.ADDRESS:
            is_consistent = similarity >= self.ACCEPTABLE_SIMILARITY_THRESHOLD
            if similarity >= self.HIGH_SIMILARITY_THRESHOLD:
                discrepancy_type = "none" if similarity >= self.EXACT_MATCH_THRESHOLD else "minor"
                explanation = "Addresses match" if similarity >= self.EXACT_MATCH_THRESHOLD else "Addresses are similar (formatting differences)"
            else:
                discrepancy_type = "major"
                explanation = f"Address discrepancy detected between documents"
        
        else:
            # Generic comparison
            is_consistent = similarity >= self.ACCEPTABLE_SIMILARITY_THRESHOLD
            discrepancy_type = "none" if is_consistent else "minor"
            explanation = f"Field values {'match' if is_consistent else 'differ'}"
        
        return ConsistencyCheck(
            entity_type=ent1.entity_type,
            doc1_value=ent1.value,
            doc2_value=ent2.value,
            doc1_source=ent1.source_document,
            doc2_source=ent2.source_document,
            similarity=similarity,
            is_consistent=is_consistent,
            discrepancy_type=discrepancy_type,
            explanation=explanation
        )
    
    def _generate_summary(
        self,
        overall_level: ConsistencyLevel,
        critical_issues: List[Dict],
        warnings: List[Dict],
        checks: List[ConsistencyCheck]
    ) -> Tuple[str, str]:
        """Generate summary and recommendation"""
        
        if overall_level == ConsistencyLevel.CONSISTENT:
            summary = "✅ All documents are consistent. Identity verification passed."
            recommendation = "Documents appear authentic and consistent. Proceed with standard due diligence."
        
        elif overall_level == ConsistencyLevel.MINOR_DISCREPANCY:
            summary = f"⚠️ Minor discrepancies found. {len(warnings)} warning(s) detected."
            recommendation = "Review flagged discrepancies manually. May be OCR errors or formatting differences."
        
        elif overall_level == ConsistencyLevel.MAJOR_DISCREPANCY:
            summary = f"⚠️ Significant inconsistencies detected across documents."
            recommendation = "Request original documents for manual verification. Consider additional identity checks."
        
        elif overall_level == ConsistencyLevel.CRITICAL_MISMATCH:
            issues_text = ', '.join(set(i['type'] for i in critical_issues))
            summary = f"🚨 CRITICAL: Identity mismatch detected ({issues_text})"
            recommendation = "DO NOT proceed with rental. Documents may be fraudulent or belong to different individuals. Verify identity in person with original documents."
        
        else:
            summary = "⚠️ Insufficient data for comprehensive verification."
            recommendation = "Request additional documents for complete identity verification."
        
        return summary, recommendation


# Singleton instance
cross_document_engine = CrossDocumentEngine()


def verify_documents(
    documents: List[Dict[str, Any]],
    expected_name: Optional[str] = None,
    expected_address: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to verify document consistency"""
    result = cross_document_engine.analyze_documents(
        documents, expected_name, expected_address
    )
    return result.to_dict()
