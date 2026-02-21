"""
Address Validation Engine - Geocoding-based address verification

This module validates rental listing addresses by:
1. Checking if the address resolves to a valid location
2. Verifying the address is in a residential area
3. Detecting fake/non-existent addresses (common in scams)
4. Extracting location data for price comparison
5. Cross-referencing against known scam address patterns
6. Computing a multi-signal risk score contribution

Uses free geocoding services (Nominatim/OpenStreetMap) for validation.
"""

import re
import asyncio
import aiohttp
import time
from collections import OrderedDict
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import urllib.parse

logger = logging.getLogger(__name__)

# ─── Known Scam Address Patterns ──────────────────────────────────────
# PO boxes, mail-drop services, vacant lots, and commercial-only patterns
# that scammers commonly use for fake rental listings.
KNOWN_SCAM_ADDRESS_PATTERNS = [
    r"\bp\.?\s*o\.?\s*box\b",                       # PO Box variants
    r"\bpost\s*office\s*box\b",                      # Post Office Box
    r"\bgeneral\s+delivery\b",                       # General Delivery
    r"\bc/?o\s+\w+",                                 # c/o (care of) — mail forwarding
    r"\bmail\s*box(es)?\b",                          # Mailbox / Mailboxes
    r"\bups\s+store\b",                              # UPS Store (mail drop)
    r"\bfedex\s+office\b",                           # FedEx Office (mail drop)
    r"\bvirtual\s+office\b",                         # Virtual office addresses
    r"\bsuite\s+pmb\b",                              # Private Mailbox suites
    r"\bpmb\s*#?\s*\d+\b",                           # PMB (Private MailBox)
]

# Non-residential place types — Nominatim class/type combos that indicate
# the resolved address is NOT a place anyone could rent.
NON_RESIDENTIAL_TYPES = [
    "industrial", "commercial", "warehouse", "office",
    "retail", "shop", "mall", "factory", "plant",
    "parking", "garage", "car_wash", "fuel",
    "cemetery", "grave_yard", "landfill", "dump",
    "construction", "brownfield",
]

# Expanded residential place types for Nominatim classification
RESIDENTIAL_TYPES = [
    "house", "residential", "apartments", "apartment", "building",
    "flat", "condominium", "condo", "townhouse", "detached",
    "semi", "semidetached", "semi-detached", "dormitory", "dorm",
    "duplex", "triplex", "bungalow", "cottage", "loft",
    "terrace", "terraced", "maisonette", "studio",
    "housing", "dwelling", "home",
]


class AddressValidationStatus(str, Enum):
    """Status of address validation"""
    VALID = "valid"
    PARTIALLY_VALID = "partially_valid"
    INVALID = "invalid"
    UNVERIFIABLE = "unverifiable"
    SUSPICIOUS = "suspicious"


@dataclass 
class AddressValidationResult:
    """Result of address validation"""
    original_address: str
    normalized_address: Optional[str]
    status: AddressValidationStatus
    confidence: float  # 0-1
    latitude: Optional[float]
    longitude: Optional[float]
    city: Optional[str]
    province: Optional[str]
    postal_code: Optional[str]
    country: Optional[str]
    is_residential: bool
    risk_score: float  # 0-1
    indicators: List[Dict[str, Any]]
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_address": self.original_address,
            "normalized_address": self.normalized_address,
            "status": self.status.value,
            "confidence": round(self.confidence, 2),
            "coordinates": {
                "latitude": self.latitude,
                "longitude": self.longitude
            } if self.latitude is not None and self.longitude is not None else None,
            "location": {
                "city": self.city,
                "province": self.province,
                "postal_code": self.postal_code,
                "country": self.country
            },
            "is_residential": self.is_residential,
            "risk_score": round(self.risk_score, 3),
            "indicators": self.indicators,
            "explanation": self.explanation
        }


class AddressValidationEngine:
    """
    AI-powered address validation for rental listings.
    
    Uses geocoding APIs to verify addresses and detect potential fraud:
    - Validates address exists and resolves to real coordinates
    - Checks if location is in a residential area
    - Detects common address manipulation patterns
    - Cross-references known scam address patterns
    - Caches geocoding results to reduce API pressure
    """
    
    # LRU geocode cache — avoids re-querying Nominatim for the same address
    _GEOCODE_CACHE_MAX_SIZE = 512
    
    # Retry settings for Nominatim resilience
    _MAX_RETRIES = 2
    _RETRY_BACKOFF = 2.0  # seconds between retries
    
    # Default country for geocoding (configurable)
    _DEFAULT_COUNTRY = "ca"
    _DEFAULT_COUNTRY_NAME = "Canada"
    
    def __init__(self):
        # Nominatim (OpenStreetMap) geocoding endpoint
        self.geocoding_url = "https://nominatim.openstreetmap.org/search"
        self.reverse_url = "https://nominatim.openstreetmap.org/reverse"
        self.user_agent = "FraudDetectionSystem/1.0 (Educational Capstone Project)"
        
        # Rate limiting (Nominatim requires 1 request per second)
        self._last_request_time = 0
        self._min_request_interval = 1.1  # seconds
        
        # Geocode result cache keyed by normalized address string
        self._geocode_cache: OrderedDict[str, Optional[Dict[str, Any]]] = OrderedDict()
    
    async def validate(
        self,
        address: str,
        listing_text: Optional[str] = None
    ) -> AddressValidationResult:
        """
        Validate an address from a rental listing.
        
        Args:
            address: The address to validate
            listing_text: Full listing text (to extract additional context)
        
        Returns:
            AddressValidationResult with validation details
        """
        # Clean and normalize address
        cleaned_address = self._clean_address(address)
        
        if not cleaned_address:
            return self._create_invalid_result(
                address, "Address is empty or contains only invalid characters"
            )
        
        # Check for suspicious patterns first
        suspicion_score, suspicion_indicators = self._check_suspicious_patterns(
            address, listing_text
        )
        
        try:
            # Geocode the address
            geocode_result = await self._geocode_address(cleaned_address)
            
            if geocode_result:
                return self._process_geocode_result(
                    address, cleaned_address, geocode_result,
                    suspicion_score, suspicion_indicators
                )
            else:
                # Try with less specific address
                simplified = self._simplify_address(cleaned_address)
                if simplified != cleaned_address:
                    geocode_result = await self._geocode_address(simplified)
                    if geocode_result:
                        return self._process_geocode_result(
                            address, simplified, geocode_result,
                            suspicion_score + 0.2, suspicion_indicators,
                            partial=True
                        )
                
                return self._create_invalid_result(
                    address, 
                    "Address could not be verified - no matching location found",
                    suspicion_indicators
                )
                
        except Exception as e:
            logger.error(f"Address validation error: {e}")
            return AddressValidationResult(
                original_address=address,
                normalized_address=None,
                status=AddressValidationStatus.UNVERIFIABLE,
                confidence=0.0,
                latitude=None,
                longitude=None,
                city=None,
                province=None,
                postal_code=None,
                country=None,
                is_residential=False,
                risk_score=0.3,
                indicators=[{
                    "code": "ADDRESS_VALIDATION_FAILED",
                    "severity": 2,
                    "description": "Could not verify address due to service unavailability",
                    "evidence": ["Geocoding service error"]
                }],
                explanation="Unable to verify address at this time. Please verify manually."
            )
    
    def _clean_address(self, address: str) -> str:
        """Clean and normalize address string"""
        if not address:
            return ""
        
        # Remove extra whitespace
        cleaned = " ".join(address.split())
        
        # Remove common non-address characters at start/end
        cleaned = cleaned.strip(".,;:-")
        
        # Normalize common abbreviations
        replacements = {
            r"\bst\b": "street",
            r"\bave?\b": "avenue",
            r"\bblvd\b": "boulevard",
            r"\bdr\b": "drive",
            r"\brd\b": "road",
            r"\bln\b": "lane",
            r"\bct\b": "court",
            r"\bapt\b": "apartment",
            r"\bunit\b": "unit",
        }
        
        cleaned_lower = cleaned.lower()
        for pattern, replacement in replacements.items():
            cleaned_lower = re.sub(pattern, replacement, cleaned_lower)
        
        return cleaned_lower
    
    def _simplify_address(self, address: str) -> str:
        """Simplify address by removing unit numbers and extra details"""
        # Remove unit/apt numbers
        simplified = re.sub(
            r"(unit|apt|apartment|suite|#)\s*\d+[a-z]?\s*,?\s*",
            "",
            address,
            flags=re.IGNORECASE
        )
        
        # Remove floor numbers
        simplified = re.sub(
            r"\d+(st|nd|rd|th)\s+floor\s*,?\s*",
            "",
            simplified,
            flags=re.IGNORECASE
        )
        
        return simplified.strip()
    
    def _check_suspicious_patterns(
        self,
        address: str,
        listing_text: Optional[str]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Check for suspicious address patterns"""
        score = 0.0
        indicators = []
        addr_lower = address.lower()
        
        # Check for vague addresses
        vague_patterns = [
            r"^(downtown|near|close to|by the)",
            r"(area|neighborhood|district)\s*$",
            r"^[a-z]+\s+(area|downtown)$",
        ]
        
        for pattern in vague_patterns:
            if re.search(pattern, addr_lower):
                score += 0.3
                indicators.append({
                    "code": "ADDRESS_TOO_VAGUE",
                    "severity": 3,
                    "description": "Address is too vague - no specific street address",
                    "evidence": [f"Vague address: {address[:50]}"]
                })
                break
        
        # Check for missing street number
        if not re.search(r"\d+", address):
            score += 0.2
            indicators.append({
                "code": "ADDRESS_NO_NUMBER",
                "severity": 2,
                "description": "Address has no street number",
                "evidence": [f"No number found in: {address[:50]}"]
            })
        
        # Check for incomplete postal code (Canada format: A1A 1A1)
        postal_match = re.search(r"[a-z]\d[a-z]\s*\d[a-z]\d", addr_lower)
        if not postal_match and listing_text:
            text_has_postal = re.search(
                r"[a-z]\d[a-z]\s*\d[a-z]\d",
                listing_text.lower()
            )
            if not text_has_postal:
                score += 0.1
                indicators.append({
                    "code": "ADDRESS_NO_POSTAL_CODE",
                    "severity": 1,
                    "description": "No postal code found in listing",
                    "evidence": ["Missing Canadian postal code"]
                })
        
        # Check for obviously fake addresses
        fake_indicators = [
            "123 main street",
            "fake street",
            "example address",
            "test address",
            "sample street",
        ]
        
        for fake in fake_indicators:
            if fake in addr_lower:
                score += 0.5
                indicators.append({
                    "code": "ADDRESS_LIKELY_FAKE",
                    "severity": 5,
                    "description": "Address appears to be a placeholder/fake",
                    "evidence": [f"Suspicious pattern: {fake}"]
                })
                break
        
        # Check for known scam address patterns (PO boxes, mail drops, etc.)
        for pattern in KNOWN_SCAM_ADDRESS_PATTERNS:
            if re.search(pattern, addr_lower):
                score += 0.4
                indicators.append({
                    "code": "ADDRESS_SCAM_PATTERN",
                    "severity": 4,
                    "description": "Address matches a known scam pattern (PO box, mail drop, virtual office)",
                    "evidence": [f"Matched scam pattern in: {address[:60]}"]
                })
                break
        
        return min(score, 1.0), indicators
    
    async def _geocode_address(self, address: str) -> Optional[Dict[str, Any]]:
        """Geocode address using Nominatim API with caching and retry"""
        # Check geocode cache first
        cache_key = address.strip().lower()
        cached = self._geocode_cache_get(cache_key)
        if cached is not None:
            # cached can be the dict OR a sentinel empty dict for "no result"
            return cached if cached else None
        
        result = await self._geocode_address_with_retry(address)
        
        # Cache both hits and misses to avoid re-querying
        self._geocode_cache_put(cache_key, result if result else {})
        return result
    
    async def _geocode_address_with_retry(self, address: str) -> Optional[Dict[str, Any]]:
        """Execute geocoding request with retry logic for resilience"""
        last_error = None
        
        for attempt in range(self._MAX_RETRIES + 1):
            if attempt > 0:
                wait_time = self._RETRY_BACKOFF * attempt
                logger.warning(
                    f"Geocoding retry {attempt}/{self._MAX_RETRIES} "
                    f"for '{address[:40]}...' after {wait_time}s"
                )
                await asyncio.sleep(wait_time)
            
            await self._rate_limit()
            
            params = {
                "q": f"{address}, {self._DEFAULT_COUNTRY_NAME}",
                "format": "json",
                "addressdetails": 1,
                "limit": 1,
                "countrycodes": self._DEFAULT_COUNTRY
            }
            
            headers = {
                "User-Agent": self.user_agent
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.geocoding_url,
                        params=params,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data and len(data) > 0:
                                return data[0]
                            return None
                        elif response.status == 429:
                            # Rate limited — wait longer before retry
                            logger.warning("Nominatim rate limit hit (429)")
                            last_error = "Rate limited by geocoding service"
                            continue
                        elif response.status >= 500:
                            # Server error — retry
                            logger.warning(f"Nominatim server error ({response.status})")
                            last_error = f"Geocoding service error ({response.status})"
                            continue
                        else:
                            logger.error(f"Geocoding unexpected status: {response.status}")
                            return None
            except asyncio.TimeoutError:
                logger.warning(f"Geocoding timeout (attempt {attempt + 1})")
                last_error = "Geocoding request timed out"
                continue
            except Exception as e:
                logger.error(f"Geocoding error: {e}")
                last_error = str(e)
                continue
        
        logger.error(f"Geocoding failed after {self._MAX_RETRIES + 1} attempts: {last_error}")
        return None
    
    # ------------------------------------------------------------------
    #  Geocode cache (same LRU pattern as real_image_engine._analysis_cache)
    # ------------------------------------------------------------------
    def _geocode_cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached geocode result by normalized address key"""
        if key in self._geocode_cache:
            self._geocode_cache.move_to_end(key)
            logger.debug(f"Geocode cache hit for '{key[:30]}...'")
            return self._geocode_cache[key]
        return None
    
    def _geocode_cache_put(self, key: str, result: Optional[Dict[str, Any]]):
        """Store geocode result in cache, evicting oldest if full"""
        self._geocode_cache[key] = result
        self._geocode_cache.move_to_end(key)
        while len(self._geocode_cache) > self._GEOCODE_CACHE_MAX_SIZE:
            self._geocode_cache.popitem(last=False)
    
    async def _rate_limit(self):
        """Enforce rate limiting for API requests"""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _process_geocode_result(
        self,
        original_address: str,
        cleaned_address: str,
        geocode_data: Dict[str, Any],
        suspicion_score: float,
        suspicion_indicators: List[Dict[str, Any]],
        partial: bool = False
    ) -> AddressValidationResult:
        """Process geocoding result into validation result"""
        
        address_details = geocode_data.get("address", {})
        
        # Extract location components
        city = (
            address_details.get("city") or
            address_details.get("town") or
            address_details.get("municipality") or
            address_details.get("village")
        )
        province = address_details.get("state")
        postal_code = address_details.get("postcode")
        country = address_details.get("country")
        
        # ── Residential detection (expanded) ────────────────────────
        place_type = geocode_data.get("type", "")
        place_class = geocode_data.get("class", "")
        combined_type = f"{place_type} {place_class}".lower()
        
        is_residential = any(t in combined_type for t in RESIDENTIAL_TYPES)
        is_non_residential = any(t in combined_type for t in NON_RESIDENTIAL_TYPES)
        
        # If explicitly non-residential, override residential=True
        if is_non_residential:
            is_residential = False
        
        # ── Multi-factor confidence ─────────────────────────────────
        # Factor 1: Nominatim importance (PageRank-style, 0-1)
        importance = float(geocode_data.get("importance", 0.3))
        
        # Factor 2: Address completeness — how many components resolved
        completeness_parts = sum(1 for v in [city, province, postal_code, country] if v)
        completeness = completeness_parts / 4.0  # 0.0 - 1.0
        
        # Factor 3: Bounding box precision — smaller box = more precise
        bbox = geocode_data.get("boundingbox", [])
        bbox_precision = 0.5  # default if no bbox
        if len(bbox) == 4:
            try:
                lat_span = abs(float(bbox[1]) - float(bbox[0]))
                lon_span = abs(float(bbox[3]) - float(bbox[2]))
                # A house-level match has span < 0.001 degrees
                # A city-level match has span > 0.1 degrees
                total_span = lat_span + lon_span
                if total_span < 0.002:
                    bbox_precision = 1.0   # Very precise (building-level)
                elif total_span < 0.01:
                    bbox_precision = 0.85  # Street-level
                elif total_span < 0.05:
                    bbox_precision = 0.6   # Neighbourhood-level
                elif total_span < 0.2:
                    bbox_precision = 0.35  # District-level
                else:
                    bbox_precision = 0.15  # City-level or larger
            except (ValueError, IndexError):
                bbox_precision = 0.5
        
        # Factor 4: House number present in response
        has_house_number = bool(address_details.get("house_number"))
        house_number_bonus = 0.15 if has_house_number else 0.0
        
        # Weighted confidence formula
        confidence = min(
            (importance * 0.30) +
            (completeness * 0.25) +
            (bbox_precision * 0.25) +
            (house_number_bonus) +
            (0.05 if is_residential else 0.0),
            1.0
        )
        
        if partial:
            confidence *= 0.65
        
        # ── Determine status ────────────────────────────────────────
        if confidence > 0.7 and not suspicion_indicators:
            status = AddressValidationStatus.VALID
        elif confidence > 0.4:
            status = AddressValidationStatus.PARTIALLY_VALID
        elif suspicion_score > 0.5:
            status = AddressValidationStatus.SUSPICIOUS
        else:
            status = AddressValidationStatus.UNVERIFIABLE
        
        # ── Multi-signal risk score (weighted) ──────────────────────
        # Each signal contributes independently with its own weight.
        #   - suspicion_score: pattern-based fraud indicators (0-1)
        #   - geocode_quality: how well the address resolved
        #   - residential_signal: penalty for non-residential locations
        #   - postal_signal: penalty for missing postal code in result
        #   - match_quality: penalty for partial / imprecise match
        geocode_risk = (1.0 - confidence) * 0.25
        residential_risk = 0.15 if (is_non_residential and confidence > 0.4) else (0.0 if is_residential else 0.08)
        postal_risk = 0.10 if not postal_code else 0.0
        match_risk = 0.12 if partial else 0.0
        
        risk_score = min(
            suspicion_score * 0.40 +
            geocode_risk +
            residential_risk +
            postal_risk +
            match_risk,
            1.0
        )
        risk_score = max(risk_score, 0.0)
        
        indicators = suspicion_indicators.copy()
        
        if partial:
            indicators.append({
                "code": "ADDRESS_PARTIAL_MATCH",
                "severity": 2,
                "description": "Address only partially matched - full address not found",
                "evidence": [f"Matched: {geocode_data.get('display_name', '')[:80]}"]
            })
        
        if is_non_residential and confidence > 0.4:
            indicators.append({
                "code": "ADDRESS_NON_RESIDENTIAL",
                "severity": 3,
                "description": "Address resolves to a non-residential location (commercial/industrial)",
                "evidence": [f"Location type: {place_type or 'unknown'}, class: {place_class or 'unknown'}"]
            })
        elif not is_residential and confidence > 0.5:
            indicators.append({
                "code": "ADDRESS_NON_RESIDENTIAL",
                "severity": 2,
                "description": "Address may not be in a residential area",
                "evidence": [f"Location type: {place_type or 'unknown'}"]
            })
        
        if not postal_code and confidence > 0.3:
            indicators.append({
                "code": "ADDRESS_NO_POSTAL_IN_RESULT",
                "severity": 1,
                "description": "Geocoded location has no postal code — may be imprecise",
                "evidence": [f"Resolved to: {city or 'unknown'}, {province or 'unknown'}"]
            })
        
        # Generate explanation
        if status == AddressValidationStatus.VALID:
            explanation = (
                f"✅ Address verified: {city}, {province} {postal_code}. "
                f"Location resolves to valid coordinates with {confidence:.0%} confidence."
            )
        elif status == AddressValidationStatus.PARTIALLY_VALID:
            explanation = (
                f"⚠️ Address partially verified: Found in {city or 'unknown city'}, {province or 'unknown province'}. "
                f"Exact street address could not be fully confirmed."
            )
        elif status == AddressValidationStatus.SUSPICIOUS:
            explanation = (
                f"🚨 Address has suspicious characteristics. "
                f"Verify the exact location before proceeding."
            )
        else:
            explanation = (
                f"❓ Address could not be fully verified. "
                f"Manual verification recommended."
            )
        
        return AddressValidationResult(
            original_address=original_address,
            normalized_address=geocode_data.get("display_name"),
            status=status,
            confidence=confidence,
            latitude=float(geocode_data.get("lat", 0)),
            longitude=float(geocode_data.get("lon", 0)),
            city=city,
            province=province,
            postal_code=postal_code,
            country=country,
            is_residential=is_residential,
            risk_score=risk_score,
            indicators=indicators,
            explanation=explanation
        )
    
    def _create_invalid_result(
        self,
        address: str,
        reason: str,
        existing_indicators: Optional[List[Dict[str, Any]]] = None
    ) -> AddressValidationResult:
        """Create result for invalid/unfound address"""
        indicators = existing_indicators or []
        indicators.append({
            "code": "ADDRESS_NOT_FOUND",
            "severity": 4,
            "description": reason,
            "evidence": [f"Searched address: {address[:80]}"]
        })
        
        return AddressValidationResult(
            original_address=address,
            normalized_address=None,
            status=AddressValidationStatus.INVALID,
            confidence=0.0,
            latitude=None,
            longitude=None,
            city=None,
            province=None,
            postal_code=None,
            country=None,
            is_residential=False,
            risk_score=0.7,
            indicators=indicators,
            explanation=(
                f"🚨 Address could not be verified: {reason}. "
                f"This is a potential fraud indicator. Verify the address independently."
            )
        )
    
    def extract_address_from_text(self, text: str) -> Optional[str]:
        """Try to extract an address from listing text"""
        # Look for common address patterns
        patterns = [
            # Full address with postal code
            r"(\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|drive|dr|blvd|boulevard|way|lane|ln|court|ct|crescent|cres)[\w\s,]*[A-Z]\d[A-Z]\s*\d[A-Z]\d)",
            # Address without postal code
            r"(\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|drive|dr|blvd|boulevard|way|lane|ln|court|ct|crescent|cres)[,\s]*[\w\s]*(?:toronto|vancouver|calgary|montreal|ottawa|edmonton))",
            # Simple street address
            r"(\d+\s+[\w]+\s+(?:street|st|avenue|ave|road|rd|drive|dr|blvd|way|lane|ln|court|ct))",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None


# Singleton instance
address_validation_engine = AddressValidationEngine()
