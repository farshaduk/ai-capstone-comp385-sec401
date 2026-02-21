"""
Price Anomaly Detection Engine

This module provides AI-based price anomaly detection for rental listings.
It compares listing prices against market averages to identify:
- Suspiciously low prices (common in scams to attract victims)
- Unusual price patterns that deviate from local market norms

The engine uses statistical methods and can be enhanced with ML models
trained on historical pricing data.
"""

import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PriceRiskLevel(str, Enum):
    """Risk levels for price anomalies"""
    NORMAL = "normal"
    SLIGHTLY_LOW = "slightly_low"
    SUSPICIOUSLY_LOW = "suspiciously_low"
    EXTREMELY_LOW = "extremely_low"
    UNUSUALLY_HIGH = "unusually_high"


@dataclass
class PriceAnalysisResult:
    """Result of price anomaly analysis"""
    listing_price: float
    market_average: float
    market_median: float
    price_deviation_percent: float
    z_score: float
    risk_level: PriceRiskLevel
    risk_score: float  # 0-1
    indicators: List[Dict[str, Any]]
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "listing_price": self.listing_price,
            "market_average": round(self.market_average, 2),
            "market_median": round(self.market_median, 2),
            "price_deviation_percent": round(self.price_deviation_percent, 1),
            "z_score": round(self.z_score, 2),
            "risk_level": self.risk_level.value,
            "risk_score": round(self.risk_score, 3),
            "indicators": self.indicators,
            "explanation": self.explanation
        }


# Canadian city rental market data (monthly rent averages in CAD)
# Source: CMHC, Rentals.ca, and market reports
# Format: {city: {property_type: {bedrooms: average_rent}}}
CANADIAN_RENTAL_MARKET_DATA = {
    "toronto": {
        "apartment": {0: 1650, 1: 2200, 2: 2800, 3: 3400},
        "condo": {0: 1800, 1: 2400, 2: 3100, 3: 3800},
        "house": {1: 2200, 2: 2800, 3: 3500, 4: 4200},
        "basement": {0: 1200, 1: 1500, 2: 1900}
    },
    "vancouver": {
        "apartment": {0: 1700, 1: 2300, 2: 2900, 3: 3500},
        "condo": {0: 1900, 1: 2500, 2: 3200, 3: 3900},
        "house": {1: 2300, 2: 3000, 3: 3700, 4: 4500},
        "basement": {0: 1300, 1: 1600, 2: 2000}
    },
    "calgary": {
        "apartment": {0: 1200, 1: 1500, 2: 1800, 3: 2200},
        "condo": {0: 1300, 1: 1600, 2: 2000, 3: 2500},
        "house": {1: 1600, 2: 2000, 3: 2400, 4: 2900},
        "basement": {0: 900, 1: 1100, 2: 1400}
    },
    "edmonton": {
        "apartment": {0: 1000, 1: 1300, 2: 1600, 3: 1900},
        "condo": {0: 1100, 1: 1400, 2: 1700, 3: 2100},
        "house": {1: 1400, 2: 1700, 3: 2100, 4: 2500},
        "basement": {0: 800, 1: 1000, 2: 1300}
    },
    "ottawa": {
        "apartment": {0: 1400, 1: 1800, 2: 2200, 3: 2700},
        "condo": {0: 1500, 1: 1900, 2: 2400, 3: 2900},
        "house": {1: 1800, 2: 2300, 3: 2800, 4: 3400},
        "basement": {0: 1000, 1: 1300, 2: 1600}
    },
    "montreal": {
        "apartment": {0: 1100, 1: 1400, 2: 1800, 3: 2200},
        "condo": {0: 1200, 1: 1500, 2: 1900, 3: 2400},
        "house": {1: 1500, 2: 1900, 3: 2300, 4: 2800},
        "basement": {0: 800, 1: 1000, 2: 1300}
    },
    "winnipeg": {
        "apartment": {0: 900, 1: 1100, 2: 1400, 3: 1700},
        "condo": {0: 1000, 1: 1200, 2: 1500, 3: 1900},
        "house": {1: 1200, 2: 1500, 3: 1800, 4: 2200},
        "basement": {0: 700, 1: 900, 2: 1100}
    },
    "default": {
        "apartment": {0: 1200, 1: 1500, 2: 1900, 3: 2300},
        "condo": {0: 1300, 1: 1600, 2: 2100, 3: 2600},
        "house": {1: 1600, 2: 2000, 3: 2500, 4: 3000},
        "basement": {0: 900, 1: 1100, 2: 1400}
    }
}


class PriceAnomalyEngine:
    """
    AI-powered price anomaly detection for rental listings.
    
    Uses statistical analysis and market data to identify suspicious pricing:
    - Z-score calculation against market averages
    - Percentage deviation from median prices
    - Risk scoring based on how far price deviates from norm
    """
    
    def __init__(self):
        self.market_data = CANADIAN_RENTAL_MARKET_DATA
    
    def analyze(
        self,
        price: float,
        location: Optional[str] = None,
        property_type: Optional[str] = None,
        bedrooms: Optional[int] = None,
        listing_text: Optional[str] = None
    ) -> PriceAnalysisResult:
        """
        Analyze a listing price for anomalies.
        
        Args:
            price: The listing price (monthly rent)
            location: City/location string
            property_type: Type of property (apartment, condo, house, basement)
            bedrooms: Number of bedrooms
            listing_text: Full listing text (to extract missing info)
        
        Returns:
            PriceAnalysisResult with risk assessment
        """
        # Extract info from listing text if not provided
        if listing_text:
            if not location:
                location = self._extract_location(listing_text)
            if not property_type:
                property_type = self._extract_property_type(listing_text)
            if bedrooms is None:
                bedrooms = self._extract_bedrooms(listing_text)
        
        # Get market data for comparison
        market_avg, market_median, market_std = self._get_market_stats(
            location, property_type, bedrooms
        )
        
        # Calculate deviation metrics
        price_deviation = ((price - market_avg) / market_avg) * 100
        z_score = (price - market_avg) / market_std if market_std > 0 else 0
        
        # Determine risk level and score
        risk_level, risk_score = self._calculate_risk(price_deviation, z_score, price)
        
        # Generate indicators
        indicators = self._generate_indicators(
            price, market_avg, market_median, price_deviation, z_score, risk_level
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            price, market_avg, price_deviation, risk_level, location, property_type
        )
        
        return PriceAnalysisResult(
            listing_price=price,
            market_average=market_avg,
            market_median=market_median,
            price_deviation_percent=price_deviation,
            z_score=z_score,
            risk_level=risk_level,
            risk_score=risk_score,
            indicators=indicators,
            explanation=explanation
        )
    
    def _extract_location(self, text: str) -> str:
        """Extract city/location from listing text"""
        text_lower = text.lower()
        
        # Check for Canadian cities
        cities = [
            "toronto", "vancouver", "calgary", "edmonton", "ottawa",
            "montreal", "winnipeg", "hamilton", "kitchener", "london",
            "victoria", "halifax", "saskatoon", "regina", "mississauga",
            "brampton", "surrey", "burnaby", "richmond", "markham"
        ]
        
        for city in cities:
            if city in text_lower:
                # Map suburbs to major cities
                if city in ["mississauga", "brampton", "markham"]:
                    return "toronto"
                if city in ["surrey", "burnaby", "richmond"]:
                    return "vancouver"
                return city
        
        return "default"
    
    def _extract_property_type(self, text: str) -> str:
        """Extract property type from listing text"""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ["basement", "bsmt", "lower level"]):
            return "basement"
        if any(w in text_lower for w in ["condo", "condominium"]):
            return "condo"
        if any(w in text_lower for w in ["house", "detached", "semi-detached", "townhouse", "townhome", "town home", "town house", "bungalow", "duplex"]):
            return "house"
        
        return "apartment"
    
    def _extract_bedrooms(self, text: str) -> int:
        """Extract number of bedrooms from listing text"""
        text_lower = text.lower()
        
        # Check for explicit bedroom mentions
        patterns = [
            r"(\d+)\s*(?:bed|br|bedroom|bdrm)",
            r"(\d+)\s*(?:bed|br)(?:room)?s?",
            r"(?:studio|bachelor)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                if "studio" in pattern or "bachelor" in pattern:
                    return 0
                return int(match.group(1))
        
        # Default assumption
        return 1
    
    # Suburb / neighbourhood → major-city mapping for market data lookup
    LOCATION_ALIASES = {
        "mississauga": "toronto", "brampton": "toronto", "markham": "toronto",
        "scarborough": "toronto", "north york": "toronto", "etobicoke": "toronto",
        "vaughan": "toronto", "richmond hill": "toronto", "oakville": "toronto",
        "pickering": "toronto", "ajax": "toronto", "whitby": "toronto",
        "oshawa": "toronto", "burlington": "toronto", "milton": "toronto",
        "newmarket": "toronto", "aurora": "toronto",
        "surrey": "vancouver", "burnaby": "vancouver", "richmond": "vancouver",
        "coquitlam": "vancouver", "langley": "vancouver",
        "north vancouver": "vancouver", "west vancouver": "vancouver",
        "new westminster": "vancouver", "delta": "vancouver",
        "gatineau": "ottawa", "kanata": "ottawa",
        "laval": "montreal", "longueuil": "montreal",
        "airdrie": "calgary", "cochrane": "calgary",
        "st. albert": "edmonton", "sherwood park": "edmonton",
        "streetsville": "toronto",
    }

    def _get_market_stats(
        self,
        location: Optional[str],
        property_type: Optional[str],
        bedrooms: Optional[int]
    ) -> Tuple[float, float, float]:
        """Get market statistics for the given parameters"""
        # Normalize inputs
        city = (location or "default").lower().strip()
        prop_type = (property_type or "apartment").lower().strip()
        beds = bedrooms if bedrooms is not None else 1
        
        # Resolve suburb / neighbourhood aliases to their major city
        city = self.LOCATION_ALIASES.get(city, city)
        # Also try partial matching for multi-word locations
        if city not in self.market_data:
            for alias, major_city in self.LOCATION_ALIASES.items():
                if alias in city or city in alias:
                    city = major_city
                    break
        
        # Get city data (fallback to default)
        city_data = self.market_data.get(city, self.market_data["default"])
        
        # Get property type data (fallback to apartment)
        type_data = city_data.get(prop_type, city_data.get("apartment", {}))
        
        # Get bedroom data (find closest match)
        if beds in type_data:
            avg_price = type_data[beds]
        else:
            # Find closest bedroom count
            available_beds = list(type_data.keys())
            if available_beds:
                closest = min(available_beds, key=lambda x: abs(x - beds))
                avg_price = type_data[closest]
            else:
                avg_price = 1500  # Default fallback
        
        # Calculate variance (assume ~20% standard deviation)
        std_dev = avg_price * 0.20
        median = avg_price * 0.95  # Median typically slightly lower
        
        return avg_price, median, std_dev
    
    def _calculate_risk(
        self,
        deviation_percent: float,
        z_score: float,
        price: float
    ) -> Tuple[PriceRiskLevel, float]:
        """Calculate risk level and score based on price deviation"""
        
        # Extremely low prices are major red flags
        if deviation_percent <= -50:
            return PriceRiskLevel.EXTREMELY_LOW, 0.95
        elif deviation_percent <= -35:
            return PriceRiskLevel.SUSPICIOUSLY_LOW, 0.80
        elif deviation_percent <= -20:
            return PriceRiskLevel.SLIGHTLY_LOW, 0.50
        elif deviation_percent >= 50:
            return PriceRiskLevel.UNUSUALLY_HIGH, 0.30
        else:
            return PriceRiskLevel.NORMAL, 0.05
    
    def _generate_indicators(
        self,
        price: float,
        market_avg: float,
        market_median: float,
        deviation: float,
        z_score: float,
        risk_level: PriceRiskLevel
    ) -> List[Dict[str, Any]]:
        """Generate risk indicators based on price analysis"""
        indicators = []
        
        if risk_level == PriceRiskLevel.EXTREMELY_LOW:
            indicators.append({
                "code": "PRICE_EXTREMELY_LOW",
                "severity": 5,
                "description": f"Price is {abs(deviation):.0f}% below market average - major scam indicator",
                "evidence": [f"Listed: ${price:.0f}/mo", f"Market avg: ${market_avg:.0f}/mo"]
            })
        elif risk_level == PriceRiskLevel.SUSPICIOUSLY_LOW:
            indicators.append({
                "code": "PRICE_SUSPICIOUSLY_LOW",
                "severity": 4,
                "description": f"Price is {abs(deviation):.0f}% below market average - potential scam",
                "evidence": [f"Listed: ${price:.0f}/mo", f"Market avg: ${market_avg:.0f}/mo"]
            })
        elif risk_level == PriceRiskLevel.SLIGHTLY_LOW:
            indicators.append({
                "code": "PRICE_BELOW_MARKET",
                "severity": 2,
                "description": f"Price is {abs(deviation):.0f}% below market average",
                "evidence": [f"Listed: ${price:.0f}/mo", f"Market avg: ${market_avg:.0f}/mo"]
            })
        elif risk_level == PriceRiskLevel.UNUSUALLY_HIGH:
            indicators.append({
                "code": "PRICE_ABOVE_MARKET",
                "severity": 1,
                "description": f"Price is {deviation:.0f}% above market average",
                "evidence": [f"Listed: ${price:.0f}/mo", f"Market avg: ${market_avg:.0f}/mo"]
            })
        
        # Add statistical indicator
        if abs(z_score) > 2:
            indicators.append({
                "code": "PRICE_STATISTICAL_OUTLIER",
                "severity": 3,
                "description": f"Price is a statistical outlier (z-score: {z_score:.1f})",
                "evidence": [f"More than 2 standard deviations from mean"]
            })
        
        return indicators
    
    def _generate_explanation(
        self,
        price: float,
        market_avg: float,
        deviation: float,
        risk_level: PriceRiskLevel,
        location: Optional[str],
        property_type: Optional[str]
    ) -> str:
        """Generate human-readable explanation of price analysis"""
        
        location_str = location.title() if location and location != "default" else "this area"
        prop_str = property_type or "this type of property"
        
        if risk_level == PriceRiskLevel.EXTREMELY_LOW:
            return (
                f"🚨 MAJOR WARNING: This listing's price of ${price:.0f}/month is {abs(deviation):.0f}% "
                f"below the market average of ${market_avg:.0f} for {prop_str} in {location_str}. "
                f"This is a common tactic used by scammers to attract victims quickly. "
                f"Exercise extreme caution and verify the listing thoroughly before proceeding."
            )
        elif risk_level == PriceRiskLevel.SUSPICIOUSLY_LOW:
            return (
                f"⚠️ WARNING: This price of ${price:.0f}/month is {abs(deviation):.0f}% below "
                f"market average (${market_avg:.0f}) for {location_str}. While not impossible, "
                f"significantly below-market prices are a common fraud indicator. "
                f"Verify the listing details and landlord identity carefully."
            )
        elif risk_level == PriceRiskLevel.SLIGHTLY_LOW:
            return (
                f"ℹ️ This price of ${price:.0f}/month is {abs(deviation):.0f}% below market average. "
                f"This could be a good deal or may warrant additional verification."
            )
        elif risk_level == PriceRiskLevel.UNUSUALLY_HIGH:
            return (
                f"ℹ️ This price of ${price:.0f}/month is {deviation:.0f}% above market average "
                f"for {location_str}. This may be justified by premium features or location."
            )
        else:
            return (
                f"✅ This price of ${price:.0f}/month is within normal market range "
                f"for {prop_str} in {location_str} (avg: ${market_avg:.0f}/month)."
            )


# Singleton instance
price_anomaly_engine = PriceAnomalyEngine()
