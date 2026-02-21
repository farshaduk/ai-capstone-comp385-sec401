"""
Real AI Image Classification Engine - CNN-based property image analysis

This module uses pretrained deep learning models (ResNet, EfficientNet) to:
1. Classify if an image is property-related (interior, exterior, room, etc.)
2. Detect AI-generated images using statistical analysis
3. Assess image quality and authenticity
4. Identify suspicious patterns in listing photos
5. GAN fingerprint detection (spectral peaks, checkerboard artifacts)
6. Diffusion artifact classification (banding, tiling seams, attention leakage)
7. Camera fingerprint / sensor noise analysis (PRNU estimation)
8. Deep EXIF analysis (GPS, date plausibility, thumbnail mismatch)
9. Tampering detection (ELA, double JPEG, copy-move, edge inconsistency)
10. Face/object consistency checks (scale, shadow, perspective)
11. Perceptual hash duplicate detection across listings
12. Confidence calibration and explainability scoring

Uses PyTorch and torchvision for real CNN-based classification.
"""

import os
import io
import json
import hashlib
import asyncio
import threading
import struct
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from collections import OrderedDict
import logging
from PIL import Image
import base64

try:
    _LANCZOS = Image.Resampling.LANCZOS  # Pillow >= 9
except AttributeError:
    _LANCZOS = Image.LANCZOS  # Older Pillow

logger = logging.getLogger(__name__)


def _safe_project_root() -> str:
    """
    Resolve a stable base directory for data files.
    Works in normal python, tests, and PyInstaller-like bundles.
    """
    # 1) Best case: file-based module
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(here, "..", ".."))
    except NameError:
        pass  # __file__ not defined

    # 2) Fallback: current working directory (common for notebooks)
    return os.path.abspath(os.getcwd())


# Lazy load heavy ML dependencies
_classifier_model = None
_transforms = None
_device = None
_IMAGENET_LABELS = None  # All 1000 ImageNet class names (populated on model load)


# ---------------------------------------------------------------------------
#  Keyword-based property / suspicious classification
#  Instead of hardcoding ImageNet class IDs (which are error-prone and often
#  WRONG), we match against the *actual* label strings shipped with the model
#  weights.  This is far more reliable.
# ---------------------------------------------------------------------------

def _term_in_label(label_lower: str, term: str) -> bool:
    """Match term as complete word(s) in label — avoids 'pot' matching 'spotted'"""
    label_norm = label_lower.replace("-", " ").replace("_", " ").replace(",", " ")
    term_norm = term.replace("-", " ").replace("_", " ")
    # Multi-word terms: substring match (they are specific enough)
    if " " in term_norm:
        return term_norm in label_norm
    # Single words: exact token match only
    return term_norm in label_norm.split()


def _get_property_weight(label_lower: str):
    """Return property-confidence weight if label is property-related, else None"""
    # Strong indicators (weight 0.7) — unmistakably interior / property
    for t in ["bathtub", "refrigerator", "sliding_door", "patio",
              "four-poster", "studio_couch", "dishwasher", "shower_curtain",
              "toilet_seat", "washbasin", "medicine_chest", "shower_cap",
              "tub", "kitchen", "living_room", "bedroom", "bathroom",
              "dining_room", "laundry", "doorway"]:
        if _term_in_label(label_lower, t):
            return 0.7
    # Good indicators (weight 0.55)
    for t in ["sofa", "day_bed", "stove", "microwave", "dining_table",
              "wardrobe", "bookcase", "china_cabinet", "chiffonier",
              "window_shade", "window_screen", "entertainment_center",
              "fire_screen", "dutch_oven", "espresso", "desk",
              "chest_of_drawers", "filing_cabinet", "home_theater",
              "rocking_chair", "folding_chair", "barber_chair",
              "park_bench", "throne", "bannister", "handrail",
              "tile_roof", "pool_table", "ping_pong_ball",
              "iron", "washer", "dryer", "dishrag", "soap_dispenser"]:
        if _term_in_label(label_lower, t):
            return 0.55
    # Medium indicators (weight 0.4)
    for t in ["chair", "table", "lamp", "bed", "crib", "cradle",
              "pillow", "quilt", "television", "monitor", "vacuum",
              "toaster", "coffeepot", "waffle_iron", "plunger", "doormat",
              "curtain", "screen_door", "safe", "hamper", "clock", "vase",
              "pot", "wok", "caldron", "plate_rack", "radiator",
              "electric_fan", "space_heater", "greenhouse",
              "library", "bookshop", "flowerpot", "fountain",
              "bannister", "picket_fence", "chain_link_fence",
              "stone_wall", "lampshade", "candelabra", "candle",
              "wall_clock", "analog_clock", "digital_clock",
              "bucket", "basket", "chest", "crate",
              "french_loaf", "pizza", "plate",
              "remote_control", "cellular_telephone", "desktop_computer",
              "laptop", "notebook", "printer", "switch",
              "toilet_tissue", "paper_towel", "tray",
              "pool", "swimming", "jacuzzi", "hot_tub"]:
        if _term_in_label(label_lower, t):
            return 0.4
    # Weak indicators — buildings/structures/outdoor (weight 0.25)
    for t in ["castle", "church", "palace", "monastery", "dome",
              "thatch", "bell_cote", "barn", "boathouse", "lakeside",
              "cliff_dwelling", "mobile_home", "trailer",
              "lumbermill", "steel_arch_bridge", "suspension_bridge",
              "pier", "dam", "breakwater", "dock", "beacon",
              "garden", "lawn_mower", "plow", "shovel",
              "mailbox", "street_sign", "parking_meter",
              "maze", "alp", "valley", "seashore", "promontory"]:
        if _term_in_label(label_lower, t):
            return 0.25
    return None


def _get_suspicious_weight(label_lower: str):
    """Return suspicion weight if label is clearly non-property, else None"""
    # Animals — strong suspicion (weight 0.5)
    for t in ["tabby", "persian_cat", "siamese_cat", "egyptian_cat",
              "tiger_cat", "retriever", "shepherd", "chihuahua",
              "dalmatian", "rottweiler", "poodle", "beagle", "collie",
              "hound", "terrier", "bulldog", "pug", "malamute", "husky",
              "dingo", "boxer", "snake", "spider", "tarantula",
              "scorpion", "centipede", "iguana", "chameleon", "gecko",
              "african_elephant", "indian_elephant", "lion", "tiger",
              "cheetah", "leopard", "jaguar", "bear", "polar_bear",
              "gorilla", "chimpanzee", "orangutan", "baboon",
              "zebra", "hippopotamus", "rhinoceros", "bison",
              "ram", "ibex", "gazelle", "impala", "hartebeest",
              "fox", "wolf", "coyote", "hyena",
              "whale", "dolphin", "shark", "stingray",
              "jellyfish", "starfish", "sea_urchin", "coral",
              "lobster", "crab", "hermit_crab", "snail",
              "flamingo", "pelican", "albatross", "ostrich",
              "peacock", "macaw", "cockatoo", "toucan",
              "salamander", "axolotl", "frog", "tree_frog",
              "turtle", "tortoise", "crocodile", "alligator",
              "dragonfly", "butterfly", "moth", "bee", "ant",
              "grasshopper", "cockroach", "ladybug", "beetle"]:
        if _term_in_label(label_lower, t):
            return 0.5
    # Vehicles & transport — moderate suspicion (weight 0.4)
    for t in ["sports_car", "race_car", "convertible", "limousine",
              "minivan", "ambulance", "fire_engine", "police_van",
              "airliner", "warplane", "fighter", "space_shuttle",
              "submarine", "aircraft_carrier", "speedboat",
              "tank", "missile", "cannon", "projectile",
              "steam_locomotive", "bullet_train", "freight_car",
              "mountain_bike", "bicycle", "motor_scooter",
              "snowmobile", "go_kart", "tractor",
              "forklift", "trailer_truck", "moving_van"]:
        if _term_in_label(label_lower, t):
            return 0.4
    # Weapons, inappropriate content — high suspicion (weight 0.6)
    for t in ["rifle", "revolver", "assault_rifle", "holster",
              "bikini", "maillot", "swimming_trunks",
              "guillotine", "gas_mask", "bulletproof_vest"]:
        if _term_in_label(label_lower, t):
            return 0.6
    # Non-property objects — mild suspicion (weight 0.3)
    for t in ["comic_book", "running_shoe", "cowboy_boot",
              "sombrero", "military_uniform", "mortarboard",
              "stethoscope", "syringe", "pill_bottle",
              "slot_machine", "pinball_machine", "joystick",
              "drum", "guitar", "violin", "saxophone",
              "baseball", "basketball", "soccer_ball", "golf_ball",
              "tennis_ball", "volleyball", "football_helmet",
              "parachute", "ski", "snowboard",
              "barbell", "dumbbell"]:
        if _term_in_label(label_lower, t):
            return 0.3
    return None


def _build_class_mappings(labels):
    """Build PROPERTY_RELATED_CLASSES & SUSPICIOUS_CLASSES from real labels"""
    global PROPERTY_RELATED_CLASSES, SUSPICIOUS_CLASSES
    prop, susp = {}, {}
    for idx, label in enumerate(labels):
        l = label.lower().replace("-", "_").replace(" ", "_")
        # Check suspicious FIRST — animals/vehicles override any property keyword
        # (e.g. "barn spider" should be suspicious, not matched on "barn")
        sw = _get_suspicious_weight(l)
        if sw is not None:
            susp[idx] = (label, sw)
            continue
        pw = _get_property_weight(l)
        if pw is not None:
            prop[idx] = (label, pw)
    PROPERTY_RELATED_CLASSES = prop
    SUSPICIOUS_CLASSES = susp
    logger.info(
        f"Mapped {len(prop)} property classes, {len(susp)} suspicious "
        f"classes from {len(labels)} ImageNet labels"
    )


def _load_image_classifier():
    """Lazy load the image classification model"""
    global _classifier_model, _transforms, _device, _IMAGENET_LABELS
    
    if _classifier_model is not None:
        return _classifier_model, _transforms, _device
    
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as T
        
        # Use GPU if available
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained ResNet50 – prefer weights API (gives real labels)
        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1
            _classifier_model = models.resnet50(weights=weights)
            _IMAGENET_LABELS = list(weights.meta["categories"])
            logger.info(f"Loaded {len(_IMAGENET_LABELS)} ImageNet class labels")
        except (ImportError, AttributeError):
            # Fallback for older torchvision (< 0.13)
            _classifier_model = models.resnet50(pretrained=True)
            _IMAGENET_LABELS = None
            logger.warning("Weights API unavailable – using fallback class IDs")
            logger.warning(
                "Using hardcoded ImageNet indices; results may vary across torchvision versions. "
                "Consider pinning torchvision>=0.13 for stable labels."
            )
        
        _classifier_model = _classifier_model.to(_device)
        _classifier_model.eval()
        
        # Build dynamic class mappings from real labels
        if _IMAGENET_LABELS:
            _build_class_mappings(_IMAGENET_LABELS)
        
        # Standard ImageNet preprocessing
        _transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"Image classifier loaded on {_device}")
        return _classifier_model, _transforms, _device
        
    except Exception as e:
        logger.error(f"Failed to load image classifier: {e}")
        return None, None, None


# ---------- Fallback hardcoded mappings (older torchvision only) -----------
# These are ONLY used when the weights API cannot provide real labels.
# Corrected ImageNet indices for property-related categories.
PROPERTY_RELATED_CLASSES = {
    # Interior / Room – corrected indices
    765: ("rocking_chair", 0.3),
    559: ("folding_chair", 0.3),
    831: ("studio_couch", 0.5),
    857: ("bathtub", 0.6),
    861: ("toilet_seat", 0.5),
    532: ("dining_table", 0.5),
    526: ("desk", 0.5),
    790: ("shower_curtain", 0.5),
    846: ("lampshade", 0.4),
    799: ("sliding_door", 0.6),
    827: ("stove", 0.5),
    760: ("refrigerator", 0.6),
    839: ("television", 0.4),
    534: ("dishwasher", 0.5),
    894: ("wardrobe", 0.5),
    904: ("window_shade", 0.4),
    905: ("window_screen", 0.4),
    453: ("bookcase", 0.4),
    689: ("pillow", 0.3),
    750: ("quilt", 0.3),
    508: ("coffeepot", 0.3),
    651: ("microwave", 0.5),
    # Exterior / Building
    483: ("castle", 0.2),
    497: ("church", 0.2),
    536: ("greenhouse", 0.3),
    663: ("monastery", 0.2),
    698: ("palace", 0.2),
    706: ("patio", 0.6),
}

# Non-property categories that suggest suspicious images
SUSPICIOUS_CLASSES = {
    281: ("tabby_cat", 0.4),
    282: ("tiger_cat", 0.4),
    207: ("golden_retriever", 0.4),
    151: ("Chihuahua", 0.4),
    770: ("running_shoe", 0.3),
    834: ("sweatshirt", 0.2),
    920: ("comic_book", 0.5),
}


class ImageRiskLevel(str, Enum):
    """Risk levels for image analysis"""
    AUTHENTIC = "authentic"
    LIKELY_AUTHENTIC = "likely_authentic"
    UNCERTAIN = "uncertain"
    SUSPICIOUS = "suspicious"
    LIKELY_FAKE = "likely_fake"


@dataclass
class ImageClassificationResult:
    """Result of CNN image classification"""
    is_property_related: bool
    property_confidence: float
    top_classes: List[Dict[str, Any]]
    risk_indicators: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_property_related": self.is_property_related,
            "property_confidence": round(self.property_confidence, 3),
            "top_classes": self.top_classes,
            "risk_indicators": self.risk_indicators
        }


@dataclass
class ImageAnalysisResult:
    """Complete image analysis result"""
    image_hash: str
    risk_level: ImageRiskLevel
    risk_score: float  # 0-1, higher = riskier
    is_property_image: bool
    property_confidence: float
    quality_score: float
    ai_detection_score: float
    indicators: List[Dict[str, Any]]
    classification: Optional[ImageClassificationResult]
    metadata: Dict[str, Any]
    explanation: str
    web_detection: Optional[Dict[str, Any]] = None
    # --- Advanced forensic fields ---
    gan_fingerprint_score: float = 0.0
    diffusion_artifact_score: float = 0.0
    sensor_noise_score: float = 0.0
    tampering_score: float = 0.0
    content_consistency_score: float = 1.0  # 1.0 = fully consistent
    duplicate_listings: List[str] = field(default_factory=list)
    perceptual_hash: Optional[str] = None
    deep_exif: Optional[Dict[str, Any]] = None
    confidence_calibration: Optional[Dict[str, Any]] = None
    explainability: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_hash": self.image_hash,
            "risk_level": self.risk_level.value,
            "risk_score": round(self.risk_score, 3),
            "is_property_image": self.is_property_image,
            "property_confidence": round(self.property_confidence, 3),
            "quality_score": round(self.quality_score, 3),
            "ai_detection_score": round(self.ai_detection_score, 3),
            "indicators": self.indicators,
            "classification": self.classification.to_dict() if self.classification else None,
            "metadata": self.metadata,
            "explanation": self.explanation,
            "web_detection": self.web_detection,
            "forensics": {
                "gan_fingerprint_score": round(self.gan_fingerprint_score, 3),
                "diffusion_artifact_score": round(self.diffusion_artifact_score, 3),
                "sensor_noise_score": round(self.sensor_noise_score, 3),
                "tampering_score": round(self.tampering_score, 3),
                "content_consistency_score": round(self.content_consistency_score, 3),
                "duplicate_listings": self.duplicate_listings,
                "perceptual_hash": self.perceptual_hash,
            },
            "deep_exif": self.deep_exif,
            "confidence_calibration": self.confidence_calibration,
            "explainability": self.explainability,
        }


class RealImageClassificationEngine:
    """
    Real AI-powered image analysis using pretrained CNN models.
    
    Capabilities:
    1.  Scene Classification - Is this a property/room/interior image?
    2.  Quality Analysis - Resolution, artifacts, compression quality
    3.  AI Detection - Statistical analysis for AI-generated patterns
    4.  Metadata Analysis - EXIF data, software signatures
    5.  GAN Fingerprint Detection - Spectral peaks, checkerboard artifacts
    6.  Diffusion Artifact Detection - Banding, tiling seams, attention leakage
    7.  Camera Fingerprint / Sensor Noise - PRNU estimation
    8.  Deep EXIF Analysis - GPS, date, thumbnail mismatch, compression history
    9.  Tampering Detection - ELA, double JPEG, copy-move, edge inconsistency
    10. Face/Object Consistency - Scale, shadow, perspective checks
    11. Perceptual Hash Duplicate Detection - Cross-listing reuse detection
    12. Confidence Calibration - Platt-scaled ensemble agreement
    13. Explainability - Per-signal contribution breakdown
    """
    
    # LRU analysis cache — avoids re-analyzing the same image bytes
    _CACHE_MAX_SIZE = 256
    
    # Maximum number of perceptual hashes to track for duplicate detection
    _HASH_DB_MAX_SIZE = 50_000
    
    def __init__(self):
        self._model = None
        self._transforms = None
        self._device = None
        
        # Analysis result cache keyed by SHA-256 hash
        self._analysis_cache: OrderedDict[str, ImageAnalysisResult] = OrderedDict()
        
        # Thread-safety lock for hash database mutations
        self._hash_lock = threading.Lock()
        
        # Perceptual hash database for duplicate detection
        base_dir = _safe_project_root()
        self._hash_db_path = os.path.join(base_dir, "data", "image_hashes.json")
        self._hash_database: Dict[str, List[str]] = {}
        self._load_hash_database()
        
        # Load ImageNet class labels
        self._load_imagenet_labels()
    
    def _load_imagenet_labels(self):
        """Load ImageNet class labels – prefer full 1000-class list from model"""
        if _IMAGENET_LABELS:
            # Use ALL real ImageNet labels (populated by _load_image_classifier)
            self.class_labels = {i: name for i, name in enumerate(_IMAGENET_LABELS)}
        else:
            # Fallback: only property + suspicious entries
            self.class_labels = {}
            for idx in PROPERTY_RELATED_CLASSES:
                self.class_labels[idx] = PROPERTY_RELATED_CLASSES[idx][0]
            for idx in SUSPICIOUS_CLASSES:
                self.class_labels[idx] = SUSPICIOUS_CLASSES[idx][0]
    
    # ------------------------------------------------------------------
    #  Perceptual hash database (ported from image_analysis_engine.py)
    # ------------------------------------------------------------------
    def _load_hash_database(self):
        """Load perceptual hash database from disk"""
        try:
            if os.path.exists(self._hash_db_path):
                with open(self._hash_db_path, 'r') as f:
                    self._hash_database = json.load(f)
                logger.info(f"Loaded {len(self._hash_database)} image hashes from database")
            else:
                self._hash_database = {}
        except Exception as e:
            logger.warning(f"Could not load hash database: {e}")
            self._hash_database = {}
    
    def _save_hash_database(self):
        """Persist perceptual hash database to disk"""
        try:
            os.makedirs(os.path.dirname(self._hash_db_path), exist_ok=True)
            with open(self._hash_db_path, 'w') as f:
                json.dump(self._hash_database, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save hash database: {e}")
    
    def _register_image_hash(self, phash: str, listing_id: str):
        """Register a perceptual hash for cross-listing duplicate detection"""
        with self._hash_lock:
            if phash not in self._hash_database:
                self._hash_database[phash] = []
            if listing_id not in self._hash_database[phash]:
                self._hash_database[phash].append(listing_id)
                # Evict oldest entries if hash database exceeds size cap
                while len(self._hash_database) > self._HASH_DB_MAX_SIZE:
                    oldest_key = next(iter(self._hash_database))
                    del self._hash_database[oldest_key]
                self._save_hash_database()
    
    # ------------------------------------------------------------------
    #  Analysis cache
    # ------------------------------------------------------------------
    def _cache_get(self, sha256_hash: str) -> Optional[ImageAnalysisResult]:
        """Retrieve cached result by full SHA-256 hash"""
        if sha256_hash in self._analysis_cache:
            # Move to end (most recently used)
            self._analysis_cache.move_to_end(sha256_hash)
            logger.debug(f"Cache hit for image {sha256_hash[:12]}")
            return self._analysis_cache[sha256_hash]
        return None
    
    def _cache_put(self, sha256_hash: str, result: ImageAnalysisResult):
        """Store result in cache, evicting oldest if full"""
        self._analysis_cache[sha256_hash] = result
        self._analysis_cache.move_to_end(sha256_hash)
        while len(self._analysis_cache) > self._CACHE_MAX_SIZE:
            self._analysis_cache.popitem(last=False)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get image analysis engine statistics"""
        return {
            "total_hashes_tracked": len(self._hash_database),
            "duplicate_images": len([h for h, l in self._hash_database.items() if len(l) > 1]),
            "cache_size": len(self._analysis_cache),
            "cache_max": self._CACHE_MAX_SIZE,
        }
    
    def _ensure_model_loaded(self):
        """Ensure the CNN model is loaded and class labels are up-to-date"""
        if self._model is None:
            self._model, self._transforms, self._device = _load_image_classifier()
            # Refresh class labels now that _IMAGENET_LABELS is populated
            if _IMAGENET_LABELS:
                self.class_labels = {i: name for i, name in enumerate(_IMAGENET_LABELS)}
    
    async def analyze_image(
        self,
        image_data: bytes,
        filename: Optional[str] = None,
        listing_id: Optional[str] = None
    ) -> ImageAnalysisResult:
        """
        Analyze a single image for authenticity and property relevance.
        
        Runs 13 analysis stages:
          1. CNN Scene Classification
          2. Image Quality Analysis
          3. AI Generation Detection (statistical forensics)
          4. GAN Fingerprint Detection
          5. Diffusion Artifact Detection
          6. Camera Sensor Noise / PRNU Analysis
          7. Deep EXIF Analysis
          8. Tampering Detection (ELA, copy-move)
          9. Face/Object Consistency Check
         10. Metadata Extraction
         11. Perceptual Hash Duplicate Detection
         12. Reverse Image Search (Google Cloud Vision)
         13. Confidence Calibration + Explainability
        
        Args:
            image_data: Raw image bytes
            filename: Optional filename for metadata
            listing_id: Optional listing ID for duplicate tracking
        
        Returns:
            ImageAnalysisResult with detailed analysis
        """
        try:
            # Generate image hash
            full_sha256 = hashlib.sha256(image_data).hexdigest()
            image_hash = full_sha256[:16]
            
            # Check cache first
            cached = self._cache_get(full_sha256)
            if cached is not None:
                return cached
            
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Preserve original format before any conversion (convert() sets .format=None)
            original_format = image.format
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                image.format = original_format  # Restore so downstream checks work
            
            # 1. Run CNN Classification
            classification = await self._classify_image(image)
            
            # 2. Analyze image quality
            quality_score, quality_indicators = await asyncio.to_thread(self._analyze_quality, image)
            
            # 3. Detect AI generation patterns (statistical forensics)
            ai_score, ai_indicators = await asyncio.to_thread(self._detect_ai_generation, image)
            
            # 4. GAN Fingerprint Detection
            gan_score, gan_indicators = await asyncio.to_thread(self._detect_gan_fingerprints, image)
            
            # 5. Diffusion Artifact Detection
            diffusion_score, diffusion_indicators = await asyncio.to_thread(self._detect_diffusion_artifacts, image)
            
            # 6. Camera Sensor Noise / PRNU Analysis
            sensor_score, sensor_indicators = await asyncio.to_thread(self._analyze_sensor_noise, image)
            
            # 7. Deep EXIF Analysis
            deep_exif, exif_indicators = await asyncio.to_thread(self._deep_exif_analysis, image, filename)
            
            # 8. Tampering Detection (ELA, double JPEG, copy-move)
            tampering_score, tampering_indicators = await asyncio.to_thread(self._detect_tampering, image, image_data)
            
            # 9. Face/Object Consistency Check
            consistency_score, consistency_indicators = await asyncio.to_thread(self._check_content_consistency, image)
            
            # 10. Extract basic metadata
            metadata = await asyncio.to_thread(self._extract_metadata, image, filename)
            
            # 11. Perceptual Hash + Duplicate Detection
            phash = await asyncio.to_thread(self._calculate_perceptual_hash, image)
            duplicate_listings = await asyncio.to_thread(self._check_duplicates, phash, listing_id or "unknown")
            if listing_id:
                self._register_image_hash(phash, listing_id)
            
            # 12. Reverse image search via Google Cloud Vision
            web_detection = await self._web_detection(image_data)
            
            # 13. Combine all indicators
            all_indicators = []
            all_indicators.extend(classification.risk_indicators if classification else [])
            all_indicators.extend(quality_indicators)
            all_indicators.extend(ai_indicators)
            all_indicators.extend(gan_indicators)
            all_indicators.extend(diffusion_indicators)
            all_indicators.extend(sensor_indicators)
            all_indicators.extend(exif_indicators)
            all_indicators.extend(tampering_indicators)
            all_indicators.extend(consistency_indicators)
            
            # Duplicate detection indicators
            if duplicate_listings:
                all_indicators.append({
                    "code": "DUPLICATE_ACROSS_LISTINGS",
                    "severity": 4,
                    "description": f"Image reused across {len(duplicate_listings)} other listing(s) — common fraud tactic",
                    "evidence": duplicate_listings[:5]
                })
            
            # Add web-detection indicators
            if web_detection and web_detection.get("has_web_matches"):
                full_ct = len(web_detection.get("full_matching_images", []))
                partial_ct = len(web_detection.get("partial_matching_images", []))
                pages_ct = len(web_detection.get("pages_with_matching_images", []))
                
                if full_ct > 0:
                    severity = 4 if full_ct >= 3 else 3
                    all_indicators.append({
                        "code": "WEB_EXACT_MATCH",
                        "severity": severity,
                        "description": f"Exact image found on {full_ct} website{'' if full_ct == 1 else 's'} — may be stock or stolen",
                        "evidence": [m["url"] for m in web_detection["full_matching_images"][:3]]
                    })
                if partial_ct > 0:
                    severity = 3 if partial_ct >= 3 else 2
                    all_indicators.append({
                        "code": "WEB_PARTIAL_MATCH",
                        "severity": severity,
                        "description": f"Visually similar images found on {partial_ct} website{'' if partial_ct == 1 else 's'}",
                        "evidence": [m["url"] for m in web_detection["partial_matching_images"][:3]]
                    })
                if pages_ct > 0:
                    all_indicators.append({
                        "code": "WEB_PAGES_FOUND",
                        "severity": 2,
                        "description": f"Image appears on {pages_ct} web page{'' if pages_ct == 1 else 's'}",
                        "evidence": [p.get("page_title") or p["url"] for p in web_detection["pages_with_matching_images"][:3]]
                    })
            
            # Merge AI subscores into composite ai_detection_score
            composite_ai = self._merge_ai_scores(
                ai_score, gan_score, diffusion_score
            )
            
            # 14. Calculate overall risk
            risk_score, risk_level = self._calculate_risk(
                classification, quality_score, composite_ai, all_indicators,
                sensor_score=sensor_score,
                tampering_score=tampering_score,
                consistency_score=consistency_score,
                duplicate_count=len(duplicate_listings),
            )
            
            # 15. Confidence calibration
            calibration = self._calibrate_confidence(
                ai_score=composite_ai,
                gan_score=gan_score,
                diffusion_score=diffusion_score,
                sensor_score=sensor_score,
                tampering_score=tampering_score,
                quality_score=quality_score,
                risk_score=risk_score,
            )
            
            # 16. Explainability
            explainability = self._compute_explainability(
                ai_score=composite_ai,
                gan_score=gan_score,
                diffusion_score=diffusion_score,
                sensor_score=sensor_score,
                tampering_score=tampering_score,
                consistency_score=consistency_score,
                quality_score=quality_score,
                web_detection=web_detection,
                duplicate_count=len(duplicate_listings),
                calibration=calibration,
            )
            
            # 17. Generate explanation
            explanation = self._generate_explanation(
                classification, quality_score, composite_ai, risk_level,
                web_detection=web_detection,
                gan_score=gan_score,
                diffusion_score=diffusion_score,
                tampering_score=tampering_score,
                duplicate_count=len(duplicate_listings),
            )
            
            result = ImageAnalysisResult(
                image_hash=image_hash,
                risk_level=risk_level,
                risk_score=risk_score,
                is_property_image=classification.is_property_related if classification else False,
                property_confidence=classification.property_confidence if classification else 0.0,
                quality_score=quality_score,
                ai_detection_score=composite_ai,
                indicators=all_indicators,
                classification=classification,
                metadata=metadata,
                explanation=explanation,
                web_detection=web_detection,
                gan_fingerprint_score=gan_score,
                diffusion_artifact_score=diffusion_score,
                sensor_noise_score=sensor_score,
                tampering_score=tampering_score,
                content_consistency_score=consistency_score,
                duplicate_listings=duplicate_listings,
                perceptual_hash=phash,
                deep_exif=deep_exif,
                confidence_calibration=calibration,
                explainability=explainability,
            )
            
            # Store in cache
            self._cache_put(full_sha256, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return ImageAnalysisResult(
                image_hash=hashlib.sha256(image_data).hexdigest()[:16],
                risk_level=ImageRiskLevel.UNCERTAIN,
                risk_score=0.5,
                is_property_image=False,
                property_confidence=0.0,
                quality_score=0.5,
                ai_detection_score=0.0,
                indicators=[{
                    "code": "ANALYSIS_FAILED",
                    "severity": 2,
                    "description": f"Image analysis could not be completed: {str(e)[:50]}",
                    "evidence": []
                }],
                classification=None,
                metadata={},
                explanation="Image could not be fully analyzed. Manual review recommended.",
                web_detection=None
            )
    
    async def _classify_image(self, image: Image.Image) -> Optional[ImageClassificationResult]:
        """Classify image using pretrained CNN"""
        self._ensure_model_loaded()
        
        if self._model is None:
            return None
        
        return await asyncio.to_thread(self._classify_image_sync, image)
    
    def _classify_image_sync(self, image: Image.Image) -> Optional[ImageClassificationResult]:
        """Synchronous CNN classification — called via asyncio.to_thread"""
        try:
            import torch
            
            # Preprocess image
            input_tensor = self._transforms(image).unsqueeze(0).to(self._device)
            
            # Run inference
            with torch.no_grad():
                outputs = self._model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 15 predictions for broader analysis
            top_k = min(15, probabilities.shape[0])
            topk_prob, topk_idx = torch.topk(probabilities, top_k)
            
            top_classes = []
            property_score = 0.0
            suspicious_score = 0.0
            risk_indicators = []
            
            for i, (prob, idx) in enumerate(zip(topk_prob.cpu().numpy(), topk_idx.cpu().numpy())):
                idx = int(idx)
                prob = float(prob)
                
                # Get class name (real ImageNet label when available)
                class_name = self.class_labels.get(idx)
                if not class_name:
                    class_name = f"imagenet_class_{idx}"
                
                top_classes.append({
                    "rank": i + 1,
                    "class_id": idx,
                    "class_name": class_name,
                    "label": class_name,
                    "probability": round(prob, 4),
                    "confidence": round(prob, 4)
                })
                
                # Check if property-related (weight decays for lower ranks)
                if idx in PROPERTY_RELATED_CLASSES:
                    weight = PROPERTY_RELATED_CLASSES[idx][1]
                    # Higher ranks contribute more (rank 1 = full, rank 15 = 30%)
                    rank_factor = max(0.3, 1.0 - (i * 0.05))
                    property_score += prob * weight * rank_factor
                
                # Check for suspicious classes
                if idx in SUSPICIOUS_CLASSES:
                    weight = abs(SUSPICIOUS_CLASSES[idx][1])
                    suspicious_score += prob * weight
                    if prob > 0.15:  # Lower threshold — flag earlier
                        severity = 4 if prob > 0.5 else (3 if prob > 0.3 else 2)
                        risk_indicators.append({
                            "code": "NON_PROPERTY_CONTENT",
                            "severity": severity,
                            "description": f"Image appears to show {class_name} rather than property",
                            "evidence": [f"Classification confidence: {prob:.1%}"]
                        })
            
            # Determine if property-related using multi-signal approach
            # Method 1: Accumulated weighted score
            score_match = property_score > 0.08
            # Method 2: Top-1 or top-2 is a property class
            top2_match = any(
                cls["class_id"] in PROPERTY_RELATED_CLASSES 
                for cls in top_classes[:2]
            )
            # Method 3: Multiple property classes in top 10
            property_in_top10 = sum(
                1 for cls in top_classes[:10]
                if cls["class_id"] in PROPERTY_RELATED_CLASSES
            )
            multi_match = property_in_top10 >= 2
            # Method 4: No strong suspicious signal
            no_strong_suspicious = suspicious_score < 0.3
            
            is_property = (score_match or top2_match or multi_match) and no_strong_suspicious
            
            return ImageClassificationResult(
                is_property_related=is_property,
                property_confidence=min(property_score, 1.0),
                top_classes=top_classes,
                risk_indicators=risk_indicators
            )
            
        except Exception as e:
            logger.error(f"CNN classification failed: {e}")
            return None
    
    def _analyze_quality(self, image: Image.Image) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze image quality for authenticity indicators"""
        indicators = []
        quality_score = 1.0
        
        width, height = image.size
        
        # Check resolution
        if width < 400 or height < 400:
            quality_score -= 0.3
            indicators.append({
                "code": "LOW_RESOLUTION",
                "severity": 2,
                "description": f"Image has low resolution ({width}x{height})",
                "evidence": ["Low resolution can hide details or indicate stolen/cropped images"]
            })
        elif width >= 1920 and height >= 1080:
            quality_score += 0.1  # High quality is a good sign
        
        # Check aspect ratio (unusual ratios may indicate cropping)
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 3.0:
            quality_score -= 0.1
            indicators.append({
                "code": "UNUSUAL_ASPECT_RATIO",
                "severity": 1,
                "description": f"Unusual image aspect ratio ({aspect_ratio:.2f})",
                "evidence": ["May indicate heavy cropping"]
            })
        
        # Analyze color distribution
        try:
            img_array = np.array(image)
            
            # Check for color uniformity (suspicious if too uniform)
            color_std = np.std(img_array)
            if color_std < 20:
                quality_score -= 0.2
                indicators.append({
                    "code": "LOW_COLOR_VARIANCE",
                    "severity": 2,
                    "description": "Image has unusually uniform colors",
                    "evidence": ["May indicate synthetic/placeholder image"]
                })
            
            # Check for pure black/white images
            if np.mean(img_array) < 10 or np.mean(img_array) > 245:
                quality_score -= 0.3
                indicators.append({
                    "code": "EXTREME_EXPOSURE",
                    "severity": 3,
                    "description": "Image is extremely dark or bright",
                    "evidence": ["May be a placeholder or corrupted image"]
                })
            
            # Sharpness analysis using Laplacian variance
            gray = np.mean(img_array, axis=2)
            try:
                from scipy import ndimage
                laplacian = ndimage.laplace(gray)
                sharpness = np.var(laplacian)
                
                if sharpness < 5:
                    quality_score -= 0.15
                    indicators.append({
                        "code": "VERY_BLURRY",
                        "severity": 2,
                        "description": f"Image is very blurry (sharpness: {sharpness:.1f})",
                        "evidence": ["Blurry images may hide details or be low-effort fakes"]
                    })
                elif sharpness < 15:
                    quality_score -= 0.05
                elif sharpness > 200:
                    quality_score += 0.05  # Sharp image bonus
            except ImportError:
                pass
            
            # Contrast analysis 
            try:
                luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
                p5, p95 = np.percentile(luminance, [5, 95])
                contrast_range = p95 - p5
                
                if contrast_range < 30:
                    quality_score -= 0.1
                    indicators.append({
                        "code": "LOW_CONTRAST",
                        "severity": 1,
                        "description": f"Image has very low contrast (range: {contrast_range:.0f})",
                        "evidence": ["Low contrast may indicate washed-out or synthetic image"]
                    })
                elif contrast_range > 180:
                    quality_score += 0.05  # Good contrast
            except Exception:
                pass
            
        except Exception:
            pass
        
        return max(0.0, min(1.0, quality_score)), indicators
    
    def _detect_ai_generation(self, image: Image.Image) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Detect if image might be AI-generated using multiple forensic signals.
        
        AI-generated images (DALL-E, Midjourney, Stable Diffusion) exhibit:
        - Unnaturally smooth noise residuals (lack of sensor noise)
        - Unusual frequency spectrum shape (missing high-frequency detail)
        - Abnormal local patch statistics
        - Missing or fake EXIF data
        - Overly uniform saturation distributions
        - Suspicious color channel relationships
        """
        indicators = []
        ai_score = 0.0
        evidence_details = []
        
        try:
            img_array = np.array(image, dtype=np.float64)
            gray = np.mean(img_array, axis=2)
            h, w = gray.shape
            
            # Skip very small images — not enough data
            if h < 64 or w < 64:
                return 0.0, []
            
            # ------------------------------------------------------------------
            # 1. NOISE RESIDUAL ANALYSIS
            # Real camera photos have sensor noise. AI images are too clean.
            # Extract noise by subtracting a blurred version, then measure variance.
            # ------------------------------------------------------------------
            try:
                from scipy import ndimage
                
                # Multi-scale noise analysis
                for sigma, label in [(1.0, "fine"), (2.0, "medium")]:
                    blurred = ndimage.gaussian_filter(gray, sigma=sigma)
                    residual = gray - blurred
                    noise_var = np.var(residual)
                    noise_std = np.std(residual)
                    
                    if sigma == 1.0:
                        # Real photos: noise_std typically 3-15+
                        # AI images: noise_std typically 0.5-3
                        # Compressed web images can also be low (2-4)
                        if noise_std < 1.5:
                            ai_score += 0.12
                            evidence_details.append(
                                f"Very low fine noise (std={noise_std:.2f}, expected >3 for real photos)")
                        elif noise_std < 2.5:
                            ai_score += 0.05
                            evidence_details.append(
                                f"Low fine noise (std={noise_std:.2f})")
                    
                    if sigma == 2.0:
                        if noise_std < 2.0:
                            ai_score += 0.10
                            evidence_details.append(
                                f"Low medium-scale noise (std={noise_std:.2f})")
                
                # Edge response analysis — AI images have unnaturally clean edges
                sobel_x = ndimage.sobel(gray, axis=1)
                sobel_y = ndimage.sobel(gray, axis=0)
                edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
                
                # Check noise in edge regions vs flat regions
                edge_mask = edge_mag > np.percentile(edge_mag, 75)
                flat_mask = edge_mag < np.percentile(edge_mag, 25)
                
                if np.sum(flat_mask) > 100 and np.sum(edge_mask) > 100:
                    # In real photos, flat regions still have sensor noise
                    flat_noise = np.std(gray[flat_mask] - ndimage.gaussian_filter(gray, 1.0)[flat_mask])
                    if flat_noise < 1.5:
                        ai_score += 0.10
                        evidence_details.append(
                            f"Flat regions too smooth (noise={flat_noise:.2f})")

            except ImportError:
                pass
            
            # ------------------------------------------------------------------
            # 2. FREQUENCY SPECTRUM ANALYSIS
            # Real photos have gradual high-frequency falloff following power law.
            # AI images often have spectral gaps or unusual rolloff patterns.
            # ------------------------------------------------------------------
            try:
                # Compute centered FFT magnitude
                fft = np.fft.fft2(gray)
                fft_shift = np.fft.fftshift(fft)
                magnitude = np.log1p(np.abs(fft_shift))
                
                cy, cx = h // 2, w // 2
                max_r = min(cy, cx) - 1
                
                if max_r > 20:
                    # Compute radial power spectrum
                    radii = np.arange(5, max_r, 2)
                    power_profile = []
                    for r in radii:
                        y_coords, x_coords = np.ogrid[-cy:h-cy, -cx:w-cx]
                        ring = (y_coords**2 + x_coords**2 >= (r-1)**2) & \
                               (y_coords**2 + x_coords**2 < (r+1)**2)
                        if np.sum(ring) > 0:
                            power_profile.append(np.mean(magnitude[ring]))
                    
                    if len(power_profile) > 10:
                        power_profile = np.array(power_profile)
                        
                        # Check high-frequency energy ratio
                        mid = len(power_profile) // 2
                        low_freq_power = np.mean(power_profile[:mid])
                        high_freq_power = np.mean(power_profile[mid:])
                        
                        if low_freq_power > 0:
                            hf_ratio = high_freq_power / low_freq_power
                            # Real photos: ratio typically 0.3-0.7
                            # AI images: ratio often < 0.2 (lacking fine detail)
                            if hf_ratio < 0.15:
                                ai_score += 0.12
                                evidence_details.append(
                                    f"Very low high-frequency content (ratio={hf_ratio:.3f})")
                            elif hf_ratio < 0.25:
                                ai_score += 0.06
                                evidence_details.append(
                                    f"Low high-frequency content (ratio={hf_ratio:.3f})")
                        
                        # Check spectrum smoothness — AI tends to be unnaturally smooth
                        diffs = np.diff(power_profile)
                        spectrum_roughness = np.std(diffs)
                        if spectrum_roughness < 0.05:
                            ai_score += 0.08
                            evidence_details.append(
                                f"Unnaturally smooth frequency spectrum")
                        
            except Exception:
                pass
            
            # ------------------------------------------------------------------
            # 3. LOCAL PATCH STATISTICS
            # AI images have unusually consistent statistics across patches.
            # Real photos vary due to different surfaces, lighting, etc.
            # ------------------------------------------------------------------
            try:
                patch_size = 32
                patches_y = h // patch_size
                patches_x = w // patch_size
                
                if patches_y >= 4 and patches_x >= 4:
                    patch_stds = []
                    patch_means = []
                    for py in range(patches_y):
                        for px in range(patches_x):
                            patch = gray[py*patch_size:(py+1)*patch_size,
                                        px*patch_size:(px+1)*patch_size]
                            patch_stds.append(np.std(patch))
                            patch_means.append(np.mean(patch))
                    
                    # Coefficient of variation of patch standard deviations
                    patch_stds = np.array(patch_stds)
                    if np.mean(patch_stds) > 0:
                        cv_std = np.std(patch_stds) / np.mean(patch_stds)
                        # Real photos: cv_std typically 0.5-2.0
                        # AI images: cv_std sometimes abnormally low (< 0.3)
                        if cv_std < 0.25:
                            ai_score += 0.08
                            evidence_details.append(
                                f"Suspiciously uniform texture across image (cv={cv_std:.3f})")
            except Exception:
                pass
            
            # ------------------------------------------------------------------
            # 4. COLOR CHANNEL ANALYSIS
            # AI images often produce unusual inter-channel relationships
            # ------------------------------------------------------------------
            try:
                r = img_array[:,:,0]
                g = img_array[:,:,1]
                b = img_array[:,:,2]
                
                # Check channel correlations
                r_flat, g_flat, b_flat = r.flatten(), g.flatten(), b.flatten()
                
                # Subsample for speed
                n = min(len(r_flat), 50000)
                step = max(1, len(r_flat) // n)
                r_s, g_s, b_s = r_flat[::step], g_flat[::step], b_flat[::step]
                
                rg_corr = np.corrcoef(r_s, g_s)[0, 1]
                rb_corr = np.corrcoef(r_s, b_s)[0, 1]
                gb_corr = np.corrcoef(g_s, b_s)[0, 1]
                
                # Extremely high correlation across ALL channels is suspicious
                if (not np.isnan(rg_corr) and not np.isnan(rb_corr) and
                    not np.isnan(gb_corr)):
                    avg_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3
                    if avg_corr > 0.97:
                        ai_score += 0.08
                        evidence_details.append(
                            f"Abnormally high color channel correlation ({avg_corr:.3f})")
                
                # Check saturation uniformity
                # AI images often have narrow saturation bands
                hsv_img = np.array(image.convert('HSV'))
                saturation = hsv_img[:,:,1].flatten()
                sat_std = np.std(saturation)
                sat_mean = np.mean(saturation)
                
                if sat_mean > 20:  # Not grayscale
                    sat_cv = sat_std / sat_mean if sat_mean > 0 else 0
                    # Real photos: wide saturation variation
                    # AI: often very uniform saturation
                    if sat_cv < 0.3:
                        ai_score += 0.06
                        evidence_details.append(
                            f"Unusually uniform saturation (cv={sat_cv:.3f})")
                
            except Exception:
                pass
            
            # ------------------------------------------------------------------
            # 5. EXIF / METADATA ANALYSIS
            # Real camera photos have EXIF data with camera model, settings, etc.
            # AI images typically have NO meaningful EXIF.
            # ------------------------------------------------------------------
            try:
                exif_data = image._getexif()
                has_camera_info = False
                has_software_hint = False
                
                if exif_data:
                    from PIL.ExifTags import TAGS
                    exif_tags = {TAGS.get(k, k): v for k, v in exif_data.items()}
                    
                    # Camera make/model = strong indicator of real photo
                    if exif_tags.get("Make") or exif_tags.get("Model"):
                        has_camera_info = True
                    
                    # Software field can indicate AI tools
                    software = str(exif_tags.get("Software", "")).lower()
                    ai_software_hints = [
                        "dall-e", "midjourney", "stable diffusion", "novelai",
                        "craiyon", "imagen", "firefly", "leonardo",
                        "comfyui", "automatic1111", "invoke"
                    ]
                    for hint in ai_software_hints:
                        if hint in software:
                            ai_score += 0.35
                            has_software_hint = True
                            evidence_details.append(
                                f"EXIF Software field references AI tool: {software}")
                            break
                    
                    if has_camera_info and not has_software_hint:
                        # Genuine camera metadata reduces AI suspicion
                        ai_score = max(0, ai_score - 0.10)
                        
                else:
                    # No EXIF at all — mild indicator 
                    # (many legitimate web images also lack EXIF)
                    ai_score += 0.04
                    evidence_details.append("No EXIF metadata found")
                    
            except Exception:
                pass
            
            # ------------------------------------------------------------------
            # 6. JPEG COMPRESSION ARTIFACT ANALYSIS
            # Real photos saved as JPEG have specific 8x8 block artifacts.
            # AI images re-saved as JPEG have different artifact patterns.
            # ------------------------------------------------------------------
            try:
                if image.format == 'JPEG' or (hasattr(image, 'info') and 'jfif' in str(image.info).lower()):
                    # Check JPEG quantization tables
                    quantization = getattr(image, 'quantization', None)
                    if quantization:
                        # Real cameras use standard quantization tables
                        # Re-encoded AI images may have unusual tables
                        for table_id, table in quantization.items():
                            q_values = list(table.values()) if isinstance(table, dict) else list(table)
                            if len(q_values) == 64:
                                # Check if quantization is unusually fine (low values)
                                # This can indicate the image was generated at high quality
                                avg_q = np.mean(q_values)
                                if avg_q < 3:
                                    ai_score += 0.04
                                    evidence_details.append(
                                        f"Unusually fine JPEG quantization (avg={avg_q:.1f})")
            except Exception:
                pass
            
            # ------------------------------------------------------------------
            # Build indicators from evidence
            # ------------------------------------------------------------------
            if ai_score > 0.06:
                severity = 1 if ai_score < 0.2 else (2 if ai_score < 0.4 else 3)
                if ai_score >= 0.4:
                    desc = "Strong indicators of AI-generated content detected"
                elif ai_score >= 0.25:
                    desc = "Moderate indicators of AI-generated content detected"
                elif ai_score >= 0.15:
                    desc = "Some indicators of possible AI-generated content"
                else:
                    desc = "Minor AI-generation indicators noted"
                
                indicators.append({
                    "code": "AI_GENERATED_SUSPECTED",
                    "severity": severity,
                    "description": desc,
                    "evidence": evidence_details[:6]  # Top 6 pieces of evidence
                })
            
            # Add specific high-confidence indicators
            if ai_score >= 0.35:
                indicators.append({
                    "code": "AI_HIGH_CONFIDENCE",
                    "severity": 4,
                    "description": "Image is very likely AI-generated",
                    "evidence": [
                        f"AI detection score: {ai_score:.0%}",
                        "Multiple forensic signals indicate non-authentic origin"
                    ]
                })
                    
        except ImportError:
            logger.debug("scipy not available for AI detection")
        except Exception as e:
            logger.debug(f"AI detection analysis error: {e}")
        
        return min(ai_score, 1.0), indicators
    
    def _extract_metadata(
        self,
        image: Image.Image,
        filename: Optional[str]
    ) -> Dict[str, Any]:
        """Extract image metadata"""
        metadata = {
            "width": image.size[0],
            "height": image.size[1],
            "format": image.format,
            "mode": image.mode
        }
        
        if filename:
            metadata["filename"] = filename
        
        # Try to extract EXIF data
        try:
            from PIL.ExifTags import TAGS
            exif_data = image._getexif()
            if exif_data:
                metadata["has_exif"] = True
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag in ["Make", "Model", "DateTime", "Software"]:
                        metadata[tag.lower()] = str(value)
            else:
                metadata["has_exif"] = False
        except Exception:
            metadata["has_exif"] = False
        
        return metadata
    
    # ------------------------------------------------------------------
    #  5. GAN FINGERPRINT DETECTION
    #  GANs (StyleGAN, ProGAN) leave characteristic spectral artifacts:
    #  - Periodic spectral peaks from transposed convolutions
    #  - Checkerboard artifacts in upsampling layers
    #  - Abnormal high-frequency spectral symmetry
    # ------------------------------------------------------------------
    def _detect_gan_fingerprints(self, image: Image.Image) -> Tuple[float, List[Dict[str, Any]]]:
        """Detect GAN-specific artifacts in frequency domain"""
        indicators = []
        gan_score = 0.0
        evidence = []
        
        try:
            img_array = np.array(image, dtype=np.float64)
            gray = np.mean(img_array, axis=2)
            h, w = gray.shape
            
            if h < 64 or w < 64:
                return 0.0, []
            
            from scipy import ndimage
            
            # --- Spectral peak detection ---
            # GANs using transposed convolutions produce periodic peaks
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            log_mag = np.log1p(magnitude)
            
            cy, cx = h // 2, w // 2
            
            # Mask out DC component (center 5x5)
            dc_mask = np.ones_like(log_mag)
            dc_mask[cy-2:cy+3, cx-2:cx+3] = 0
            masked_mag = log_mag * dc_mask
            
            # Find spectral peaks — GAN artifacts appear as isolated bright spots
            mean_mag = np.mean(masked_mag[masked_mag > 0])
            std_mag = np.std(masked_mag[masked_mag > 0])
            peak_threshold = mean_mag + 4.0 * std_mag
            peaks = np.sum(masked_mag > peak_threshold)
            
            if peaks > 8:
                gan_score += 0.15
                evidence.append(f"Spectral peaks detected ({peaks} anomalous frequencies)")
            elif peaks > 4:
                gan_score += 0.08
                evidence.append(f"Some spectral peaks ({peaks})")
            
            # --- Checkerboard artifact detection ---
            # Transposed convolutions produce 2x2 repeating patterns
            # Detect by checking even/odd pixel correlation difference
            if h >= 128 and w >= 128:
                even_rows = gray[::2, :]
                odd_rows = gray[1::2, :]
                min_rows = min(even_rows.shape[0], odd_rows.shape[0])
                
                if min_rows > 10:
                    row_diff = np.mean(np.abs(even_rows[:min_rows] - odd_rows[:min_rows]))
                    
                    even_cols = gray[:, ::2]
                    odd_cols = gray[:, 1::2]
                    min_cols = min(even_cols.shape[1], odd_cols.shape[1])
                    col_diff = np.mean(np.abs(even_cols[:, :min_cols] - odd_cols[:, :min_cols]))
                    
                    overall_diff = np.mean(np.abs(np.diff(gray, axis=0)))
                    
                    if overall_diff > 0:
                        checkerboard_ratio = (row_diff + col_diff) / (2 * overall_diff)
                        # GAN checkerboard: ratio tends toward 1.0 (even/odd nearly identical)
                        # Real photos: ratio typically 0.7-0.95
                        if checkerboard_ratio > 0.98:
                            gan_score += 0.12
                            evidence.append(f"Checkerboard pattern detected (ratio={checkerboard_ratio:.3f})")
                        elif checkerboard_ratio > 0.96:
                            gan_score += 0.06
                            evidence.append(f"Mild checkerboard pattern (ratio={checkerboard_ratio:.3f})")
            
            # --- Spectral symmetry analysis ---
            # GAN outputs often have unnaturally symmetric frequency spectra
            if h > 64 and w > 64:
                left_half = log_mag[:, :cx]
                right_half = np.fliplr(log_mag[:, cx:])
                min_w = min(left_half.shape[1], right_half.shape[1])
                if min_w > 10:
                    symmetry = np.corrcoef(
                        left_half[:, :min_w].flatten(),
                        right_half[:, :min_w].flatten()
                    )[0, 1]
                    if not np.isnan(symmetry) and symmetry > 0.98:
                        gan_score += 0.08
                        evidence.append(f"Abnormally symmetric frequency spectrum ({symmetry:.3f})")
            
            if gan_score > 0.05:
                severity = 3 if gan_score >= 0.25 else (2 if gan_score >= 0.12 else 1)
                indicators.append({
                    "code": "GAN_FINGERPRINT",
                    "severity": severity,
                    "description": "GAN-generated image fingerprints detected",
                    "evidence": evidence[:4]
                })
            
        except ImportError:
            logger.debug("scipy not available for GAN detection")
        except Exception as e:
            logger.debug(f"GAN fingerprint analysis error: {e}")
        
        return min(gan_score, 1.0), indicators
    
    # ------------------------------------------------------------------
    #  6. DIFFUSION ARTIFACT DETECTION
    #  Diffusion models (Stable Diffusion, DALL-E 3, Midjourney) produce:
    #  - Banding artifacts in gradients (denoising steps)
    #  - Tiling seams from latent space tiling
    #  - Attention mask leakage patterns
    #  - Characteristic texture at specific spatial scales
    # ------------------------------------------------------------------
    def _detect_diffusion_artifacts(self, image: Image.Image) -> Tuple[float, List[Dict[str, Any]]]:
        """Detect diffusion model artifacts"""
        indicators = []
        diff_score = 0.0
        evidence = []
        
        try:
            img_array = np.array(image, dtype=np.float64)
            gray = np.mean(img_array, axis=2)
            h, w = gray.shape
            
            if h < 64 or w < 64:
                return 0.0, []
            
            from scipy import ndimage
            
            # --- Gradient banding detection ---
            # Diffusion models produce subtle banding in smooth gradient regions
            # Detect by looking at gradient histogram in flat regions
            sobel_mag = np.sqrt(
                ndimage.sobel(gray, axis=0)**2 + ndimage.sobel(gray, axis=1)**2
            )
            flat_mask = sobel_mag < np.percentile(sobel_mag, 30)
            
            if np.sum(flat_mask) > 500:
                flat_gradients = sobel_mag[flat_mask]
                # Banding creates multi-modal gradient distribution in flat areas
                hist, bin_edges = np.histogram(flat_gradients, bins=50)
                hist_norm = hist / (np.sum(hist) + 1e-10)
                
                # Count number of local maxima in gradient histogram
                local_maxima = 0
                for i in range(1, len(hist_norm) - 1):
                    if hist_norm[i] > hist_norm[i-1] and hist_norm[i] > hist_norm[i+1]:
                        if hist_norm[i] > 0.02:  # Significant peak
                            local_maxima += 1
                
                if local_maxima >= 4:
                    diff_score += 0.10
                    evidence.append(f"Gradient banding detected ({local_maxima} modes in flat regions)")
                elif local_maxima >= 3:
                    diff_score += 0.05
                    evidence.append(f"Mild gradient banding ({local_maxima} modes)")
            
            # --- Tiling seam detection ---
            # Latent-space diffusion at 64x64 upscaled to 512x512 leaves seams
            # Check for periodic intensity drops at regular intervals
            tile_sizes = [64, 128, 256]
            for ts in tile_sizes:
                if w >= ts * 2 and h >= ts * 2:
                    # Check vertical seams
                    col_means = np.mean(gray, axis=0)
                    seam_positions = np.arange(ts, w - ts, ts)
                    if len(seam_positions) > 0:
                        seam_diffs = []
                        for sp in seam_positions:
                            left = np.mean(col_means[max(0, sp-3):sp])
                            right = np.mean(col_means[sp:min(w, sp+3)])
                            seam_diffs.append(abs(left - right))
                        
                        avg_seam_diff = np.mean(seam_diffs)
                        overall_col_diff = np.mean(np.abs(np.diff(col_means)))
                        
                        if overall_col_diff > 0:
                            seam_ratio = avg_seam_diff / overall_col_diff
                            if seam_ratio > 1.8:
                                diff_score += 0.08
                                evidence.append(f"Tiling seams at {ts}px intervals (ratio={seam_ratio:.2f})")
                                break
            
            # --- Attention mask leakage ---
            # Cross-attention in diffusion models can leave boundary artifacts
            # around semantic regions (objects vs background)
            # Detect by analyzing edge sharpness distribution
            laplacian = ndimage.laplace(gray)
            edge_sharpness = np.abs(laplacian)
            
            # Compare sharpness in edge regions vs overall
            p90 = np.percentile(edge_sharpness, 90)
            p10 = np.percentile(edge_sharpness, 10)
            
            if p10 > 0:
                sharpness_ratio = p90 / p10
                # Diffusion: very sharp boundaries with very smooth interiors
                if sharpness_ratio > 200:
                    diff_score += 0.08
                    evidence.append(f"Extreme edge/interior sharpness contrast (ratio={sharpness_ratio:.0f})")
                elif sharpness_ratio > 100:
                    diff_score += 0.04
                    evidence.append(f"High edge/interior contrast (ratio={sharpness_ratio:.0f})")
            
            # --- Texture scale consistency ---
            # Diffusion models process at fixed latent resolution then upscale
            # This creates unnaturally consistent texture detail at all scales
            patch_sizes = [16, 32, 64]
            texture_vars = []
            for ps in patch_sizes:
                if h >= ps * 3 and w >= ps * 3:
                    patches_stds = []
                    for py in range(0, h - ps, ps):
                        for px in range(0, w - ps, ps):
                            patch = gray[py:py+ps, px:px+ps]
                            # High-pass filter to measure texture
                            hp = patch - ndimage.gaussian_filter(patch, 2.0)
                            patches_stds.append(np.std(hp))
                    if patches_stds:
                        texture_vars.append(np.std(patches_stds) / (np.mean(patches_stds) + 1e-10))
            
            if len(texture_vars) >= 2:
                # Real photos: texture CV varies significantly across scales
                # Diffusion: CV is similar across scales
                cv_range = max(texture_vars) - min(texture_vars)
                if cv_range < 0.08:
                    diff_score += 0.07
                    evidence.append(f"Unnaturally consistent texture across scales (range={cv_range:.3f})")
            
            if diff_score > 0.05:
                severity = 3 if diff_score >= 0.2 else (2 if diff_score >= 0.1 else 1)
                indicators.append({
                    "code": "DIFFUSION_ARTIFACT",
                    "severity": severity,
                    "description": "Diffusion model artifacts detected",
                    "evidence": evidence[:4]
                })
        
        except ImportError:
            logger.debug("scipy not available for diffusion detection")
        except Exception as e:
            logger.debug(f"Diffusion artifact analysis error: {e}")
        
        return min(diff_score, 1.0), indicators
    
    # ------------------------------------------------------------------
    #  7. CAMERA SENSOR NOISE / PRNU ANALYSIS
    #  Real cameras have Photo Response Non-Uniformity (PRNU) —
    #  a unique noise pattern from manufacturing imperfections.
    #  AI images lack genuine sensor noise entirely.
    # ------------------------------------------------------------------
    def _analyze_sensor_noise(self, image: Image.Image) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze sensor noise patterns for camera fingerprint verification"""
        indicators = []
        sensor_score = 0.0  # Higher = more suspicious (lacks real sensor noise)
        evidence = []
        
        try:
            img_array = np.array(image, dtype=np.float64)
            gray = np.mean(img_array, axis=2)
            h, w = gray.shape
            
            if h < 128 or w < 128:
                return 0.0, []
            
            from scipy import ndimage
            
            # --- PRNU estimation ---
            # Extract noise residual using Wiener-like denoising
            denoised = ndimage.gaussian_filter(gray, sigma=3.0)
            noise_residual = gray - denoised
            
            # Real cameras: noise has spatial non-uniformity (PRNU fingerprint)
            # AI images: noise is either absent or uniformly random
            
            # Split image into quadrants and compare noise patterns
            qh, qw = h // 2, w // 2
            q1 = noise_residual[:qh, :qw]
            q2 = noise_residual[:qh, qw:qw*2]
            q3 = noise_residual[qh:qh*2, :qw]
            q4 = noise_residual[qh:qh*2, qw:qw*2]
            
            # In real photos, PRNU is consistent — same pixel positions have
            # correlated noise across different brightness levels
            q_stds = [np.std(q) for q in [q1, q2, q3, q4]]
            
            # Check noise uniformity across quadrants
            if np.mean(q_stds) > 0:
                noise_cv = np.std(q_stds) / np.mean(q_stds)
                
                # Real photos: moderate CV (0.1-0.5) from scene-dependent noise
                # AI images: very low CV (< 0.05) — no real sensor pattern
                # OR very high CV (> 0.8) — artificial noise added unevenly
                if noise_cv < 0.03:
                    sensor_score += 0.12
                    evidence.append(
                        f"No sensor noise non-uniformity detected (cv={noise_cv:.4f})")
                elif noise_cv > 0.8:
                    sensor_score += 0.06
                    evidence.append(
                        f"Artificially uneven noise distribution (cv={noise_cv:.3f})")
            
            # --- Noise autocorrelation ---
            # Real sensor noise has spatial correlation from sensor readout
            # AI noise is typically i.i.d. (independent identically distributed)
            center_patch = noise_residual[qh-32:qh+32, qw-32:qw+32]
            if center_patch.shape == (64, 64):
                # Compute autocorrelation at lag 1
                auto_h = np.corrcoef(center_patch[:, :-1].flatten(),
                                      center_patch[:, 1:].flatten())[0, 1]
                auto_v = np.corrcoef(center_patch[:-1, :].flatten(),
                                      center_patch[1:, :].flatten())[0, 1]
                
                if not np.isnan(auto_h) and not np.isnan(auto_v):
                    avg_auto = (abs(auto_h) + abs(auto_v)) / 2
                    # Real sensor noise: autocorrelation typically 0.1-0.4
                    # i.i.d. noise: autocorrelation near 0
                    # Over-smoothed AI: autocorrelation near 0 or very high
                    if avg_auto < 0.02:
                        sensor_score += 0.08
                        evidence.append(
                            f"Noise lacks spatial correlation (auto={avg_auto:.4f})")
                    elif avg_auto > 0.6:
                        sensor_score += 0.05
                        evidence.append(
                            f"Noise over-correlated — possible AI smoothing (auto={avg_auto:.3f})")
            
            # --- Per-channel noise independence ---
            # Real cameras: R/G/B noise somewhat correlated (from Bayer filter)
            # AI images: channels may be too independent or too correlated
            for ch_name, ch_idx in [("R", 0), ("G", 1), ("B", 2)]:
                ch = img_array[:, :, ch_idx]
                ch_denoised = ndimage.gaussian_filter(ch, sigma=3.0)
                ch_noise = ch - ch_denoised
                ch_noise_std = np.std(ch_noise)
                
                if ch_noise_std < 0.5:
                    sensor_score += 0.03
                    evidence.append(f"{ch_name} channel has near-zero noise (std={ch_noise_std:.3f})")
            
            if sensor_score > 0.05:
                severity = 3 if sensor_score >= 0.2 else (2 if sensor_score >= 0.1 else 1)
                indicators.append({
                    "code": "SENSOR_NOISE_ANOMALY",
                    "severity": severity,
                    "description": "Camera sensor noise pattern missing or abnormal",
                    "evidence": evidence[:4]
                })
        
        except ImportError:
            logger.debug("scipy not available for sensor noise analysis")
        except Exception as e:
            logger.debug(f"Sensor noise analysis error: {e}")
        
        return min(sensor_score, 1.0), indicators
    
    # ------------------------------------------------------------------
    #  8. DEEP EXIF ANALYSIS
    #  Goes beyond basic EXIF to check:
    #  - GPS coordinate plausibility
    #  - Date/time consistency and plausibility
    #  - Thumbnail vs main image mismatch
    #  - Compression history (re-save chain)
    #  - Orientation chain consistency
    # ------------------------------------------------------------------
    def _deep_exif_analysis(
        self, image: Image.Image, filename: Optional[str]
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Deep EXIF metadata forensic analysis"""
        indicators = []
        deep_exif: Dict[str, Any] = {
            "has_exif": False,
            "has_gps": False,
            "has_thumbnail": False,
            "has_camera_info": False,
            "compression_history": [],
            "anomalies": [],
        }
        
        try:
            exif_data = image._getexif()
            if not exif_data:
                deep_exif["anomalies"].append("No EXIF metadata present")
                return deep_exif, indicators
            
            from PIL.ExifTags import TAGS, GPSTAGS
            exif_tags = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, str(tag_id))
                exif_tags[tag] = value
            
            deep_exif["has_exif"] = True
            
            # --- Camera info ---
            make = str(exif_tags.get("Make", "")).strip()
            model = str(exif_tags.get("Model", "")).strip()
            if make or model:
                deep_exif["has_camera_info"] = True
                deep_exif["camera"] = f"{make} {model}".strip()
            
            # --- Date/Time plausibility ---
            date_str = str(exif_tags.get("DateTime", ""))
            date_orig = str(exif_tags.get("DateTimeOriginal", ""))
            date_digi = str(exif_tags.get("DateTimeDigitized", ""))
            
            dates = {}
            for label, ds in [("DateTime", date_str), ("Original", date_orig), ("Digitized", date_digi)]:
                if ds and ds != "None" and len(ds) >= 10:
                    try:
                        dt = datetime.strptime(ds[:19], "%Y:%m:%d %H:%M:%S")
                        dates[label] = dt
                        # Future date check
                        if dt > datetime.now():
                            deep_exif["anomalies"].append(f"{label} is in the future: {ds}")
                            indicators.append({
                                "code": "EXIF_FUTURE_DATE",
                                "severity": 3,
                                "description": f"EXIF {label} is set to a future date",
                                "evidence": [ds]
                            })
                        # Very old date (before digital cameras existed)
                        if dt.year < 1990:
                            deep_exif["anomalies"].append(f"{label} predates digital cameras: {ds}")
                    except ValueError:
                        deep_exif["anomalies"].append(f"Malformed {label}: {ds}")
            
            # Check for date inconsistency
            if "Original" in dates and "Digitized" in dates:
                diff = abs((dates["Original"] - dates["Digitized"]).total_seconds())
                if diff > 86400:  # More than 1 day apart
                    deep_exif["anomalies"].append(
                        f"Original and Digitized dates differ by {diff/3600:.0f} hours")
                    indicators.append({
                        "code": "EXIF_DATE_MISMATCH",
                        "severity": 2,
                        "description": "EXIF capture dates are inconsistent",
                        "evidence": [f"Original: {date_orig}", f"Digitized: {date_digi}"]
                    })
            
            # --- GPS plausibility ---
            gps_info = exif_tags.get("GPSInfo")
            if gps_info:
                deep_exif["has_gps"] = True
                try:
                    gps_tags = {}
                    for k, v in gps_info.items():
                        gps_tags[GPSTAGS.get(k, k)] = v
                    
                    lat = gps_tags.get("GPSLatitude")
                    lat_ref = gps_tags.get("GPSLatitudeRef", "N")
                    lon = gps_tags.get("GPSLongitude")
                    lon_ref = gps_tags.get("GPSLongitudeRef", "W")
                    
                    if lat and lon:
                        def _to_float(val):
                            """Convert IFDRational / tuple / number to float safely"""
                            if isinstance(val, tuple) and len(val) == 2:
                                # (numerator, denominator) form
                                return float(val[0]) / float(val[1]) if val[1] else 0.0
                            return float(val)
                        
                        def _dms_to_dd(dms, ref):
                            d = _to_float(dms[0])
                            m = _to_float(dms[1])
                            s = _to_float(dms[2])
                            dd = d + m / 60 + s / 3600
                            if ref in ("S", "W"):
                                dd = -dd
                            return dd
                        
                        lat_dd = _dms_to_dd(lat, lat_ref)
                        lon_dd = _dms_to_dd(lon, lon_ref)
                        deep_exif["gps"] = {"lat": round(lat_dd, 6), "lon": round(lon_dd, 6)}
                        
                        # Check if coordinates are at (0,0) — null island
                        if abs(lat_dd) < 0.01 and abs(lon_dd) < 0.01:
                            deep_exif["anomalies"].append("GPS coordinates at (0,0) — likely fake")
                            indicators.append({
                                "code": "EXIF_GPS_NULL_ISLAND",
                                "severity": 3,
                                "description": "GPS coordinates point to Null Island (0°, 0°)",
                                "evidence": ["Coordinates (0,0) suggest fabricated EXIF"]
                            })
                except Exception:
                    pass
            
            # --- Thumbnail mismatch ---
            # Some manipulated images update the main image but forget the thumbnail
            try:
                if hasattr(image, '_getexif') and image.info.get("exif"):
                    # Check for EXIF thumbnail
                    thumb_offset = exif_tags.get("JPEGInterchangeFormat")
                    thumb_length = exif_tags.get("JPEGInterchangeFormatLength")
                    if thumb_offset and thumb_length:
                        deep_exif["has_thumbnail"] = True
                        # Compare main image orientation with thumbnail
                        orientation = exif_tags.get("Orientation", 1)
                        if orientation not in [1, 2, 3, 4, 5, 6, 7, 8]:
                            deep_exif["anomalies"].append(
                                f"Invalid EXIF orientation value: {orientation}")
            except Exception:
                pass
            
            # --- Software / editing chain ---
            software = str(exif_tags.get("Software", "")).strip()
            if software:
                deep_exif["software"] = software
                # Check for photo editors that might indicate manipulation
                edit_software = [
                    "photoshop", "gimp", "lightroom", "affinity",
                    "snapseed", "pixelmator", "paint.net", "canva"
                ]
                for es in edit_software:
                    if es in software.lower():
                        deep_exif["anomalies"].append(f"Edited with: {software}")
                        indicators.append({
                            "code": "EXIF_EDITING_SOFTWARE",
                            "severity": 1,
                            "description": f"Image was processed with editing software",
                            "evidence": [f"Software: {software}"]
                        })
                        break
            
            # --- Compression quality ---
            # JPEG quality from quantization tables
            quantization = getattr(image, 'quantization', None)
            if quantization:
                for table_id, table in quantization.items():
                    q_values = list(table.values()) if isinstance(table, dict) else list(table)
                    if len(q_values) == 64:
                        avg_q = float(np.mean(q_values))
                        deep_exif["compression_history"].append({
                            "table_id": table_id,
                            "avg_quantization": round(avg_q, 1),
                            "estimated_quality": round(max(0, min(100, 100 - avg_q * 1.5)), 0)
                        })
            
        except Exception as e:
            logger.debug(f"Deep EXIF analysis error: {e}")
        
        return deep_exif, indicators
    
    # ------------------------------------------------------------------
    #  9. TAMPERING DETECTION
    #  Detects image manipulation via:
    #  - Error Level Analysis (ELA) — re-saved regions have different error
    #  - Double JPEG compression detection
    #  - Copy-move forgery detection via block matching
    #  - Edge inconsistency at splice boundaries
    # ------------------------------------------------------------------
    def _detect_tampering(
        self, image: Image.Image, image_data: bytes
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Detect image tampering, splicing, and manipulation"""
        indicators = []
        tampering_score = 0.0
        evidence = []
        
        try:
            img_array = np.array(image, dtype=np.float64)
            gray = np.mean(img_array, axis=2)
            h, w = gray.shape
            
            if h < 64 or w < 64:
                return 0.0, []
            
            from scipy import ndimage
            
            # --- Error Level Analysis (ELA) ---
            # Re-save the image at a known quality and compare
            # Tampered regions show different error levels
            try:
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=90)
                buffer.seek(0)
                resaved = Image.open(buffer).convert('RGB')
                resaved_array = np.array(resaved).astype(np.float64)
                
                # Compute error level per pixel
                ela = np.abs(img_array - resaved_array)
                ela_gray = np.mean(ela, axis=2)
                
                # Analyze ELA distribution
                ela_mean = np.mean(ela_gray)
                ela_std = np.std(ela_gray)
                
                if ela_std > 0 and ela_mean > 0:
                    # High ELA variance indicates regions saved at different qualities
                    ela_cv = ela_std / ela_mean
                    
                    # Check for localized high-ELA regions (potential splices)
                    ela_thresh = ela_mean + 3 * ela_std
                    high_ela_ratio = np.sum(ela_gray > ela_thresh) / ela_gray.size
                    
                    if high_ela_ratio > 0.02 and high_ela_ratio < 0.3:
                        # Small localized regions with much higher error = likely splice
                        tampering_score += 0.12
                        evidence.append(
                            f"ELA detects {high_ela_ratio:.1%} of image with anomalous error levels")
                    
                    if ela_cv > 2.0:
                        tampering_score += 0.06
                        evidence.append(
                            f"High ELA variance (cv={ela_cv:.2f}) suggests mixed compression")
            except Exception:
                pass
            
            # --- Double JPEG compression detection ---
            # Images re-saved as JPEG show periodic artifacts in DCT histogram
            try:
                # Analyze 8x8 block boundaries
                if h >= 16 and w >= 16:
                    block_boundary_diffs = []
                    interior_diffs = []
                    
                    for y in range(8, h - 8, 8):
                        for x in range(8, w - 8, 1):
                            if x % 8 == 0:
                                block_boundary_diffs.append(abs(gray[y, x] - gray[y, x-1]))
                            elif x % 8 == 4:
                                interior_diffs.append(abs(gray[y, x] - gray[y, x-1]))
                    
                    if block_boundary_diffs and interior_diffs:
                        boundary_mean = np.mean(block_boundary_diffs)
                        interior_mean = np.mean(interior_diffs)
                        
                        if interior_mean > 0:
                            blocking_ratio = boundary_mean / interior_mean
                            # Double JPEG: block boundaries are more prominent
                            if blocking_ratio > 1.5:
                                tampering_score += 0.08
                                evidence.append(
                                    f"Double JPEG compression artifacts (blocking ratio={blocking_ratio:.2f})")
            except Exception:
                pass
            
            # --- Copy-move detection (simplified) ---
            # Check for identical blocks at different positions
            try:
                block_size = 16
                if h >= block_size * 6 and w >= block_size * 6:
                    # Sample blocks at regular positions
                    blocks = {}
                    step = block_size * 2
                    for by in range(0, h - block_size, step):
                        for bx in range(0, w - block_size, step):
                            block = gray[by:by+block_size, bx:bx+block_size]
                            # Quantize to reduce noise sensitivity
                            block_quant = (block // 4).astype(np.uint8)
                            block_hash = hashlib.md5(block_quant.tobytes()).hexdigest()[:8]
                            
                            if block_hash in blocks:
                                prev_y, prev_x = blocks[block_hash]
                                dist = np.sqrt((by - prev_y)**2 + (bx - prev_x)**2)
                                # Ignore adjacent blocks and very far blocks
                                if block_size * 3 < dist < min(h, w) * 0.8:
                                    tampering_score += 0.15
                                    evidence.append(
                                        f"Identical block found at ({prev_x},{prev_y}) and ({bx},{by})")
                                    break
                            else:
                                blocks[block_hash] = (by, bx)
                        else:
                            continue
                        break
            except Exception:
                pass
            
            # --- Edge inconsistency at potential splice boundaries ---
            try:
                # Look for abrupt changes in noise level along edges
                edge_map = ndimage.sobel(gray)
                
                # Analyze noise consistency along strong edges
                strong_edges = edge_map > np.percentile(edge_map, 95)
                if np.sum(strong_edges) > 50:
                    # Check noise on both sides of strong edges
                    dilated = ndimage.binary_dilation(strong_edges, iterations=3)
                    edge_border = dilated & ~strong_edges
                    
                    if np.sum(edge_border) > 100:
                        border_noise = np.std(gray[edge_border] - 
                                            ndimage.gaussian_filter(gray, 1.0)[edge_border])
                        overall_noise = np.std(gray - ndimage.gaussian_filter(gray, 1.0))
                        
                        if overall_noise > 0:
                            noise_ratio = border_noise / overall_noise
                            if noise_ratio > 2.0:
                                tampering_score += 0.08
                                evidence.append(
                                    f"Edge noise inconsistency (ratio={noise_ratio:.2f})")
            except Exception:
                pass
            
            if tampering_score > 0.05:
                severity = 4 if tampering_score >= 0.3 else (3 if tampering_score >= 0.15 else 2)
                indicators.append({
                    "code": "TAMPERING_DETECTED",
                    "severity": severity,
                    "description": "Image tampering or manipulation detected",
                    "evidence": evidence[:4]
                })
        
        except ImportError:
            logger.debug("scipy not available for tampering detection")
        except Exception as e:
            logger.debug(f"Tampering detection error: {e}")
        
        return min(tampering_score, 1.0), indicators
    
    # ------------------------------------------------------------------
    #  10. FACE / OBJECT CONSISTENCY CHECK
    #  Checks for anomalies that suggest compositing:
    #  - Shadow direction inconsistency
    #  - Perspective / vanishing point violations
    #  - Scale inconsistency between objects
    #  - Lighting direction analysis
    # ------------------------------------------------------------------
    def _check_content_consistency(self, image: Image.Image) -> Tuple[float, List[Dict[str, Any]]]:
        """Check for face/object consistency anomalies"""
        indicators = []
        consistency_score = 1.0  # 1.0 = fully consistent, lower = inconsistent
        evidence = []
        
        try:
            img_array = np.array(image, dtype=np.float64)
            gray = np.mean(img_array, axis=2)
            h, w = gray.shape
            
            if h < 128 or w < 128:
                return 1.0, []
            
            from scipy import ndimage
            
            # --- Shadow direction analysis ---
            # In a real scene, shadows should all point roughly the same direction
            # Detect shadow regions and estimate their direction
            luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
            shadow_thresh = np.percentile(luminance, 15)
            shadow_mask = luminance < shadow_thresh
            
            if np.sum(shadow_mask) > 500:
                # Compute gradient direction in shadow regions
                grad_x = ndimage.sobel(luminance, axis=1)
                grad_y = ndimage.sobel(luminance, axis=0)
                
                # Only look at shadow boundaries
                shadow_edge = ndimage.binary_dilation(shadow_mask, iterations=2) & ~shadow_mask
                
                if np.sum(shadow_edge) > 50:
                    edge_angles = np.arctan2(
                        grad_y[shadow_edge],
                        grad_x[shadow_edge]
                    )
                    # Filter out very weak gradients
                    grad_mag = np.sqrt(grad_x[shadow_edge]**2 + grad_y[shadow_edge]**2)
                    strong = grad_mag > np.percentile(grad_mag, 50)
                    
                    if np.sum(strong) > 20:
                        strong_angles = edge_angles[strong]
                        # Circular standard deviation
                        angle_consistency = np.abs(np.mean(np.exp(1j * strong_angles)))
                        
                        # Perfectly consistent shadows: ~0.8+
                        # Inconsistent (composited): < 0.3
                        if angle_consistency < 0.2:
                            consistency_score -= 0.15
                            evidence.append(
                                f"Shadow directions inconsistent (coherence={angle_consistency:.3f})")
            
            # --- Lighting direction estimation ---
            # Compare lighting direction between image quadrants
            qh, qw = h // 2, w // 2
            quadrants = [
                luminance[:qh, :qw],
                luminance[:qh, qw:],
                luminance[qh:, :qw],
                luminance[qh:, qw:],
            ]
            q_brightnesses = [np.mean(q) for q in quadrants]
            brightness_range = max(q_brightnesses) - min(q_brightnesses)
            brightness_mean = np.mean(q_brightnesses)
            
            if brightness_mean > 0:
                # Check if lighting gradient is smooth (natural) or abrupt (composite)
                # Natural: gradual gradient across quadrants
                # Composite: sudden brightness jumps
                adjacent_diffs = [
                    abs(q_brightnesses[0] - q_brightnesses[1]),  # top-left vs top-right
                    abs(q_brightnesses[2] - q_brightnesses[3]),  # bottom-left vs bottom-right
                    abs(q_brightnesses[0] - q_brightnesses[2]),  # top-left vs bottom-left
                    abs(q_brightnesses[1] - q_brightnesses[3]),  # top-right vs bottom-right
                ]
                max_adj_diff = max(adjacent_diffs)
                
                if brightness_range > 0 and max_adj_diff > brightness_range * 0.8:
                    consistency_score -= 0.08
                    evidence.append("Abrupt lighting transition between image regions")
            
            # --- Perspective consistency (vanishing point) ---
            # Check if strong edges converge to consistent vanishing points
            sobel_x = ndimage.sobel(gray, axis=1)
            sobel_y = ndimage.sobel(gray, axis=0)
            edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Sample strong horizontal and vertical edges
            strong_edges = edge_mag > np.percentile(edge_mag, 95)
            h_edges = strong_edges & (np.abs(sobel_x) > np.abs(sobel_y) * 2)
            v_edges = strong_edges & (np.abs(sobel_y) > np.abs(sobel_x) * 2)
            
            # If we have mostly horizontal lines, check their angles for convergence
            h_angles = np.arctan2(sobel_y[h_edges], sobel_x[h_edges]) if np.sum(h_edges) > 30 else np.array([])
            
            if len(h_angles) > 30:
                # Filter near-horizontal angles
                near_horiz = h_angles[np.abs(h_angles) < 0.3]
                if len(near_horiz) > 20:
                    angle_spread = np.std(near_horiz)
                    # Very tight spread with few outliers is normal
                    # Bimodal or very wide spread suggests compositing
                    if angle_spread > 0.15:
                        # Check if bimodal
                        hist, _ = np.histogram(near_horiz, bins=20)
                        hist_norm = hist / np.sum(hist)
                        peaks = sum(1 for i in range(1, len(hist_norm)-1) 
                                   if hist_norm[i] > hist_norm[i-1] and hist_norm[i] > hist_norm[i+1]
                                   and hist_norm[i] > 0.05)
                        if peaks >= 3:
                            consistency_score -= 0.10
                            evidence.append(
                                f"Multiple vanishing points detected ({peaks} edge groups)")
            
            # Clamp consistency score
            consistency_score = max(0.0, min(1.0, consistency_score))
            
            if consistency_score < 0.85:
                severity = 3 if consistency_score < 0.7 else 2
                indicators.append({
                    "code": "CONTENT_INCONSISTENCY",
                    "severity": severity,
                    "description": "Visual inconsistencies suggest image compositing",
                    "evidence": evidence[:4]
                })
        
        except ImportError:
            logger.debug("scipy not available for content consistency check")
        except Exception as e:
            logger.debug(f"Content consistency check error: {e}")
        
        return consistency_score, indicators
    
    # ------------------------------------------------------------------
    #  11. PERCEPTUAL HASH (pHash) — Duplicate Detection
    #  DCT-based perceptual hash for near-duplicate matching across listings
    # ------------------------------------------------------------------
    def _calculate_perceptual_hash(self, image: Image.Image, hash_size: int = 8) -> str:
        """Calculate perceptual hash (pHash) using DCT"""
        try:
            # Resize to small square (slightly larger than hash for DCT)
            img_small = image.convert('L').resize(
                (hash_size * 4, hash_size * 4), _LANCZOS
            )
            pixels = np.array(img_small, dtype=np.float64)
            
            # Apply 2D DCT
            # Use scipy DCT if available, otherwise approximate with FFT
            try:
                from scipy.fft import dctn
                dct = dctn(pixels, type=2, norm='ortho')
            except ImportError:
                # Approximate DCT with FFT
                dct = np.real(np.fft.fft2(pixels))
            
            # Take top-left low-frequency block
            dct_low = dct[:hash_size, :hash_size]
            
            # Compute median (excluding DC component)
            dct_low_flat = dct_low.flatten()
            median = np.median(dct_low_flat[1:])  # Skip DC
            
            # Build hash — each bit is 1 if above median
            hash_bits = dct_low_flat > median
            
            # Convert to hex string
            hash_int = 0
            for bit in hash_bits:
                hash_int = (hash_int << 1) | int(bit)
            
            return format(hash_int, f'0{hash_size * hash_size // 4}x')
            
        except Exception as e:
            logger.debug(f"Perceptual hash calculation error: {e}")
            # Fallback: simple average hash
            img_small = image.convert('L').resize((8, 8), _LANCZOS)
            pixels = np.array(img_small)
            avg = np.mean(pixels)
            bits = pixels > avg
            hash_int = 0
            for bit in bits.flatten():
                hash_int = (hash_int << 1) | int(bit)
            return format(hash_int, '016x')
    
    def _check_duplicates(
        self, phash: str, listing_id: str
    ) -> List[str]:
        """Check if perceptual hash matches any other listing's images"""
        matches = []
        
        try:
            if not phash or not self._hash_database:
                return matches
            
            # Exact hash match
            if phash in self._hash_database:
                other_listings = [
                    lid for lid in self._hash_database[phash]
                    if lid != listing_id
                ]
                matches.extend(other_listings)
            
            # Near-match: Hamming distance ≤ 5 bits
            try:
                phash_int = int(phash, 16)
                for stored_hash, listing_ids in self._hash_database.items():
                    if stored_hash == phash:
                        continue
                    try:
                        stored_int = int(stored_hash, 16)
                        xor = phash_int ^ stored_int
                        hamming = bin(xor).count('1')
                        if hamming <= 5:
                            for lid in listing_ids:
                                if lid != listing_id and lid not in matches:
                                    matches.append(lid)
                    except ValueError:
                        continue
            except ValueError:
                pass
        
        except Exception as e:
            logger.debug(f"Duplicate check error: {e}")
        
        return matches
    
    # ------------------------------------------------------------------
    #  COMPOSITE AI SCORE MERGER
    # ------------------------------------------------------------------
    def _merge_ai_scores(
        self,
        statistical_score: float,
        gan_score: float,
        diffusion_score: float,
    ) -> float:
        """
        Merge multiple AI detection subscores into a single composite.
        Uses max-dominant blend: the highest signal drives the result,
        with supporting signals adding incremental evidence.
        """
        scores = [statistical_score, gan_score, diffusion_score]
        
        if max(scores) == 0:
            return 0.0
        
        # Max-dominant blend — strongest detector drives the score
        dominant = max(scores)
        # Supporting evidence from other detectors
        supporting = sorted(scores, reverse=True)
        composite = dominant
        if len(supporting) > 1 and supporting[1] >= 0.08:
            composite += supporting[1] * 0.40  # Second signal adds 40%
        if len(supporting) > 2 and supporting[2] >= 0.08:
            composite += supporting[2] * 0.25  # Third signal adds 25%
        
        # Agreement bonus: if multiple methods flag the image, boost confidence
        flagged = sum(1 for s in scores if s >= 0.10)
        if flagged >= 3:
            composite = min(composite * 1.25, 1.0)
        elif flagged >= 2:
            composite = min(composite * 1.15, 1.0)
        
        return min(composite, 1.0)
    
    # ------------------------------------------------------------------
    #  CONFIDENCE CALIBRATION
    #  Platt-scaled cross-signal agreement for calibrated probability
    # ------------------------------------------------------------------
    def _calibrate_confidence(
        self, *,
        ai_score: float,
        gan_score: float,
        diffusion_score: float,
        sensor_score: float,
        tampering_score: float,
        quality_score: float,
        risk_score: float,
    ) -> Dict[str, Any]:
        """
        Compute calibrated confidence for the overall risk assessment.
        
        Uses signal agreement and Platt-like scaling to produce a
        well-calibrated probability estimate.
        """
        # Individual signal confidences
        signals = {
            "ai_forensic": ai_score,
            "gan_detection": gan_score,
            "diffusion_detection": diffusion_score,
            "sensor_analysis": sensor_score,
            "tampering_analysis": tampering_score,
            "quality_analysis": 1.0 - quality_score,  # Invert: low quality = high suspicion
        }
        
        active_signals = {k: v for k, v in signals.items() if v > 0.05}
        
        if not active_signals:
            return {
                "calibrated_probability": round(risk_score, 3),
                "confidence_level": "low",
                "signal_agreement": 0.0,
                "active_signals": 0,
                "total_signals": len(signals),
            }
        
        # Signal agreement — how many methods agree on risk direction
        high_risk_signals = sum(1 for v in active_signals.values() if v >= 0.15)
        low_risk_signals = sum(1 for v in active_signals.values() if v < 0.10)
        
        agreement = high_risk_signals / len(signals) if high_risk_signals > 0 else 0
        
        # Platt-like calibration: sigmoid(a * raw_score + b)
        # Tuned for property image context — steeper curve to avoid
        # under-confident outputs when multiple signals agree
        a = 6.0  # Steepness (higher = sharper transition)
        b = -1.8  # Offset (shift sigmoid center left to catch mid-range risks)
        raw = risk_score
        calibrated = 1.0 / (1.0 + np.exp(-(a * raw + b)))
        
        # Adjust by agreement — strong multi-signal agreement should push
        # the calibrated probability higher, not just +10%
        if agreement >= 0.5:
            calibrated = min(calibrated * 1.25, 1.0)
        elif agreement >= 0.33:
            calibrated = min(calibrated * 1.15, 1.0)
        elif agreement < 0.17 and calibrated > 0.5:
            calibrated *= 0.85
        
        # Determine confidence level
        if len(active_signals) >= 4 and agreement >= 0.4:
            conf_level = "high"
        elif len(active_signals) >= 2:
            conf_level = "medium"
        else:
            conf_level = "low"
        
        return {
            "calibrated_probability": round(float(calibrated), 3),
            "confidence_level": conf_level,
            "signal_agreement": round(float(agreement), 3),
            "active_signals": len(active_signals),
            "total_signals": len(signals),
        }
    
    # ------------------------------------------------------------------
    #  EXPLAINABILITY
    #  Per-signal contribution breakdown with human-readable reasoning
    # ------------------------------------------------------------------
    def _compute_explainability(
        self, *,
        ai_score: float,
        gan_score: float,
        diffusion_score: float,
        sensor_score: float,
        tampering_score: float,
        consistency_score: float,
        quality_score: float,
        web_detection: Optional[Dict[str, Any]],
        duplicate_count: int,
        calibration: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate per-signal explainability breakdown.
        
        Returns contribution weights, reasoning chain, and
        human-readable summary of what drove the risk decision.
        """
        # Define signal contributions with weights
        contributions = []
        reasoning = []
        
        def _add(name: str, score: float, weight: float, description: str):
            weighted = score * weight
            contributions.append({
                "signal": name,
                "raw_score": round(score, 3),
                "weight": weight,
                "weighted_contribution": round(weighted, 4),
                "description": description,
            })
            if score >= 0.1:
                reasoning.append(description)
        
        _add("AI Forensic Analysis", ai_score, 0.25,
             f"Statistical forensics scored {ai_score:.0%} — "
             + ("strong AI indicators" if ai_score >= 0.3
                else "moderate indicators" if ai_score >= 0.15
                else "minor/no indicators"))
        
        _add("GAN Fingerprint", gan_score, 0.12,
             f"GAN artifact score {gan_score:.0%} — "
             + ("GAN artifacts found" if gan_score >= 0.15
                else "no GAN fingerprints"))
        
        _add("Diffusion Artifacts", diffusion_score, 0.12,
             f"Diffusion score {diffusion_score:.0%} — "
             + ("diffusion model patterns found" if diffusion_score >= 0.15
                else "no diffusion artifacts"))
        
        _add("Sensor Noise", sensor_score, 0.10,
             f"Sensor analysis scored {sensor_score:.0%} — "
             + ("no genuine camera sensor noise" if sensor_score >= 0.15
                else "sensor noise appears normal"))
        
        _add("Tampering Detection", tampering_score, 0.12,
             f"Tampering score {tampering_score:.0%} — "
             + ("manipulation evidence found" if tampering_score >= 0.15
                else "no tampering detected"))
        
        consistency_risk = 1.0 - consistency_score
        _add("Content Consistency", consistency_risk, 0.08,
             f"Consistency {consistency_score:.0%} — "
             + ("visual inconsistencies detected" if consistency_score < 0.85
                else "content appears consistent"))
        
        quality_risk = 1.0 - quality_score
        _add("Image Quality", quality_risk, 0.08,
             f"Quality {quality_score:.0%} — "
             + ("poor quality" if quality_score < 0.5
                else "acceptable quality"))
        
        web_risk = 0.0
        if web_detection and web_detection.get("has_web_matches"):
            web_risk = min(web_detection.get("total_matches", 0) * 0.1, 0.8)
        _add("Web Detection", web_risk, 0.08,
             f"Found {web_detection.get('total_matches', 0) if web_detection else 0} web matches"
             + (" — possible stock/stolen image" if web_risk > 0.2 else ""))
        
        dup_risk = min(duplicate_count * 0.3, 1.0)
        _add("Duplicate Detection", dup_risk, 0.05,
             f"Image matched {duplicate_count} other listing(s)"
             if duplicate_count > 0 else "No cross-listing duplicates found")
        
        # Sort by contribution (highest first)
        contributions.sort(key=lambda c: c["weighted_contribution"], reverse=True)
        
        # Top drivers
        top_drivers = [c["signal"] for c in contributions[:3] if c["weighted_contribution"] > 0.01]
        
        return {
            "contributions": contributions,
            "reasoning_chain": reasoning[:6],
            "top_risk_drivers": top_drivers,
            "confidence": calibration.get("confidence_level", "low"),
            "calibrated_probability": calibration.get("calibrated_probability", 0.0),
        }
    
    # ------------------------------------------------------------------
    #  GOOGLE CLOUD VISION  –  Web Detection (reverse image search)
    # ------------------------------------------------------------------
    async def _web_detection(self, image_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Query Google Cloud Vision Web Detection API to find where this
        image (or visually similar images) appears on the web.

        Returns None when:
          - No API key configured (graceful skip)
          - API call fails / quota exceeded (graceful degradation)

        Returns dict with:
          web_entities        – topics/labels Google associates with the image
          full_matching_images – exact matches found online
          partial_matching_images – visually similar matches
          pages_with_matching_images – web pages containing this image
          best_guess_labels   – Google's best-guess description
        """
        try:
            from config import get_settings
        except Exception as e:
            logger.debug(f"Config import failed; skipping web detection: {e}")
            return None

        try:
            settings = get_settings()
            api_key = getattr(settings, "GOOGLE_CLOUD_VISION_API_KEY", None)
        except Exception as e:
            logger.debug(f"Settings load failed; skipping web detection: {e}")
            return None

        if not api_key:
            logger.debug("Google Cloud Vision API key not configured – skipping web detection")
            return None

        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        payload = {
            "requests": [{
                "image": {
                    "content": base64.b64encode(image_data).decode("utf-8")
                },
                "features": [{
                    "type": "WEB_DETECTION",
                    "maxResults": 20
                }]
            }]
        }

        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json=payload)

            if resp.status_code != 200:
                logger.warning(
                    f"Google Vision API returned {resp.status_code}: "
                    f"{resp.text[:200]}"
                )
                return None

            data = resp.json()
            annotation = (
                data.get("responses", [{}])[0]
                    .get("webDetection", {})
            )

            if not annotation:
                return {
                    "web_entities": [],
                    "full_matching_images": [],
                    "partial_matching_images": [],
                    "pages_with_matching_images": [],
                    "best_guess_labels": [],
                    "has_web_matches": False,
                    "total_matches": 0,
                }

            # ---- web entities (topics) ----
            web_entities = []
            for ent in annotation.get("webEntities", []):
                if ent.get("description"):
                    web_entities.append({
                        "description": ent["description"],
                        "score": round(ent.get("score", 0), 3),
                    })

            # ---- full matching images (exact duplicates online) ----
            full_matches = []
            for img in annotation.get("fullMatchingImages", []):
                full_matches.append({"url": img.get("url", "")})

            # ---- partial / visually-similar matches ----
            partial_matches = []
            for img in annotation.get("partialMatchingImages", []):
                partial_matches.append({"url": img.get("url", "")})

            # ---- pages containing matching images ----
            pages = []
            for pg in annotation.get("pagesWithMatchingImages", []):
                pages.append({
                    "url": pg.get("url", ""),
                    "page_title": pg.get("pageTitle", ""),
                })

            # ---- best guess labels ----
            best_labels = []
            for lbl in annotation.get("bestGuessLabels", []):
                if lbl.get("label"):
                    best_labels.append(lbl["label"])

            total = len(full_matches) + len(partial_matches)

            return {
                "web_entities": web_entities[:10],
                "full_matching_images": full_matches[:10],
                "partial_matching_images": partial_matches[:10],
                "pages_with_matching_images": pages[:10],
                "best_guess_labels": best_labels,
                "has_web_matches": total > 0,
                "total_matches": total,
            }

        except Exception as e:
            logger.warning(f"Google Vision web detection failed: {e}")
            return None
    
    def _calculate_risk(
        self,
        classification: Optional[ImageClassificationResult],
        quality_score: float,
        ai_score: float,
        indicators: List[Dict[str, Any]],
        *,
        sensor_score: float = 0.0,
        tampering_score: float = 0.0,
        consistency_score: float = 1.0,
        duplicate_count: int = 0,
    ) -> Tuple[float, ImageRiskLevel]:
        """Calculate overall risk score and level with forensic signals"""
        
        risk_score = 0.0
        
        # Factor in classification (10% weight)
        if classification:
            if not classification.is_property_related:
                risk_score += 0.08
                if classification.risk_indicators:
                    risk_score += 0.02
            else:
                risk_score += max(0, (1 - classification.property_confidence) * 0.03 - 0.01)
        else:
            risk_score += 0.05
        
        # Factor in quality (8% weight)
        risk_score += (1 - quality_score) * 0.08
        
        # Factor in AI detection (35% weight — composite already merges
        # statistical + GAN + diffusion via _merge_ai_scores)
        # Linear scaling — no diminishing returns, the composite already
        # handles blending / agreement.
        risk_score += ai_score * 0.35
        
        # Factor in sensor noise (10% weight)
        # Missing sensor noise is a strong indicator of AI generation
        risk_score += sensor_score * 0.10
        
        # Factor in tampering (15% weight)
        # Tampering is a direct fraud signal — give it strong weight
        risk_score += tampering_score * 0.15
        
        # Factor in content consistency (8% weight)
        risk_score += (1.0 - consistency_score) * 0.08
        
        # Factor in indicator severity (7% weight)
        total_severity = sum(i.get("severity", 1) for i in indicators)
        risk_score += min(total_severity * 0.012, 0.07)
        
        # Factor in web detection matches
        web_codes = {i.get("code") for i in indicators}
        if "WEB_EXACT_MATCH" in web_codes:
            risk_score += 0.10
        elif "WEB_PARTIAL_MATCH" in web_codes:
            risk_score += 0.04
        
        # Factor in duplicate listings (7% weight)
        if duplicate_count > 0:
            risk_score += min(duplicate_count * 0.05, 0.12)
        
        # Cap risk score
        risk_score = min(risk_score, 1.0)
        
        # Determine risk level
        if risk_score < 0.15:
            level = ImageRiskLevel.AUTHENTIC
        elif risk_score < 0.30:
            level = ImageRiskLevel.LIKELY_AUTHENTIC
        elif risk_score < 0.50:
            level = ImageRiskLevel.UNCERTAIN
        elif risk_score < 0.70:
            level = ImageRiskLevel.SUSPICIOUS
        else:
            level = ImageRiskLevel.LIKELY_FAKE
        
        return risk_score, level
    
    def _generate_explanation(
        self,
        classification: Optional[ImageClassificationResult],
        quality_score: float,
        ai_score: float,
        risk_level: ImageRiskLevel,
        web_detection: Optional[Dict[str, Any]] = None,
        *,
        gan_score: float = 0.0,
        diffusion_score: float = 0.0,
        tampering_score: float = 0.0,
        duplicate_count: int = 0,
    ) -> str:
        """Generate human-readable explanation with forensic detail"""
        
        parts = []
        
        # Classification explanation
        if classification:
            if classification.is_property_related:
                conf = classification.property_confidence
                parts.append(f"✅ Image appears to be property-related ({conf:.0%} confidence)")
            else:
                parts.append("⚠️ Image does not appear to show property/interior content")
        else:
            parts.append("❓ Image classification unavailable")
        
        # Quality explanation
        if quality_score > 0.8:
            parts.append("✅ Good image quality")
        elif quality_score > 0.5:
            parts.append("ℹ️ Moderate image quality")
        else:
            parts.append("⚠️ Low image quality detected")
        
        # AI detection explanation
        if ai_score >= 0.4:
            parts.append(f"🤖 Strong AI-generation indicators detected (score: {ai_score:.0%})")
        elif ai_score >= 0.25:
            parts.append(f"🤖 Moderate AI-generation indicators detected (score: {ai_score:.0%})")
        elif ai_score >= 0.12:
            parts.append(f"ℹ️ Some AI-generation indicators noted (score: {ai_score:.0%})")
        else:
            parts.append("✅ No significant AI-generation indicators")
        
        # GAN-specific
        if gan_score >= 0.15:
            parts.append(f"🔬 GAN fingerprints detected (score: {gan_score:.0%})")
        
        # Diffusion-specific
        if diffusion_score >= 0.15:
            parts.append(f"🔬 Diffusion model artifacts detected (score: {diffusion_score:.0%})")
        
        # Tampering
        if tampering_score >= 0.15:
            parts.append(f"✂️ Image tampering/manipulation detected (score: {tampering_score:.0%})")
        elif tampering_score >= 0.08:
            parts.append(f"ℹ️ Minor tampering indicators (score: {tampering_score:.0%})")
        
        # Duplicates
        if duplicate_count > 0:
            parts.append(
                f"📋 Image matches {duplicate_count} other listing{'s' if duplicate_count > 1 else ''}")
        
        # Web detection explanation
        if web_detection and web_detection.get("has_web_matches"):
            full_ct = len(web_detection.get("full_matching_images", []))
            partial_ct = len(web_detection.get("partial_matching_images", []))
            if full_ct > 0:
                parts.append(f"🌐 Exact match found on {full_ct} site{'' if full_ct == 1 else 's'} — possible stock/stolen image")
            elif partial_ct > 0:
                parts.append(f"🌐 Visually similar images found on {partial_ct} site{'' if partial_ct == 1 else 's'}")
        elif web_detection is not None:
            parts.append("✅ No web matches found")
        
        # Overall conclusion
        if risk_level == ImageRiskLevel.AUTHENTIC:
            parts.append("Overall: Image appears authentic")
        elif risk_level == ImageRiskLevel.LIKELY_AUTHENTIC:
            parts.append("Overall: Image is likely authentic")
        elif risk_level == ImageRiskLevel.UNCERTAIN:
            parts.append("Overall: Image authenticity uncertain - verify independently")
        elif risk_level == ImageRiskLevel.SUSPICIOUS:
            parts.append("Overall: Image has suspicious characteristics")
        else:
            parts.append("Overall: Image is likely fake or manipulated")
        
        return " | ".join(parts)
    
    async def analyze_multiple_images(
        self,
        images: List[Tuple[bytes, Optional[str]]],
        listing_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Analyze multiple images from a listing using parallel processing.
        
        Args:
            images: List of (image_bytes, filename) tuples
            listing_id: Listing identifier for cross-listing duplicate detection
        
        Returns:
            Combined analysis result with forensic aggregation
        """
        if not images:
            return {
                "image_count": 0,
                "overall_risk_score": 0.5,
                "overall_risk_level": ImageRiskLevel.UNCERTAIN.value,
                "property_images_count": 0,
                "suspicious_images_count": 0,
                "images": [],
                "summary": "No images to analyze"
            }
        
        # --- Parallel analysis using asyncio.gather ---
        async def _analyze_one(pair):
            image_data, filename = pair
            return await self.analyze_image(image_data, filename, listing_id=listing_id)
        
        results = await asyncio.gather(
            *[_analyze_one(pair) for pair in images],
            return_exceptions=True,
        )
        
        # Filter out exceptions
        valid_results: List[ImageAnalysisResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning(f"Image {i} analysis failed: {r}")
            else:
                valid_results.append(r)
        
        if not valid_results:
            return {
                "image_count": len(images),
                "overall_risk_score": 0.5,
                "overall_risk_level": ImageRiskLevel.UNCERTAIN.value,
                "property_images_count": 0,
                "suspicious_images_count": 0,
                "images": [],
                "summary": "All image analyses failed"
            }
        
        results = valid_results
        
        # Aggregate results
        overall_risk = float(np.mean([r.risk_score for r in results]))
        property_count = sum(1 for r in results if r.is_property_image)
        suspicious_count = sum(
            1 for r in results 
            if r.risk_level in [ImageRiskLevel.SUSPICIOUS, ImageRiskLevel.LIKELY_FAKE]
        )
        ai_suspected_count = sum(
            1 for r in results if r.ai_detection_score >= 0.25
        )
        avg_ai_score = float(np.mean([r.ai_detection_score for r in results]))
        web_match_count = sum(
            1 for r in results
            if r.web_detection and r.web_detection.get("has_web_matches")
        )
        
        # --- New forensic aggregations ---
        avg_gan = float(np.mean([r.gan_fingerprint_score for r in results]))
        avg_diffusion = float(np.mean([r.diffusion_artifact_score for r in results]))
        avg_sensor = float(np.mean([r.sensor_noise_score for r in results]))
        avg_tampering = float(np.mean([r.tampering_score for r in results]))
        avg_consistency = float(np.mean([r.content_consistency_score for r in results]))
        
        total_duplicates = sum(len(r.duplicate_listings) for r in results)
        all_duplicate_ids: List[str] = []
        for r in results:
            for dup in r.duplicate_listings:
                if dup not in all_duplicate_ids:
                    all_duplicate_ids.append(dup)
        
        tampering_count = sum(1 for r in results if r.tampering_score >= 0.15)
        gan_count = sum(1 for r in results if r.gan_fingerprint_score >= 0.15)
        diffusion_count = sum(1 for r in results if r.diffusion_artifact_score >= 0.15)
        
        # Determine overall level
        if overall_risk < 0.2:
            overall_level = ImageRiskLevel.AUTHENTIC
        elif overall_risk < 0.4:
            overall_level = ImageRiskLevel.LIKELY_AUTHENTIC
        elif overall_risk < 0.6:
            overall_level = ImageRiskLevel.UNCERTAIN
        else:
            overall_level = ImageRiskLevel.SUSPICIOUS
        
        # Generate summary
        total = len(results)
        parts = []
        if ai_suspected_count > 0:
            parts.append(f"🤖 {ai_suspected_count} of {total} images show AI-generation indicators")
        if gan_count > 0:
            parts.append(f"🔬 {gan_count} of {total} images have GAN fingerprints")
        if diffusion_count > 0:
            parts.append(f"🔬 {diffusion_count} of {total} images have diffusion artifacts")
        if tampering_count > 0:
            parts.append(f"✂️ {tampering_count} of {total} images show tampering")
        if web_match_count > 0:
            parts.append(f"🌐 {web_match_count} of {total} images found elsewhere on the web")
        if total_duplicates > 0:
            parts.append(f"📋 Images match {len(all_duplicate_ids)} other listing(s)")
        if suspicious_count > 0:
            parts.append(f"⚠️ {suspicious_count} of {total} images appear suspicious")
        if property_count == 0 and total > 0:
            parts.append(f"⚠️ None of the {total} images appear to be property photos")
        
        if not parts:
            if property_count > 0:
                summary = f"✅ All {total} images appear authentic and property-related"
            else:
                summary = f"ℹ️ {property_count} of {total} images appear property-related"
        else:
            summary = " | ".join(parts)
        
        return {
            "image_count": total,
            "overall_risk_score": round(overall_risk, 3),
            "overall_risk_level": overall_level.value,
            "property_images_count": property_count,
            "suspicious_images_count": suspicious_count,
            "ai_suspected_count": ai_suspected_count,
            "web_match_count": web_match_count,
            "average_ai_score": round(avg_ai_score, 3),
            # Forensic aggregations
            "forensics": {
                "average_gan_score": round(avg_gan, 3),
                "average_diffusion_score": round(avg_diffusion, 3),
                "average_sensor_score": round(avg_sensor, 3),
                "average_tampering_score": round(avg_tampering, 3),
                "average_consistency_score": round(avg_consistency, 3),
                "gan_flagged_count": gan_count,
                "diffusion_flagged_count": diffusion_count,
                "tampering_flagged_count": tampering_count,
                "duplicate_listing_ids": all_duplicate_ids,
            },
            "images": [r.to_dict() for r in results],
            "summary": summary
        }


# Singleton instance
real_image_engine = RealImageClassificationEngine()

# Alias for compatibility with existing code
image_analysis_engine = real_image_engine
