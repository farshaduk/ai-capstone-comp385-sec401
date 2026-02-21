from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from infrastructure.database import UserModel, SubscriptionPlanModel, ListingModel
from passlib.context import CryptContext
from datetime import datetime

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def seed_database(db: AsyncSession):
    """Seed the database with initial data"""
    
    # Check if data already exists
    result = await db.execute(text("SELECT COUNT(*) FROM users"))
    count = result.scalar()
    
    if count > 0:
        print("Database already seeded, skipping...")
        return
    
    # All features available for every plan — differentiated by scans & support level
    all_features_base = {
        "basic_analysis": True,
        "risk_score": True,
        "detailed_indicators": True,
        "image_analysis": True,
        "advanced_ml": True,
        "export_reports": True,
        "api_access": True,
        "custom_models": True,
        "message_analysis": True,
        "conversation_analysis": True,
        "address_validation": True,
        "price_anomaly": True,
        "xai_explanations": True,
        "url_analysis": True,
    }

    # Create subscription plans
    plans = [
        SubscriptionPlanModel(
            name="free",
            display_name="Free Plan",
            price=0.0,
            scans_per_month=1000,
            features={**all_features_base, "history": 1000, "support": "community"},
            is_active=True
        ),
        SubscriptionPlanModel(
            name="basic",
            display_name="Basic Plan",
            price=9.99,
            scans_per_month=50,
            features={**all_features_base, "history": 50, "support": "email"},
            is_active=True
        ),
        SubscriptionPlanModel(
            name="premium",
            display_name="Premium Plan",
            price=29.99,
            scans_per_month=200,
            features={**all_features_base, "history": 200, "support": "priority"},
            is_active=True
        ),
        SubscriptionPlanModel(
            name="enterprise",
            display_name="Enterprise Plan",
            price=99.99,
            scans_per_month=1000,
            features={**all_features_base, "history": -1, "support": "24/7"},
            is_active=True
        )
    ]
    
    for plan in plans:
        db.add(plan)
    
    # Create admin user
    admin_user = UserModel(
        email="admin@rentalfraud.com",
        hashed_password=pwd_context.hash("admin123"),
        full_name="System Administrator",
        role="admin",
        subscription_plan="enterprise",
        is_active=True,
        scans_remaining=1000,
        created_at=datetime.utcnow()
    )
    db.add(admin_user)
    
    # Create demo renter users
    renter_users = [
        UserModel(
            email="renter1@example.com",
            hashed_password=pwd_context.hash("renter123"),
            full_name="John Doe",
            role="renter",
            subscription_plan="free",
            is_active=True,
            scans_remaining=1000
        ),
        UserModel(
            email="renter2@example.com",
            hashed_password=pwd_context.hash("renter123"),
            full_name="Jane Smith",
            role="renter",
            subscription_plan="premium",
            is_active=True,
            scans_remaining=200
        )
    ]
    
    for user in renter_users:
        db.add(user)
    
    # Create demo landlord user
    landlord_user = UserModel(
        email="landlord@example.com",
        hashed_password=pwd_context.hash("landlord123"),
        full_name="Property Owner",
        role="landlord",
        subscription_plan="premium",
        is_active=True,
        scans_remaining=200
    )
    db.add(landlord_user)
    
    await db.commit()

    # Refresh landlord user to get ID
    await db.refresh(landlord_user)

    # Seed sample listings owned by the landlord
    sample_listings = [
        ListingModel(
            owner_id=landlord_user.id,
            title="2BR Apartment - Downtown Core",
            address="123 King St W",
            city="Toronto",
            province="ON",
            postal_code="M5H 1A1",
            price=2400,
            beds=2,
            baths=1,
            sqft=850,
            property_type="apartment",
            description="Bright 2-bedroom apartment in the heart of downtown Toronto. Walking distance to TTC, restaurants, and entertainment.",
            amenities=["Gym", "Concierge", "Balcony", "AC"],
            laundry="in_unit",
            utilities="not_included",
            pet_friendly=False,
            parking_included=False,
            available_date="2026-03-01",
            is_active=True,
            is_verified=True,
            risk_score=0.1,
            views=45,
        ),
        ListingModel(
            owner_id=landlord_user.id,
            title="Spacious 1BR + Den - Liberty Village",
            address="45 East Liberty St",
            city="Toronto",
            province="ON",
            postal_code="M6K 3P8",
            price=2100,
            beds=1,
            baths=1,
            sqft=720,
            property_type="condo",
            description="Modern 1BR+Den in Liberty Village with open-concept layout. Steps to King streetcar and Liberty Village park.",
            amenities=["Gym", "Pool", "Rooftop", "Bike Room"],
            laundry="in_unit",
            utilities="partial",
            pet_friendly=True,
            parking_included=True,
            available_date="2026-03-15",
            is_active=True,
            is_verified=True,
            risk_score=0.05,
            views=32,
        ),
        ListingModel(
            owner_id=landlord_user.id,
            title="Luxury 3BR Penthouse - Harbourfront",
            address="88 Harbour St",
            city="Toronto",
            province="ON",
            postal_code="M5J 0B7",
            price=4500,
            beds=3,
            baths=2,
            sqft=1500,
            property_type="condo",
            description="Stunning penthouse with panoramic lake views. Premium finishes throughout, floor-to-ceiling windows.",
            amenities=["Gym", "Pool", "Concierge", "Rooftop", "AC", "Storage"],
            laundry="in_unit",
            utilities="included",
            pet_friendly=False,
            parking_included=True,
            available_date="2026-04-01",
            is_active=True,
            is_verified=False,
            risk_score=0.15,
            views=78,
        ),
        ListingModel(
            owner_id=landlord_user.id,
            title="Studio Apartment - Yonge & Eglinton",
            address="2300 Yonge St",
            city="Toronto",
            province="ON",
            postal_code="M4P 1E4",
            price=1750,
            beds=0,
            baths=1,
            sqft=450,
            property_type="apartment",
            description="Cozy studio in midtown Toronto. Perfect for young professionals. Steps to Eglinton subway station.",
            amenities=["AC", "Dishwasher"],
            laundry="shared",
            utilities="included",
            pet_friendly=False,
            parking_included=False,
            available_date="2026-02-15",
            is_active=True,
            is_verified=True,
            risk_score=0.08,
            views=21,
        ),
        ListingModel(
            owner_id=landlord_user.id,
            title="Renovated 2BR - The Annex",
            address="560 Bloor St W",
            city="Toronto",
            province="ON",
            postal_code="M5S 1Y6",
            price=2200,
            beds=2,
            baths=1,
            sqft=900,
            property_type="apartment",
            description="Newly renovated 2-bedroom in the charming Annex neighborhood. Hardwood floors, updated kitchen.",
            amenities=["Balcony", "Storage"],
            laundry="shared",
            utilities="not_included",
            pet_friendly=True,
            parking_included=False,
            available_date="2026-03-01",
            is_active=True,
            is_verified=False,
            risk_score=0.35,
            views=15,
        ),
        ListingModel(
            owner_id=landlord_user.id,
            title="Waterfront 1BR Condo",
            address="1 Queens Quay W",
            city="Toronto",
            province="ON",
            postal_code="M5J 2Y3",
            price=2600,
            beds=1,
            baths=1,
            sqft=650,
            property_type="condo",
            description="Beautiful waterfront condo with lake views. Direct access to the Harbourfront trails and ferry terminal.",
            amenities=["Gym", "Pool", "Concierge", "AC", "Balcony"],
            laundry="in_unit",
            utilities="not_included",
            pet_friendly=False,
            parking_included=True,
            available_date="2026-04-01",
            is_active=True,
            is_verified=True,
            risk_score=0.02,
            views=56,
        ),
    ]

    for listing in sample_listings:
        db.add(listing)

    await db.commit()
    print("Database seeded successfully!")


async def update_subscription_plans(db: AsyncSession):
    """Update existing subscription plans to ensure all features are enabled for every plan."""
    from sqlalchemy import select, update
    import json

    all_features_base = {
        "basic_analysis": True,
        "risk_score": True,
        "detailed_indicators": True,
        "image_analysis": True,
        "advanced_ml": True,
        "export_reports": True,
        "api_access": True,
        "custom_models": True,
        "message_analysis": True,
        "conversation_analysis": True,
        "address_validation": True,
        "price_anomaly": True,
        "xai_explanations": True,
        "url_analysis": True,
    }

    plan_overrides = {
        "free":       {"history": 1000, "support": "community"},
        "basic":      {"history": 50,  "support": "email"},
        "premium":    {"history": 200, "support": "priority"},
        "enterprise": {"history": -1,  "support": "24/7"},
    }

    result = await db.execute(select(SubscriptionPlanModel))
    plans = result.scalars().all()

    updated = 0
    for plan in plans:
        if plan.name in plan_overrides:
            new_features = {**all_features_base, **plan_overrides[plan.name]}
            if plan.features != new_features:
                plan.features = new_features
                updated += 1

    # Also update free plan scans_per_month to 1000
    for plan in plans:
        if plan.name == "free" and plan.scans_per_month != 1000:
            plan.scans_per_month = 1000
            updated += 1

    if updated:
        await db.commit()
        print(f"Updated features for {updated} subscription plan(s) — all abilities now enabled.")
    else:
        print("All subscription plans already have full features.")

    # Update all existing free-plan users to have 1000 scans_remaining
    user_result = await db.execute(
        select(UserModel).where(
            UserModel.subscription_plan == "free",
            UserModel.scans_remaining < 1000
        )
    )
    free_users = user_result.scalars().all()
    if free_users:
        for u in free_users:
            u.scans_remaining = 1000
        await db.commit()
        print(f"Updated scans_remaining to 1000 for {len(free_users)} free-plan user(s).")

