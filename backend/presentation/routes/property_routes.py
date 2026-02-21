"""
Property management routes: Listings, Applications, Leases, Messages, Profile, Address verification.
These routes power the CRUD features of the tenant and landlord dashboards.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from infrastructure.database import (
    get_db, UserModel, ListingModel, SavedListingModel,
    ApplicationModel, ApplicationMessageModel, LeaseModel, MessageModel
)
from presentation.dependencies import get_current_user, get_current_renter, get_current_landlord
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import re
from application.use_cases.address_validation_engine import address_validation_engine


router = APIRouter()


# ======================== SCHEMAS ========================

class ListingCreate(BaseModel):
    title: str
    address: str
    city: str = "Toronto"
    province: str = "ON"
    postal_code: str = ""
    price: float
    beds: int = 1
    baths: float = 1
    sqft: Optional[int] = None
    property_type: str = "apartment"
    description: str = ""
    amenities: list = []
    laundry: str = "in_unit"
    utilities: str = "not_included"
    pet_friendly: bool = False
    parking_included: bool = False
    available_date: str = ""


class ListingUpdate(BaseModel):
    title: Optional[str] = None
    price: Optional[float] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    amenities: Optional[list] = None


class ApplicationCreate(BaseModel):
    listing_id: int
    message: str = ""


class ApplicationStatusUpdate(BaseModel):
    status: str  # approved, rejected, viewing_scheduled


class AppMessageCreate(BaseModel):
    text: str


class LeaseCreate(BaseModel):
    listing_id: int
    tenant_id: int
    start_date: str
    end_date: str
    rent: float
    deposit: float = 0.0


class MessageCreate(BaseModel):
    receiver_id: int
    text: str


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    bio: Optional[str] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


class SaveListingRequest(BaseModel):
    listing_id: int
    notes: str = ""


class AddressCheckRequest(BaseModel):
    address: str
    city: str = "Toronto"
    postal_code: str = ""


# ======================== LISTINGS ========================

@router.get("/listings")
async def browse_listings(
    search: str = "",
    min_price: float = 0,
    max_price: float = 999999,
    beds: Optional[int] = None,
    property_type: str = "",
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Browse all active listings (public for authenticated users)"""
    query = select(ListingModel).where(
        and_(ListingModel.is_active == True, ListingModel.listing_status == "approved")
    )

    if search:
        search_filter = f"%{search}%"
        query = query.where(
            or_(
                ListingModel.title.ilike(search_filter),
                ListingModel.address.ilike(search_filter),
                ListingModel.city.ilike(search_filter),
                ListingModel.description.ilike(search_filter),
            )
        )
    if min_price > 0:
        query = query.where(ListingModel.price >= min_price)
    if max_price < 999999:
        query = query.where(ListingModel.price <= max_price)
    if beds is not None:
        query = query.where(ListingModel.beds == beds)
    if property_type:
        query = query.where(ListingModel.property_type == property_type)

    total_q = select(func.count()).select_from(query.subquery())
    total = (await db.execute(total_q)).scalar() or 0

    query = query.order_by(ListingModel.created_at.desc()).offset(skip).limit(limit)
    rows = (await db.execute(query)).scalars().all()

    return {
        "total": total,
        "listings": [_listing_dict(r) for r in rows],
    }


@router.get("/listings/{listing_id}")
async def get_listing(listing_id: int, db: AsyncSession = Depends(get_db)):
    listing = (await db.execute(select(ListingModel).where(ListingModel.id == listing_id))).scalar_one_or_none()
    if not listing:
        raise HTTPException(404, "Listing not found")
    # Increment views
    listing.views = (listing.views or 0) + 1
    await db.commit()
    return _listing_dict(listing)


@router.post("/landlord/listings", status_code=201)
async def create_listing(
    data: ListingCreate,
    user: UserModel = Depends(get_current_landlord),
    db: AsyncSession = Depends(get_db),
):
    listing = ListingModel(
        owner_id=user.id,
        title=data.title,
        address=data.address,
        city=data.city,
        province=data.province,
        postal_code=data.postal_code,
        price=data.price,
        beds=data.beds,
        baths=data.baths,
        sqft=data.sqft,
        property_type=data.property_type,
        description=data.description,
        amenities=data.amenities,
        laundry=data.laundry,
        utilities=data.utilities,
        pet_friendly=data.pet_friendly,
        parking_included=data.parking_included,
        available_date=data.available_date,
        is_active=False,
        listing_status="pending_review",
    )
    db.add(listing)
    await db.commit()
    await db.refresh(listing)
    result = _listing_dict(listing)
    result["message"] = "Listing submitted for admin review. It will be visible to tenants once approved."
    return result


@router.get("/landlord/listings")
async def my_listings(
    user: UserModel = Depends(get_current_landlord),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(ListingModel).where(ListingModel.owner_id == user.id).order_by(ListingModel.created_at.desc())
    )).scalars().all()

    # Count applicants per listing
    results = []
    for r in rows:
        d = _listing_dict(r)
        cnt = (await db.execute(
            select(func.count()).where(ApplicationModel.listing_id == r.id)
        )).scalar() or 0
        d["applicants"] = cnt
        results.append(d)

    total_active = sum(1 for r in rows if r.is_active)
    total_views = sum(r.views or 0 for r in rows)
    total_apps = sum(r["applicants"] for r in results)

    return {
        "listings": results,
        "stats": {
            "total": len(rows),
            "active": total_active,
            "inactive": len(rows) - total_active,
            "total_views": total_views,
            "total_applicants": total_apps,
        }
    }


@router.patch("/landlord/listings/{listing_id}")
async def update_listing(
    listing_id: int,
    data: ListingUpdate,
    user: UserModel = Depends(get_current_landlord),
    db: AsyncSession = Depends(get_db),
):
    listing = (await db.execute(
        select(ListingModel).where(and_(ListingModel.id == listing_id, ListingModel.owner_id == user.id))
    )).scalar_one_or_none()
    if not listing:
        raise HTTPException(404, "Listing not found")
    for k, v in data.dict(exclude_unset=True).items():
        setattr(listing, k, v)
    await db.commit()
    await db.refresh(listing)
    return _listing_dict(listing)


@router.delete("/landlord/listings/{listing_id}")
async def delete_listing(
    listing_id: int,
    user: UserModel = Depends(get_current_landlord),
    db: AsyncSession = Depends(get_db),
):
    listing = (await db.execute(
        select(ListingModel).where(and_(ListingModel.id == listing_id, ListingModel.owner_id == user.id))
    )).scalar_one_or_none()
    if not listing:
        raise HTTPException(404, "Listing not found")

    # ── Restrict: block delete if active or pending leases exist ──
    active_leases = (await db.execute(
        select(func.count()).where(and_(
            LeaseModel.listing_id == listing_id,
            LeaseModel.status.in_(["active", "pending_signature"]),
        ))
    )).scalar() or 0
    if active_leases > 0:
        raise HTTPException(
            400,
            "Cannot delete listing — it has active or pending leases. "
            "Expire or cancel the leases first."
        )

    # ── Cascade cleanup: remove orphaned child records ──
    # Delete application messages for this listing's applications
    app_ids_q = select(ApplicationModel.id).where(ApplicationModel.listing_id == listing_id)
    await db.execute(
        delete(ApplicationMessageModel).where(
            ApplicationMessageModel.application_id.in_(app_ids_q)
        )
    )
    # Delete applications
    await db.execute(
        delete(ApplicationModel).where(ApplicationModel.listing_id == listing_id)
    )
    # Delete saved listings
    await db.execute(
        delete(SavedListingModel).where(SavedListingModel.listing_id == listing_id)
    )
    # Delete expired leases (active ones blocked above)
    await db.execute(
        delete(LeaseModel).where(and_(
            LeaseModel.listing_id == listing_id,
            LeaseModel.status.notin_(["active", "pending_signature"]),
        ))
    )

    await db.delete(listing)
    await db.commit()
    return {"ok": True}


# ======================== SAVED LISTINGS ========================

@router.get("/renter/saved-listings")
async def get_saved_listings(
    user: UserModel = Depends(get_current_renter),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(SavedListingModel).where(SavedListingModel.user_id == user.id).order_by(SavedListingModel.created_at.desc())
    )).scalars().all()

    results = []
    for s in rows:
        listing = (await db.execute(select(ListingModel).where(ListingModel.id == s.listing_id))).scalar_one_or_none()
        if listing:
            d = _listing_dict(listing)
            d["saved_id"] = s.id
            d["notes"] = s.notes
            d["saved_at"] = s.created_at.isoformat() if s.created_at else None
            results.append(d)
    return {"saved": results}


@router.post("/renter/saved-listings")
async def save_listing(
    data: SaveListingRequest,
    user: UserModel = Depends(get_current_renter),
    db: AsyncSession = Depends(get_db),
):
    # Check duplicate
    existing = (await db.execute(
        select(SavedListingModel).where(and_(
            SavedListingModel.user_id == user.id,
            SavedListingModel.listing_id == data.listing_id
        ))
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(400, "Already saved")
    s = SavedListingModel(user_id=user.id, listing_id=data.listing_id, notes=data.notes)
    db.add(s)
    await db.commit()
    return {"ok": True, "id": s.id}


@router.delete("/renter/saved-listings/{saved_id}")
async def unsave_listing(
    saved_id: int,
    user: UserModel = Depends(get_current_renter),
    db: AsyncSession = Depends(get_db),
):
    s = (await db.execute(
        select(SavedListingModel).where(and_(SavedListingModel.id == saved_id, SavedListingModel.user_id == user.id))
    )).scalar_one_or_none()
    if not s:
        raise HTTPException(404, "Not found")
    await db.delete(s)
    await db.commit()
    return {"ok": True}


# ======================== APPLICATIONS ========================

@router.post("/renter/applications")
async def apply_to_listing(
    data: ApplicationCreate,
    user: UserModel = Depends(get_current_renter),
    db: AsyncSession = Depends(get_db),
):
    listing = (await db.execute(select(ListingModel).where(ListingModel.id == data.listing_id))).scalar_one_or_none()
    if not listing:
        raise HTTPException(404, "Listing not found")
    # Check duplicate
    existing = (await db.execute(
        select(ApplicationModel).where(and_(
            ApplicationModel.listing_id == data.listing_id,
            ApplicationModel.applicant_id == user.id,
        ))
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(400, "Already applied")
    app = ApplicationModel(
        listing_id=data.listing_id,
        applicant_id=user.id,
        landlord_id=listing.owner_id,
        message=data.message,
    )
    db.add(app)
    await db.commit()
    await db.refresh(app)
    return _application_dict(app, listing, user)


@router.get("/renter/applications")
async def my_applications(
    user: UserModel = Depends(get_current_renter),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(ApplicationModel).where(ApplicationModel.applicant_id == user.id).order_by(ApplicationModel.created_at.desc())
    )).scalars().all()

    results = []
    for app in rows:
        listing = (await db.execute(select(ListingModel).where(ListingModel.id == app.listing_id))).scalar_one_or_none()
        landlord = (await db.execute(select(UserModel).where(UserModel.id == app.landlord_id))).scalar_one_or_none()
        d = _application_dict(app, listing, landlord)
        # Get last message
        last_msg = (await db.execute(
            select(ApplicationMessageModel).where(ApplicationMessageModel.application_id == app.id)
            .order_by(ApplicationMessageModel.created_at.desc()).limit(1)
        )).scalar_one_or_none()
        if last_msg:
            sender = (await db.execute(select(UserModel).where(UserModel.id == last_msg.sender_id))).scalar_one_or_none()
            d["last_message"] = {
                "from": "landlord" if last_msg.sender_id == app.landlord_id else "you",
                "text": last_msg.text,
                "at": last_msg.created_at.isoformat() if last_msg.created_at else None,
            }
        # Unread count
        unread = (await db.execute(
            select(func.count()).where(and_(
                ApplicationMessageModel.application_id == app.id,
                ApplicationMessageModel.sender_id != user.id,
            ))
        )).scalar() or 0
        d["unread"] = unread
        results.append(d)
    return {"applications": results}


@router.get("/landlord/applicants")
async def landlord_applicants(
    status_filter: str = "",
    user: UserModel = Depends(get_current_landlord),
    db: AsyncSession = Depends(get_db),
):
    query = select(ApplicationModel).where(ApplicationModel.landlord_id == user.id)
    if status_filter:
        query = query.where(ApplicationModel.status == status_filter)
    query = query.order_by(ApplicationModel.created_at.desc())
    rows = (await db.execute(query)).scalars().all()

    results = []
    for app in rows:
        listing = (await db.execute(select(ListingModel).where(ListingModel.id == app.listing_id))).scalar_one_or_none()
        applicant = (await db.execute(select(UserModel).where(UserModel.id == app.applicant_id))).scalar_one_or_none()
        d = {
            "id": app.id,
            "listing": listing.title if listing else "Unknown",
            "listing_id": app.listing_id,
            "applicant_name": applicant.full_name if applicant else "Unknown",
            "applicant_email": applicant.email if applicant else "",
            "status": app.status,
            "message": app.message,
            "applied_at": app.created_at.isoformat() if app.created_at else None,
        }
        results.append(d)

    counts = {}
    for s in ["pending", "approved", "rejected", "viewing_scheduled"]:
        counts[s] = sum(1 for r in results if r["status"] == s)

    return {"applicants": results, "counts": counts}


@router.patch("/landlord/applicants/{app_id}")
async def update_application_status(
    app_id: int,
    data: ApplicationStatusUpdate,
    user: UserModel = Depends(get_current_landlord),
    db: AsyncSession = Depends(get_db),
):
    app = (await db.execute(
        select(ApplicationModel).where(and_(ApplicationModel.id == app_id, ApplicationModel.landlord_id == user.id))
    )).scalar_one_or_none()
    if not app:
        raise HTTPException(404, "Application not found")
    app.status = data.status

    # ── Approval cascades ──
    if data.status == "approved":
        # 1. Deactivate the listing so it no longer appears in tenant browse
        listing = (await db.execute(
            select(ListingModel).where(ListingModel.id == app.listing_id)
        )).scalar_one_or_none()
        if listing:
            listing.is_active = False

        # 2. Auto-reject all other pending/viewing_scheduled applications for this listing
        await db.execute(
            update(ApplicationModel).where(and_(
                ApplicationModel.listing_id == app.listing_id,
                ApplicationModel.id != app.id,
                ApplicationModel.status.in_(["pending", "viewing_scheduled"]),
            )).values(status="rejected")
        )

        # 3. Auto-create a draft lease linking landlord, tenant, and listing
        lease = LeaseModel(
            listing_id=app.listing_id,
            landlord_id=user.id,
            tenant_id=app.applicant_id,
            start_date=datetime.utcnow().strftime("%Y-%m-%d"),
            end_date="",  # landlord fills in via Leases page
            rent=listing.price if listing else 0,
            deposit=0,
            status="pending_signature",
        )
        db.add(lease)

    await db.commit()
    return {"ok": True, "status": data.status}


@router.post("/applications/{app_id}/messages")
async def send_app_message(
    app_id: int,
    data: AppMessageCreate,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    app = (await db.execute(select(ApplicationModel).where(ApplicationModel.id == app_id))).scalar_one_or_none()
    if not app:
        raise HTTPException(404, "Application not found")
    if user.id not in (app.applicant_id, app.landlord_id):
        raise HTTPException(403, "Not authorized")
    msg = ApplicationMessageModel(application_id=app_id, sender_id=user.id, text=data.text)
    db.add(msg)
    await db.commit()
    return {"ok": True}


@router.get("/applications/{app_id}/messages")
async def get_app_messages(
    app_id: int,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    app = (await db.execute(select(ApplicationModel).where(ApplicationModel.id == app_id))).scalar_one_or_none()
    if not app:
        raise HTTPException(404)
    if user.id not in (app.applicant_id, app.landlord_id):
        raise HTTPException(403)
    rows = (await db.execute(
        select(ApplicationMessageModel).where(ApplicationMessageModel.application_id == app_id)
        .order_by(ApplicationMessageModel.created_at.asc())
    )).scalars().all()
    return {"messages": [
        {"id": m.id, "sender_id": m.sender_id, "text": m.text,
         "from_me": m.sender_id == user.id,
         "at": m.created_at.isoformat() if m.created_at else None}
        for m in rows
    ]}


# ======================== LEASES ========================

@router.get("/landlord/leases")
async def landlord_leases(
    user: UserModel = Depends(get_current_landlord),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(LeaseModel).where(LeaseModel.landlord_id == user.id).order_by(LeaseModel.created_at.desc())
    )).scalars().all()

    results = []
    for lease in rows:
        listing = (await db.execute(select(ListingModel).where(ListingModel.id == lease.listing_id))).scalar_one_or_none()
        tenant = (await db.execute(select(UserModel).where(UserModel.id == lease.tenant_id))).scalar_one_or_none()
        results.append({
            "id": lease.id,
            "property": listing.title if listing else "Unknown",
            "address": listing.address if listing else "",
            "tenant": tenant.full_name if tenant else "Unknown",
            "tenant_email": tenant.email if tenant else "",
            "start_date": lease.start_date,
            "end_date": lease.end_date,
            "rent": lease.rent,
            "deposit": lease.deposit,
            "status": lease.status,
            "created_at": lease.created_at.isoformat() if lease.created_at else None,
        })

    active = sum(1 for r in results if r["status"] in ("active", "pending_signature"))
    monthly_revenue = sum(r["rent"] for r in results if r["status"] == "active")

    return {
        "leases": results,
        "stats": {
            "total": len(results),
            "active": active,
            "monthly_revenue": monthly_revenue,
        },
    }


@router.post("/landlord/leases", status_code=201)
async def create_lease(
    data: LeaseCreate,
    user: UserModel = Depends(get_current_landlord),
    db: AsyncSession = Depends(get_db),
):
    lease = LeaseModel(
        listing_id=data.listing_id,
        landlord_id=user.id,
        tenant_id=data.tenant_id,
        start_date=data.start_date,
        end_date=data.end_date,
        rent=data.rent,
        deposit=data.deposit,
    )
    db.add(lease)
    await db.commit()
    return {"ok": True, "id": lease.id}


# ======================== MESSAGES ========================

@router.get("/messages/contacts")
async def get_contacts(
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get unique contacts the user has messaged with"""
    # Find all user ids this user exchanged messages with
    sent = (await db.execute(
        select(MessageModel.receiver_id).where(MessageModel.sender_id == user.id).distinct()
    )).scalars().all()
    received = (await db.execute(
        select(MessageModel.sender_id).where(MessageModel.receiver_id == user.id).distinct()
    )).scalars().all()
    contact_ids = set(sent) | set(received)

    contacts = []
    for cid in contact_ids:
        contact = (await db.execute(select(UserModel).where(UserModel.id == cid))).scalar_one_or_none()
        if not contact:
            continue
        # Last message
        last = (await db.execute(
            select(MessageModel).where(
                or_(
                    and_(MessageModel.sender_id == user.id, MessageModel.receiver_id == cid),
                    and_(MessageModel.sender_id == cid, MessageModel.receiver_id == user.id),
                )
            ).order_by(MessageModel.created_at.desc()).limit(1)
        )).scalar_one_or_none()
        # Unread count
        unread = (await db.execute(
            select(func.count()).where(and_(
                MessageModel.sender_id == cid,
                MessageModel.receiver_id == user.id,
                MessageModel.is_read == False,
            ))
        )).scalar() or 0

        contacts.append({
            "id": contact.id,
            "name": contact.full_name,
            "email": contact.email,
            "role": contact.role,
            "unread": unread,
            "last_message": last.text if last else "",
            "last_time": last.created_at.isoformat() if last and last.created_at else None,
        })

    contacts.sort(key=lambda c: c["last_time"] or "", reverse=True)
    return {"contacts": contacts}


@router.get("/messages/{other_id}")
async def get_conversation(
    other_id: int,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(MessageModel).where(
            or_(
                and_(MessageModel.sender_id == user.id, MessageModel.receiver_id == other_id),
                and_(MessageModel.sender_id == other_id, MessageModel.receiver_id == user.id),
            )
        ).order_by(MessageModel.created_at.asc())
    )).scalars().all()

    # Mark as read
    await db.execute(
        update(MessageModel).where(and_(
            MessageModel.sender_id == other_id,
            MessageModel.receiver_id == user.id,
            MessageModel.is_read == False,
        )).values(is_read=True)
    )
    await db.commit()

    return {"messages": [
        {"id": m.id, "from_me": m.sender_id == user.id, "text": m.text,
         "at": m.created_at.isoformat() if m.created_at else None, "read": m.is_read}
        for m in rows
    ]}


@router.post("/messages")
async def send_message(
    data: MessageCreate,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    receiver = (await db.execute(select(UserModel).where(UserModel.id == data.receiver_id))).scalar_one_or_none()
    if not receiver:
        raise HTTPException(404, "Recipient not found")
    msg = MessageModel(sender_id=user.id, receiver_id=data.receiver_id, text=data.text)
    db.add(msg)
    await db.commit()
    return {"ok": True, "id": msg.id}


# ======================== PROFILE ========================

@router.get("/auth/profile")
async def get_profile(user: UserModel = Depends(get_current_user)):
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "subscription_plan": user.subscription_plan,
        "scans_remaining": user.scans_remaining,
        "is_active": user.is_active,
        "phone": user.phone or "",
        "address": user.address or "",
        "bio": user.bio or "",
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.patch("/auth/profile")
async def update_profile(
    data: ProfileUpdate,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if data.full_name is not None:
        user.full_name = data.full_name
    if data.phone is not None:
        user.phone = data.phone
    if data.address is not None:
        user.address = data.address
    if data.bio is not None:
        user.bio = data.bio
    await db.commit()
    return {"ok": True}


@router.post("/auth/change-password")
async def change_password(
    data: PasswordChange,
    user: UserModel = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from application.use_cases.auth_use_cases import AuthUseCases
    if not AuthUseCases.verify_password(data.current_password, user.hashed_password):
        raise HTTPException(400, "Current password is incorrect")
    user.hashed_password = AuthUseCases.hash_password(data.new_password)
    await db.commit()
    return {"ok": True}


# ======================== ADDRESS VERIFICATION ========================

@router.post("/renter/verify-address")
async def verify_address(
    data: AddressCheckRequest,
    user: UserModel = Depends(get_current_renter),
    db: AsyncSession = Depends(get_db),
):
    """Verify if a rental address is real using the AddressValidationEngine + DB scam cross-check."""
    full_address = f"{data.address}, {data.city}, ON {data.postal_code}".strip()

    # ── Run the full validation engine ──────────────────────────────
    engine_result = await address_validation_engine.validate(full_address)

    # ── DB scam cross-check ─────────────────────────────────────────
    scam_matches = 0
    if data.address.strip():
        scam_q = select(func.count()).where(
            and_(
                ListingModel.address.ilike(f"%{data.address.strip()}%"),
                ListingModel.risk_score > 0.6
            )
        )
        scam_matches = (await db.execute(scam_q)).scalar() or 0

    # ── Postal code quick check ─────────────────────────────────────
    postal_valid = bool(
        re.match(r'^[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d$', data.postal_code.strip())
    ) if data.postal_code.strip() else False

    geocoded = engine_result.latitude is not None and engine_result.longitude is not None

    # ── Build user-facing checks list ───────────────────────────────
    checks = [
        {
            "name": "Postal Code Validation",
            "status": "pass" if postal_valid else "fail",
            "detail": f"Postal code {'is valid for Ontario' if postal_valid else 'is invalid or missing'}.",
        },
        {
            "name": "Geocoding Verification",
            "status": "pass" if geocoded else ("warn" if postal_valid else "fail"),
            "detail": (
                f"Address resolved to: {engine_result.normalized_address}"
                if geocoded
                else "Address could not be geocoded — may not exist."
            ),
        },
        {
            "name": "Residential Area Check",
            "status": "pass" if engine_result.is_residential else "warn",
            "detail": (
                "Address is in a residential area."
                if engine_result.is_residential
                else "Address may not be in a residential zone — verify independently."
            ),
        },
        {
            "name": "Scam Address Database",
            "status": "warn" if scam_matches > 0 else "pass",
            "detail": (
                f"Found {scam_matches} suspicious listing(s) at this address."
                if scam_matches
                else "No scam reports for this address."
            ),
        },
    ]

    # Add engine-detected suspicious indicators as additional checks
    for ind in engine_result.indicators:
        severity_map = {1: "pass", 2: "warn", 3: "warn", 4: "fail", 5: "fail"}
        checks.append({
            "name": ind["description"][:60],
            "status": severity_map.get(ind.get("severity", 2), "warn"),
            "detail": "; ".join(ind.get("evidence", [])),
        })

    # ── Overall risk level ──────────────────────────────────────────
    fails = sum(1 for c in checks if c["status"] == "fail")
    warns = sum(1 for c in checks if c["status"] == "warn")
    risk_level = "high" if fails >= 2 else "medium" if fails >= 1 or warns >= 2 else "low"

    return {
        "full_address": full_address,
        "risk_level": risk_level,
        "is_valid": postal_valid and geocoded,
        "geocoded": geocoded,
        "latitude": engine_result.latitude,
        "longitude": engine_result.longitude,
        "display_name": engine_result.normalized_address or "",
        "confidence": engine_result.confidence,
        "engine_status": engine_result.status.value,
        "is_residential": engine_result.is_residential,
        "explanation": engine_result.explanation,
        "checks": checks,
    }


# ======================== LANDLORD ANALYTICS ========================

@router.get("/landlord/analytics")
async def landlord_analytics(
    user: UserModel = Depends(get_current_landlord),
    db: AsyncSession = Depends(get_db),
):
    listings = (await db.execute(
        select(ListingModel).where(ListingModel.owner_id == user.id)
    )).scalars().all()

    total_views = sum(l.views or 0 for l in listings)
    active_count = sum(1 for l in listings if l.is_active)
    total_apps = 0
    listing_perf = []
    for l in listings:
        apps = (await db.execute(
            select(func.count()).where(ApplicationModel.listing_id == l.id)
        )).scalar() or 0
        total_apps += apps
        listing_perf.append({
            "name": l.title,
            "views": l.views or 0,
            "applications": apps,
            "occupancy": 100 if not l.is_active else 0,
            "revenue": l.price if not l.is_active else 0,
        })

    # Revenue from active leases
    leases = (await db.execute(
        select(LeaseModel).where(and_(LeaseModel.landlord_id == user.id, LeaseModel.status == "active"))
    )).scalars().all()
    monthly_revenue = sum(l.rent for l in leases)

    occupancy_rate = round((len(leases) / max(len(listings), 1)) * 100)

    return {
        "stats": {
            "total_views": total_views,
            "total_applications": total_apps,
            "monthly_revenue": monthly_revenue,
            "occupancy_rate": occupancy_rate,
            "active_listings": active_count,
        },
        "listing_performance": listing_perf,
    }


# ======================== HELPERS ========================

def _listing_dict(l: ListingModel) -> dict:
    return {
        "id": l.id,
        "owner_id": l.owner_id,
        "title": l.title,
        "address": l.address,
        "city": l.city,
        "province": l.province,
        "postal_code": l.postal_code,
        "price": l.price,
        "beds": l.beds,
        "baths": l.baths,
        "sqft": l.sqft,
        "property_type": l.property_type,
        "description": l.description,
        "amenities": l.amenities or [],
        "laundry": l.laundry,
        "utilities": l.utilities,
        "pet_friendly": l.pet_friendly,
        "parking_included": l.parking_included,
        "available_date": l.available_date,
        "is_active": l.is_active,
        "is_verified": l.is_verified,
        "listing_status": l.listing_status or "pending_review",
        "admin_notes": l.admin_notes,
        "reviewed_by": l.reviewed_by,
        "reviewed_at": l.reviewed_at.isoformat() if l.reviewed_at else None,
        "risk_score": l.risk_score,
        "views": l.views or 0,
        "created_at": l.created_at.isoformat() if l.created_at else None,
    }


def _application_dict(app: ApplicationModel, listing, user_or_landlord) -> dict:
    return {
        "id": app.id,
        "listing_id": app.listing_id,
        "listing": listing.title if listing else "Unknown",
        "address": listing.address if listing else "",
        "price": listing.price if listing else 0,
        "status": app.status,
        "message": app.message,
        "landlord": user_or_landlord.full_name if user_or_landlord else "Unknown",
        "applied_at": app.created_at.isoformat() if app.created_at else None,
    }
