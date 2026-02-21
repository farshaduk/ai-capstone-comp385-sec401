import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import TenantLayout from '../../components/layouts/TenantLayout'
import { renterAPI } from '../../services/api'
import {
  MapPin, Shield, Bed, Bath, Maximize, Calendar, DollarSign,
  Loader, ArrowLeft, Bookmark, Send, PawPrint, Car, Droplets,
  Shirt, CheckCircle, Eye
} from 'lucide-react'
import toast from 'react-hot-toast'

const riskColors = { very_low: 'badge-success', low: 'badge-success', medium: 'badge-warning', high: 'badge-danger', very_high: 'badge-danger' }
const riskLabels = { very_low: 'Very Low Risk', low: 'Low Risk', medium: 'Medium Risk', high: 'High Risk', very_high: 'Very High Risk' }
const riskFromScore = (s) => { if (s == null) return 'low'; if (s < 0.2) return 'very_low'; if (s < 0.4) return 'low'; if (s < 0.6) return 'medium'; if (s < 0.8) return 'high'; return 'very_high' }

const laundryLabels = { in_unit: 'In-Unit', in_building: 'In Building', none: 'None', not_included: 'Not Included' }
const utilityLabels = { included: 'Included', not_included: 'Not Included', partially_included: 'Partially Included' }

const ListingDetail = () => {
  const { id } = useParams()
  const navigate = useNavigate()
  const [listing, setListing] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const { data } = await renterAPI.getListing(id)
        setListing(data)
      } catch (err) {
        console.error(err)
        toast.error('Failed to load listing')
        navigate('/tenant/listings')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [id])

  const handleSave = async () => {
    try {
      await renterAPI.saveListing(listing.id)
      toast.success('Listing saved!')
    } catch (err) {
      if (err.response?.status === 400) toast.error('Already saved')
      else toast.error('Failed to save listing')
    }
  }

  const handleApply = async () => {
    try {
      await renterAPI.applyToListing(listing.id, 'I am interested in this listing.')
      toast.success('Application submitted!')
    } catch (err) {
      if (err.response?.status === 400) toast.error('Already applied')
      else toast.error('Failed to apply')
    }
  }

  if (loading) {
    return (
      <TenantLayout title="Listing Details" subtitle="Loading...">
        <div className="card p-16 text-center"><Loader className="h-8 w-8 animate-spin text-tenant-500 mx-auto mb-3" /><p className="text-surface-500">Loading listing...</p></div>
      </TenantLayout>
    )
  }

  if (!listing) return null

  const risk = riskFromScore(listing.risk_score)

  return (
    <TenantLayout title="Listing Details" subtitle={listing.title}>
      {/* Back button */}
      <button onClick={() => navigate('/tenant/listings')} className="btn btn-secondary btn-sm text-xs mb-6">
        <ArrowLeft className="h-4 w-4" /> Back to Listings
      </button>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Hero image placeholder */}
          <div className="card overflow-hidden">
            <div className="h-64 bg-gradient-to-br from-surface-100 to-surface-200 dark:from-surface-800 dark:to-surface-700 relative flex items-center justify-center">
              <MapPin className="h-16 w-16 text-surface-300 dark:text-surface-600" />
              <div className="absolute top-4 left-4 flex items-center gap-2">
                <span className={riskColors[risk]}>{riskLabels[risk]}</span>
                {listing.is_verified && (
                  <span className="bg-white/90 dark:bg-surface-800/90 backdrop-blur px-2 py-1 rounded-lg flex items-center gap-1">
                    <Shield className="h-3 w-3 text-tenant-500" /><span className="text-xs font-semibold text-tenant-600 dark:text-tenant-400">Verified</span>
                  </span>
                )}
              </div>
              <div className="absolute bottom-4 right-4 bg-white/90 dark:bg-surface-800/90 backdrop-blur px-3 py-1 rounded-lg flex items-center gap-1">
                <Eye className="h-3 w-3 text-surface-500" /><span className="text-xs text-surface-600 dark:text-surface-400">{listing.views} views</span>
              </div>
            </div>
          </div>

          {/* Title & address */}
          <div className="card p-6">
            <h2 className="text-xl font-bold text-surface-900 dark:text-white mb-2">{listing.title}</h2>
            <p className="text-sm text-surface-500 dark:text-surface-400 flex items-center gap-1 mb-4">
              <MapPin className="h-4 w-4" />{listing.address}, {listing.city}, {listing.province} {listing.postal_code}
            </p>
            <div className="flex items-center gap-6 text-sm text-surface-600 dark:text-surface-400">
              <span className="flex items-center gap-1"><Bed className="h-4 w-4" /> {listing.beds === 0 ? 'Studio' : `${listing.beds} Bed`}</span>
              <span className="flex items-center gap-1"><Bath className="h-4 w-4" /> {listing.baths} Bath</span>
              {listing.sqft && <span className="flex items-center gap-1"><Maximize className="h-4 w-4" /> {listing.sqft} sqft</span>}
              <span className="capitalize flex items-center gap-1">{listing.property_type}</span>
            </div>
          </div>

          {/* Description */}
          {listing.description && (
            <div className="card p-6">
              <h3 className="font-semibold text-surface-900 dark:text-white mb-3">Description</h3>
              <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed whitespace-pre-line">{listing.description}</p>
            </div>
          )}

          {/* Details grid */}
          <div className="card p-6">
            <h3 className="font-semibold text-surface-900 dark:text-white mb-4">Details</h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              <div className="flex items-center gap-3 text-sm">
                <div className="p-2 bg-tenant-100 dark:bg-tenant-900/30 rounded-xl"><Shirt className="h-4 w-4 text-tenant-600 dark:text-tenant-400" /></div>
                <div><p className="text-xs text-surface-500">Laundry</p><p className="font-medium text-surface-900 dark:text-white">{laundryLabels[listing.laundry] || listing.laundry}</p></div>
              </div>
              <div className="flex items-center gap-3 text-sm">
                <div className="p-2 bg-tenant-100 dark:bg-tenant-900/30 rounded-xl"><Droplets className="h-4 w-4 text-tenant-600 dark:text-tenant-400" /></div>
                <div><p className="text-xs text-surface-500">Utilities</p><p className="font-medium text-surface-900 dark:text-white">{utilityLabels[listing.utilities] || listing.utilities}</p></div>
              </div>
              <div className="flex items-center gap-3 text-sm">
                <div className="p-2 bg-tenant-100 dark:bg-tenant-900/30 rounded-xl"><PawPrint className="h-4 w-4 text-tenant-600 dark:text-tenant-400" /></div>
                <div><p className="text-xs text-surface-500">Pets</p><p className="font-medium text-surface-900 dark:text-white">{listing.pet_friendly ? 'Allowed' : 'Not Allowed'}</p></div>
              </div>
              <div className="flex items-center gap-3 text-sm">
                <div className="p-2 bg-tenant-100 dark:bg-tenant-900/30 rounded-xl"><Car className="h-4 w-4 text-tenant-600 dark:text-tenant-400" /></div>
                <div><p className="text-xs text-surface-500">Parking</p><p className="font-medium text-surface-900 dark:text-white">{listing.parking_included ? 'Included' : 'Not Included'}</p></div>
              </div>
              {listing.available_date && (
                <div className="flex items-center gap-3 text-sm">
                  <div className="p-2 bg-tenant-100 dark:bg-tenant-900/30 rounded-xl"><Calendar className="h-4 w-4 text-tenant-600 dark:text-tenant-400" /></div>
                  <div><p className="text-xs text-surface-500">Available</p><p className="font-medium text-surface-900 dark:text-white">{listing.available_date}</p></div>
                </div>
              )}
            </div>
          </div>

          {/* Amenities */}
          {listing.amenities && listing.amenities.length > 0 && (
            <div className="card p-6">
              <h3 className="font-semibold text-surface-900 dark:text-white mb-3">Amenities</h3>
              <div className="flex flex-wrap gap-2">
                {listing.amenities.map((a, i) => (
                  <span key={i} className="inline-flex items-center gap-1 bg-surface-100 dark:bg-surface-800 text-surface-700 dark:text-surface-300 rounded-full px-3 py-1 text-xs font-medium">
                    <CheckCircle className="h-3 w-3 text-tenant-500" />{a}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Price card */}
          <div className="card p-6 sticky top-6">
            <div className="text-center mb-6">
              <p className="text-3xl font-bold text-surface-900 dark:text-white">${listing.price?.toLocaleString()}<span className="text-base font-normal text-surface-500">/mo</span></p>
            </div>
            <div className="space-y-3">
              <button onClick={handleApply} className="btn btn-primary btn-md w-full">
                <Send className="h-4 w-4" /> Apply Now
              </button>
              <button onClick={handleSave} className="btn btn-secondary btn-md w-full">
                <Bookmark className="h-4 w-4" /> Save Listing
              </button>
            </div>
            <div className="mt-6 pt-4 border-t border-surface-200 dark:border-surface-700 space-y-2 text-xs text-surface-500">
              {listing.created_at && (
                <p>Listed: {new Date(listing.created_at).toLocaleDateString()}</p>
              )}
              <p>Views: {listing.views}</p>
              <p>Risk Score: {(listing.risk_score * 100).toFixed(0)}%</p>
            </div>
          </div>
        </div>
      </div>
    </TenantLayout>
  )
}

export default ListingDetail
