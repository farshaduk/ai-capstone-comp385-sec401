import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import TenantLayout from '../../components/layouts/TenantLayout'
import { renterAPI } from '../../services/api'
import { Search, MapPin, Shield, Eye, SlidersHorizontal, Loader, Bookmark } from 'lucide-react'
import toast from 'react-hot-toast'

const riskColors = { very_low: 'badge-success', low: 'badge-success', medium: 'badge-warning', high: 'badge-danger', very_high: 'badge-danger' }
const riskLabels = { very_low: 'Very Low Risk', low: 'Low Risk', medium: 'Medium Risk', high: 'High Risk', very_high: 'Very High Risk' }
const riskFromScore = (s) => { if (s == null) return 'low'; if (s < 0.2) return 'very_low'; if (s < 0.4) return 'low'; if (s < 0.6) return 'medium'; if (s < 0.8) return 'high'; return 'very_high' }

const Listings = () => {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [maxPrice, setMaxPrice] = useState(5000)
  const [bedsFilter, setBedsFilter] = useState('')
  const [propertyType, setPropertyType] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  const [listings, setListings] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)

  const fetchListings = async () => {
    setLoading(true)
    try {
      const params = { max_price: maxPrice }
      if (search) params.search = search
      if (bedsFilter) params.beds = parseInt(bedsFilter)
      if (propertyType) params.property_type = propertyType
      const { data } = await renterAPI.browseListings(params)
      setListings(data.listings || [])
      setTotal(data.total || 0)
    } catch (err) {
      console.error(err)
      toast.error('Failed to load listings')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchListings() }, [])

  const handleSearch = (e) => { e.preventDefault(); fetchListings() }

  const handleSave = async (listingId) => {
    try {
      await renterAPI.saveListing(listingId)
      toast.success('Listing saved!')
    } catch (err) {
      if (err.response?.status === 400) toast.error('Already saved')
      else toast.error('Failed to save listing')
    }
  }

  const handleApply = async (listingId) => {
    try {
      await renterAPI.applyToListing(listingId, 'I am interested in this listing.')
      toast.success('Application submitted!')
    } catch (err) {
      if (err.response?.status === 400) toast.error('Already applied')
      else toast.error('Failed to apply')
    }
  }

  return (
    <TenantLayout title="Browse Listings" subtitle="Discover verified rental listings in your area">
      <form onSubmit={handleSearch} className="card p-4 mb-6">
        <div className="flex flex-col lg:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
            <input type="text" placeholder="Search by title, address, or neighborhood..." value={search} onChange={e => setSearch(e.target.value)} className="input-field pl-12" />
          </div>
          <button type="submit" className="btn btn-primary btn-md"><Search className="h-4 w-4" /> Search</button>
          <button type="button" onClick={() => setShowFilters(!showFilters)} className={`btn ${showFilters ? 'btn-primary' : 'btn-secondary'} btn-md`}>
            <SlidersHorizontal className="h-4 w-4" /> Filters
          </button>
        </div>
        {showFilters && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4 pt-4 border-t border-surface-200 dark:border-surface-700">
            <div><label className="input-label">Bedrooms</label>
              <select value={bedsFilter} onChange={e => setBedsFilter(e.target.value)} className="input-field">
                <option value="">All</option><option value="0">Studio</option><option value="1">1 Bed</option><option value="2">2 Beds</option><option value="3">3+ Beds</option>
              </select></div>
            <div><label className="input-label">Property Type</label>
              <select value={propertyType} onChange={e => setPropertyType(e.target.value)} className="input-field">
                <option value="">All</option><option value="apartment">Apartment</option><option value="condo">Condo</option><option value="house">House</option><option value="basement">Basement</option><option value="room">Room</option>
              </select></div>
            <div><label className="input-label">Max Price</label>
              <input type="range" min="500" max="5000" step="100" value={maxPrice} onChange={e => setMaxPrice(parseInt(e.target.value))} className="w-full mt-2" />
              <p className="text-xs text-surface-500 mt-1">Up to ${maxPrice.toLocaleString()}/mo</p></div>
          </div>
        )}
      </form>

      <p className="text-sm text-surface-500 dark:text-surface-400 mb-4">
        Showing <span className="font-semibold text-surface-900 dark:text-white">{listings.length}</span> of {total} listings
      </p>

      {loading ? (
        <div className="card p-16 text-center"><Loader className="h-8 w-8 animate-spin text-tenant-500 mx-auto mb-3" /><p className="text-surface-500">Loading listings...</p></div>
      ) : listings.length === 0 ? (
        <div className="card p-16 text-center"><Search className="h-12 w-12 text-surface-300 dark:text-surface-600 mx-auto mb-4" /><h3 className="text-lg font-semibold text-surface-900 dark:text-white mb-2">No listings found</h3>
          <p className="text-sm text-surface-500 dark:text-surface-400">{total === 0 ? 'No listings have been posted yet. Check back soon!' : 'Try adjusting your filters or search terms.'}</p></div>
      ) : (
        <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
          {listings.map(listing => {
            const risk = riskFromScore(listing.risk_score)
            return (
              <div key={listing.id} className="card-hover overflow-hidden group cursor-pointer" onClick={() => navigate(`/tenant/listings/${listing.id}`)}>
                <div className="h-48 bg-gradient-to-br from-surface-100 to-surface-200 dark:from-surface-800 dark:to-surface-700 relative">
                  <div className="absolute inset-0 flex items-center justify-center"><MapPin className="h-12 w-12 text-surface-300 dark:text-surface-600" /></div>
                  <div className="absolute top-3 left-3"><span className={riskColors[risk]}>{riskLabels[risk]}</span></div>
                  {listing.is_verified && (
                    <div className="absolute top-3 right-3 bg-white/90 dark:bg-surface-800/90 backdrop-blur px-2 py-1 rounded-lg flex items-center gap-1">
                      <Shield className="h-3 w-3 text-tenant-500" /><span className="text-xs font-semibold text-tenant-600 dark:text-tenant-400">Verified</span>
                    </div>
                  )}
                  <div className="absolute bottom-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity flex gap-2" onClick={e => e.stopPropagation()}>
                    <button onClick={() => handleSave(listing.id)} className="bg-white/90 dark:bg-surface-800/90 backdrop-blur p-2 rounded-lg hover:bg-white" title="Save">
                      <Bookmark className="h-4 w-4 text-surface-600 dark:text-surface-300" />
                    </button>
                  </div>
                </div>
                <div className="p-5">
                  <h3 className="font-semibold text-surface-900 dark:text-white text-sm leading-tight mb-2">{listing.title}</h3>
                  <p className="text-xs text-surface-500 dark:text-surface-400 flex items-center gap-1 mb-3"><MapPin className="h-3 w-3" />{listing.address}, {listing.city}</p>
                  <div className="flex items-center gap-3 text-xs text-surface-500 dark:text-surface-400 mb-4">
                    <span>{listing.beds === 0 ? 'Studio' : `${listing.beds} Bed`}</span><span>•</span><span>{listing.baths} Bath</span>
                    {listing.sqft && <><span>•</span><span>{listing.sqft} sqft</span></>}
                  </div>
                  <div className="flex items-center justify-between pt-3 border-t border-surface-100 dark:border-surface-700" onClick={e => e.stopPropagation()}>
                    <div><span className="text-lg font-bold text-surface-900 dark:text-white">${listing.price?.toLocaleString()}</span><span className="text-xs text-surface-500">/mo</span></div>
                    <button onClick={() => handleApply(listing.id)} className="btn btn-primary btn-sm text-xs">Apply</button>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </TenantLayout>
  )
}

export default Listings
