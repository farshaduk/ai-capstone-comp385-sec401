import { useState, useEffect } from 'react'
import LandlordLayout from '../../components/layouts/LandlordLayout'
import { landlordAPI } from '../../services/api'
import { Home, Plus, Eye, Users, ToggleLeft, ToggleRight, Trash2, Loader, MapPin, Edit, Clock, CheckCircle, XCircle, Ban } from 'lucide-react'
import toast from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'

const MyListings = () => {
  const [data, setData] = useState({ listings: [], stats: {} })
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  const fetchListings = async () => {
    setLoading(true)
    try {
      const { data: res } = await landlordAPI.getMyListings()
      setData(res)
    } catch (err) {
      console.error(err)
      toast.error('Failed to load listings')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchListings() }, [])

  const toggleActive = async (id, currentActive) => {
    try {
      await landlordAPI.updateListing(id, { is_active: !currentActive })
      toast.success(currentActive ? 'Listing deactivated' : 'Listing activated')
      fetchListings()
    } catch {
      toast.error('Failed to update listing')
    }
  }

  const handleDelete = async (id) => {
    if (!confirm('Delete this listing?')) return
    try {
      await landlordAPI.deleteListing(id)
      toast.success('Listing deleted')
      fetchListings()
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Failed to delete')
    }
  }

  const stats = data.stats || {}

  return (
    <LandlordLayout title="My Listings" subtitle="Manage your rental property listings">
      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {[
          { label: 'Total Listings', value: stats.total || 0, icon: Home },
          { label: 'Active', value: stats.active || 0, icon: ToggleRight },
          { label: 'Total Views', value: stats.total_views || 0, icon: Eye },
          { label: 'Total Applicants', value: stats.total_applicants || 0, icon: Users },
        ].map((s, i) => (
          <div key={i} className="card p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-landlord-100 dark:bg-landlord-900/30 rounded-xl">
                <s.icon className="h-5 w-5 text-landlord-600 dark:text-landlord-400" />
              </div>
              <div>
                <p className="text-xs text-surface-500">{s.label}</p>
                <p className="text-xl font-bold text-surface-900 dark:text-white">{s.value}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-surface-500 uppercase tracking-wider">
          {data.listings.length} Listing{data.listings.length !== 1 && 's'}
        </h3>
        <button onClick={() => navigate('/landlord/create-listing')} className="btn btn-primary btn-md">
          <Plus className="h-4 w-4" /> New Listing
        </button>
      </div>

      {loading ? (
        <div className="card p-16 text-center"><Loader className="h-8 w-8 animate-spin text-landlord-500 mx-auto mb-3" /><p className="text-surface-500">Loading...</p></div>
      ) : data.listings.length === 0 ? (
        <div className="card p-16 text-center">
          <Home className="h-12 w-12 text-surface-300 dark:text-surface-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-surface-900 dark:text-white mb-2">No listings yet</h3>
          <p className="text-sm text-surface-500 mb-4">Create your first rental listing to start receiving applications.</p>
          <button onClick={() => navigate('/landlord/create-listing')} className="btn btn-primary btn-md">
            <Plus className="h-4 w-4" /> Create Listing
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {data.listings.map(listing => (
            <div key={listing.id} className="card p-5 flex flex-col md:flex-row md:items-center gap-4">
              <div className="w-20 h-20 rounded-xl bg-surface-100 dark:bg-surface-800 flex items-center justify-center flex-shrink-0">
                <MapPin className="h-8 w-8 text-surface-300 dark:text-surface-600" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h4 className="font-semibold text-surface-900 dark:text-white truncate">{listing.title}</h4>
                  {listing.listing_status === 'pending_review' && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400 inline-flex items-center gap-1">
                      <Clock className="h-3 w-3" /> Pending Review
                    </span>
                  )}
                  {listing.listing_status === 'approved' && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 inline-flex items-center gap-1">
                      <CheckCircle className="h-3 w-3" /> Approved
                    </span>
                  )}
                  {listing.listing_status === 'rejected' && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 inline-flex items-center gap-1">
                      <XCircle className="h-3 w-3" /> Rejected
                    </span>
                  )}
                  {listing.listing_status === 'disabled' && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-surface-200 text-surface-600 dark:bg-surface-700 dark:text-surface-400 inline-flex items-center gap-1">
                      <Ban className="h-3 w-3" /> Disabled by Admin
                    </span>
                  )}
                  {(!listing.listing_status || listing.listing_status === 'approved') && (
                    <span className={`text-xs px-2 py-0.5 rounded-full ${listing.is_active ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : 'bg-surface-100 text-surface-500 dark:bg-surface-800'}`}>
                      {listing.is_active ? 'Active' : 'Inactive'}
                    </span>
                  )}
                </div>
                <p className="text-sm text-surface-500">{listing.address}, {listing.city}</p>
                {(listing.listing_status === 'rejected' || listing.listing_status === 'disabled') && listing.admin_notes && (
                  <p className="text-xs text-red-500 mt-1 flex items-center gap-1">
                    <XCircle className="h-3 w-3" /> {listing.admin_notes}
                  </p>
                )}
                {listing.listing_status === 'disabled' && !listing.admin_notes && (
                  <p className="text-xs text-surface-500 mt-1">This listing has been disabled by an administrator</p>
                )}
                {listing.listing_status === 'pending_review' && (
                  <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">Submitted for admin review — not yet visible to tenants</p>
                )}
                <div className="flex items-center gap-4 text-xs text-surface-400 mt-1">
                  <span>${listing.price?.toLocaleString()}/mo</span>
                  <span>{listing.beds} bed • {listing.baths} bath</span>
                  <span><Eye className="h-3 w-3 inline" /> {listing.views} views</span>
                  <span><Users className="h-3 w-3 inline" /> {listing.applicants} applicants</span>
                </div>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <button onClick={() => toggleActive(listing.id, listing.is_active)}
                  className="btn btn-secondary btn-sm text-xs" title={listing.is_active ? 'Deactivate' : 'Activate'}>
                  {listing.is_active ? <ToggleRight className="h-4 w-4 text-green-500" /> : <ToggleLeft className="h-4 w-4" />}
                </button>
                <button onClick={() => handleDelete(listing.id)} className="btn btn-secondary btn-sm text-xs text-red-500 hover:text-red-600">
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </LandlordLayout>
  )
}

export default MyListings
