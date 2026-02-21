import { useState, useEffect } from 'react'
import AdminLayout from '../../components/layouts/AdminLayout'
import { adminAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  Home, CheckCircle, XCircle, Clock, Filter,
  RefreshCw, Loader, ChevronDown, ChevronUp, Shield,
  MapPin, DollarSign, User, Eye, AlertTriangle, Ban
} from 'lucide-react'

const statusColors = {
  pending_review: { bg: 'bg-amber-100 dark:bg-amber-900/30', text: 'text-amber-700 dark:text-amber-300', icon: Clock },
  approved: { bg: 'bg-emerald-100 dark:bg-emerald-900/30', text: 'text-emerald-700 dark:text-emerald-300', icon: CheckCircle },
  rejected: { bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-300', icon: XCircle },
  disabled: { bg: 'bg-surface-200 dark:bg-surface-700', text: 'text-surface-600 dark:text-surface-400', icon: Ban },
}

const typeLabels = {
  apartment: 'Apartment', condo: 'Condo', house: 'House', basement: 'Basement', room: 'Room',
}

const ListingApprovalNew = () => {
  const [listings, setListings] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [statusFilter, setStatusFilter] = useState('pending_review')
  const [typeFilter, setTypeFilter] = useState('')
  const [expandedId, setExpandedId] = useState(null)
  const [reviewingId, setReviewingId] = useState(null)
  const [notesInput, setNotesInput] = useState({})

  useEffect(() => {
    fetchAll()
  }, [statusFilter, typeFilter])

  const fetchAll = async () => {
    setLoading(true)
    try {
      const params = {}
      if (statusFilter) params.listing_status = statusFilter
      if (typeFilter) params.property_type = typeFilter

      const [ls, st] = await Promise.all([
        adminAPI.getAdminListings(params),
        adminAPI.getAdminListingStats()
      ])
      setListings(ls.data)
      setStats(st.data)
    } catch (e) {
      toast.error('Failed to load listings')
    } finally {
      setLoading(false)
    }
  }

  const handleReview = async (id, status) => {
    setReviewingId(id)
    try {
      await adminAPI.reviewListing(id, { status, admin_notes: notesInput[id] || '' })
      toast.success(`Listing #${id} ${status}`)
      setNotesInput(prev => ({ ...prev, [id]: '' }))
      await fetchAll()
    } catch (e) {
      toast.error(`Review failed: ${e.response?.data?.detail || e.message}`)
    } finally {
      setReviewingId(null)
    }
  }

  return (
    <AdminLayout title="Listing Approval" subtitle="Review and approve landlord listings before they go live">
      <div className="space-y-6 animate-fade-in">

        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            {[
              { label: 'Pending Review', value: stats.pending_review ?? 0, icon: Clock, color: 'text-amber-600', bg: 'bg-amber-50 dark:bg-amber-900/20' },
              { label: 'Approved', value: stats.approved ?? 0, icon: CheckCircle, color: 'text-emerald-600', bg: 'bg-emerald-50 dark:bg-emerald-900/20' },
              { label: 'Rejected', value: stats.rejected ?? 0, icon: XCircle, color: 'text-red-600', bg: 'bg-red-50 dark:bg-red-900/20' },
              { label: 'Disabled', value: stats.disabled ?? 0, icon: Ban, color: 'text-surface-500', bg: 'bg-surface-100 dark:bg-surface-800' },
              { label: 'Total Listings', value: stats.total_listings ?? 0, icon: Home, color: 'text-blue-600', bg: 'bg-blue-50 dark:bg-blue-900/20' },
            ].map((s, i) => (
              <div key={i} className={`card p-4 ${s.bg}`}>
                <s.icon className={`h-5 w-5 ${s.color} mb-1`} />
                <p className={`text-2xl font-display font-bold ${s.color}`}>{s.value}</p>
                <p className="text-xs text-surface-500 dark:text-surface-400">{s.label}</p>
              </div>
            ))}
          </div>
        )}

        {/* Info Banner */}
        <div className="p-4 rounded-xl bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800">
          <div className="flex items-start gap-3">
            <Shield className="h-5 w-5 text-primary-600 dark:text-primary-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-semibold text-primary-800 dark:text-primary-200">Listing Moderation</p>
              <p className="text-xs text-primary-600 dark:text-primary-400 mt-1">
                Landlord listings are <strong>not visible to tenants</strong> until approved by an admin.
                Approved listings are automatically activated and marked as verified.
                Rejected listings remain hidden from tenants.
              </p>
            </div>
          </div>
        </div>

        {/* Filters + Refresh */}
        <div className="card p-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-surface-400" />
              <span className="text-sm font-medium text-surface-600 dark:text-surface-300">Filters:</span>
            </div>

            <select
              value={statusFilter}
              onChange={e => setStatusFilter(e.target.value)}
              className="input-field w-auto text-sm py-1.5 px-3"
            >
              <option value="">All Statuses</option>
              <option value="pending_review">Pending Review</option>
              <option value="approved">Approved</option>
              <option value="rejected">Rejected</option>
              <option value="disabled">Disabled</option>
            </select>

            <select
              value={typeFilter}
              onChange={e => setTypeFilter(e.target.value)}
              className="input-field w-auto text-sm py-1.5 px-3"
            >
              <option value="">All Types</option>
              <option value="apartment">Apartment</option>
              <option value="condo">Condo</option>
              <option value="house">House</option>
              <option value="basement">Basement</option>
              <option value="room">Room</option>
            </select>

            <div className="flex-1" />

            <button onClick={fetchAll} className="btn btn-sm btn-secondary">
              <RefreshCw className="h-3.5 w-3.5" /> Refresh
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="card overflow-hidden">
          {loading ? (
            <div className="flex justify-center py-16">
              <Loader className="h-8 w-8 animate-spin text-primary-600" />
            </div>
          ) : listings.length === 0 ? (
            <div className="text-center py-16">
              <Home className="h-12 w-12 text-surface-300 mx-auto mb-3" />
              <p className="text-surface-500 dark:text-surface-400 text-sm">No listings found for this filter</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-surface-50 dark:bg-surface-800/50 border-b border-surface-200 dark:border-surface-700">
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">Listing</th>
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">Owner</th>
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">Type</th>
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">Price</th>
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">Status</th>
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">Submitted</th>
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-surface-100 dark:divide-surface-800">
                  {listings.map(l => {
                    const sc = statusColors[l.listing_status] || statusColors.pending_review
                    const StatusIcon = sc.icon
                    const isExpanded = expandedId === l.id
                    const isReviewing = reviewingId === l.id

                    return (
                      <>
                        <tr
                          key={l.id}
                          className="hover:bg-surface-50/80 dark:hover:bg-surface-800/50 transition-colors cursor-pointer"
                          onClick={() => setExpandedId(isExpanded ? null : l.id)}
                        >
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-3">
                              <div className="w-10 h-10 rounded-lg bg-surface-100 dark:bg-surface-800 flex items-center justify-center flex-shrink-0">
                                <MapPin className="h-4 w-4 text-surface-400" />
                              </div>
                              <div className="min-w-0">
                                <p className="text-sm font-semibold text-surface-900 dark:text-white truncate max-w-[200px]">{l.title}</p>
                                <p className="text-xs text-surface-500 truncate max-w-[200px]">{l.address}, {l.city}</p>
                              </div>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <div>
                              <p className="text-sm text-surface-700 dark:text-surface-300">{l.owner_name}</p>
                              <p className="text-xs text-surface-400">{l.owner_email}</p>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-surface-100 text-surface-600 dark:bg-surface-800 dark:text-surface-300">
                              {typeLabels[l.property_type] || l.property_type}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <span className="text-sm font-semibold text-surface-900 dark:text-white">${l.price?.toLocaleString()}</span>
                            <span className="text-xs text-surface-500">/mo</span>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium ${sc.bg} ${sc.text}`}>
                              <StatusIcon className="h-3 w-3" /> {l.listing_status === 'pending_review' ? 'Pending' : l.listing_status}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-xs text-surface-500">
                            {l.created_at ? new Date(l.created_at).toLocaleDateString() : '—'}
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2" onClick={e => e.stopPropagation()}>
                              {l.listing_status === 'pending_review' && (
                                <>
                                  <button
                                    onClick={() => handleReview(l.id, 'approved')}
                                    disabled={isReviewing}
                                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium
                                      bg-emerald-100 text-emerald-700 hover:bg-emerald-200
                                      dark:bg-emerald-900/30 dark:text-emerald-300 dark:hover:bg-emerald-900/50
                                      disabled:opacity-50 transition-colors"
                                  >
                                    <CheckCircle className="h-3 w-3" /> Approve
                                  </button>
                                  <button
                                    onClick={() => handleReview(l.id, 'rejected')}
                                    disabled={isReviewing}
                                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium
                                      bg-red-100 text-red-700 hover:bg-red-200
                                      dark:bg-red-900/30 dark:text-red-300 dark:hover:bg-red-900/50
                                      disabled:opacity-50 transition-colors"
                                  >
                                    <XCircle className="h-3 w-3" /> Reject
                                  </button>
                                </>
                              )}
                              {l.listing_status === 'approved' && (
                                <button
                                  onClick={() => handleReview(l.id, 'disabled')}
                                  disabled={isReviewing}
                                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium
                                    bg-surface-200 text-surface-700 hover:bg-surface-300
                                    dark:bg-surface-700 dark:text-surface-300 dark:hover:bg-surface-600
                                    disabled:opacity-50 transition-colors"
                                >
                                  <Ban className="h-3 w-3" /> Disable
                                </button>
                              )}
                              {(l.listing_status === 'disabled' || l.listing_status === 'rejected') && (
                                <button
                                  onClick={() => handleReview(l.id, 'approved')}
                                  disabled={isReviewing}
                                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium
                                    bg-emerald-100 text-emerald-700 hover:bg-emerald-200
                                    dark:bg-emerald-900/30 dark:text-emerald-300 dark:hover:bg-emerald-900/50
                                    disabled:opacity-50 transition-colors"
                                >
                                  <CheckCircle className="h-3 w-3" /> Re-approve
                                </button>
                              )}
                              {isExpanded
                                ? <ChevronUp className="h-4 w-4 text-surface-400" />
                                : <ChevronDown className="h-4 w-4 text-surface-400" />
                              }
                            </div>
                          </td>
                        </tr>

                        {/* Expanded detail row */}
                        {isExpanded && (
                          <tr key={`${l.id}-detail`} className="bg-surface-50/50 dark:bg-surface-800/30">
                            <td colSpan={7} className="px-6 py-5">
                              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                {/* Property Details */}
                                <div>
                                  <p className="text-xs font-semibold text-surface-500 uppercase mb-2">Property Details</p>
                                  <div className="space-y-1.5 text-sm">
                                    <p className="text-surface-700 dark:text-surface-300"><span className="text-surface-500">Beds:</span> {l.beds === 0 ? 'Studio' : l.beds}</p>
                                    <p className="text-surface-700 dark:text-surface-300"><span className="text-surface-500">Baths:</span> {l.baths}</p>
                                    {l.sqft && <p className="text-surface-700 dark:text-surface-300"><span className="text-surface-500">Sqft:</span> {l.sqft}</p>}
                                    <p className="text-surface-700 dark:text-surface-300"><span className="text-surface-500">Laundry:</span> {l.laundry}</p>
                                    <p className="text-surface-700 dark:text-surface-300"><span className="text-surface-500">Utilities:</span> {l.utilities}</p>
                                    <p className="text-surface-700 dark:text-surface-300"><span className="text-surface-500">Pet Friendly:</span> {l.pet_friendly ? 'Yes' : 'No'}</p>
                                    <p className="text-surface-700 dark:text-surface-300"><span className="text-surface-500">Parking:</span> {l.parking_included ? 'Included' : 'Not included'}</p>
                                    {l.available_date && <p className="text-surface-700 dark:text-surface-300"><span className="text-surface-500">Available:</span> {l.available_date}</p>}
                                  </div>
                                </div>

                                {/* Description + Amenities */}
                                <div>
                                  <p className="text-xs font-semibold text-surface-500 uppercase mb-2">Description</p>
                                  <p className="text-sm text-surface-700 dark:text-surface-300 mb-3">
                                    {l.description || <span className="italic text-surface-400">No description provided</span>}
                                  </p>
                                  {l.amenities?.length > 0 && (
                                    <>
                                      <p className="text-xs font-semibold text-surface-500 uppercase mb-2">Amenities</p>
                                      <div className="flex flex-wrap gap-1.5">
                                        {l.amenities.map((a, i) => (
                                          <span key={i} className="text-xs px-2 py-0.5 rounded-full bg-primary-50 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300">{a}</span>
                                        ))}
                                      </div>
                                    </>
                                  )}
                                </div>

                                {/* Review Info + Admin Notes */}
                                <div>
                                  <p className="text-xs font-semibold text-surface-500 uppercase mb-2">Review Info</p>
                                  <div className="space-y-1.5 text-sm">
                                    <p className="text-surface-700 dark:text-surface-300 flex items-center gap-1.5">
                                      <Eye className="h-3.5 w-3.5 text-surface-400" /> {l.views} views
                                    </p>
                                    <p className="text-surface-700 dark:text-surface-300">
                                      <span className="text-surface-500">Risk Score:</span> {l.risk_score?.toFixed(2) ?? '0.00'}
                                    </p>
                                    {l.reviewed_by && (
                                      <p className="text-surface-700 dark:text-surface-300 flex items-center gap-1.5">
                                        <Shield className="h-3.5 w-3.5 text-primary-500" /> Reviewed by Admin #{l.reviewed_by}
                                      </p>
                                    )}
                                    {l.reviewed_at && (
                                      <p className="text-surface-700 dark:text-surface-300">
                                        <span className="text-surface-500">Reviewed:</span> {new Date(l.reviewed_at).toLocaleString()}
                                      </p>
                                    )}
                                  </div>

                                  {/* Admin notes input for pending listings */}
                                  {l.listing_status === 'pending_review' && (
                                    <div className="mt-3">
                                      <label className="text-xs font-semibold text-surface-500 uppercase block mb-1">Admin Notes (optional)</label>
                                      <textarea
                                        value={notesInput[l.id] || ''}
                                        onChange={e => setNotesInput(prev => ({ ...prev, [l.id]: e.target.value }))}
                                        placeholder="Reason for approval/rejection..."
                                        rows={2}
                                        className="input-field text-sm w-full"
                                        onClick={e => e.stopPropagation()}
                                      />
                                    </div>
                                  )}

                                  {/* Show existing admin notes */}
                                  {l.admin_notes && l.listing_status !== 'pending_review' && (
                                    <div className="mt-3 p-3 rounded-lg bg-surface-100 dark:bg-surface-800">
                                      <p className="text-xs font-semibold text-surface-500 uppercase mb-1">Admin Notes</p>
                                      <p className="text-sm text-surface-700 dark:text-surface-300">{l.admin_notes}</p>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Footer info */}
        <div className="text-xs text-surface-400 text-center">
          Showing {listings.length} listings • Tenants only see approved listings
        </div>
      </div>
    </AdminLayout>
  )
}

export default ListingApprovalNew
