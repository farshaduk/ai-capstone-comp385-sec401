import { useState, useEffect } from 'react'
import TenantLayout from '../../components/layouts/TenantLayout'
import { renterAPI } from '../../services/api'
import { Bookmark, MapPin, Trash2, Loader, Search } from 'lucide-react'
import toast from 'react-hot-toast'

const SavedListings = () => {
  const [saved, setSaved] = useState([])
  const [loading, setLoading] = useState(true)

  const fetchSaved = async () => {
    setLoading(true)
    try {
      const { data } = await renterAPI.getSavedListings()
      setSaved(data.saved || [])
    } catch (err) {
      console.error(err)
      toast.error('Failed to load saved listings')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchSaved() }, [])

  const handleRemove = async (savedId) => {
    try {
      await renterAPI.unsaveListing(savedId)
      setSaved(prev => prev.filter(s => s.saved_id !== savedId))
      toast.success('Listing removed from saved')
    } catch {
      toast.error('Failed to remove')
    }
  }

  return (
    <TenantLayout title="Saved Listings" subtitle="Your bookmarked properties for quick reference">
      {loading ? (
        <div className="card p-16 text-center">
          <Loader className="h-8 w-8 animate-spin text-tenant-500 mx-auto mb-3" />
          <p className="text-surface-500">Loading saved listings...</p>
        </div>
      ) : saved.length === 0 ? (
        <div className="card p-16 text-center">
          <Bookmark className="h-12 w-12 text-surface-300 dark:text-surface-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-surface-900 dark:text-white mb-2">No saved listings</h3>
          <p className="text-sm text-surface-500 dark:text-surface-400">
            Browse listings and click the bookmark icon to save properties you are interested in.
          </p>
        </div>
      ) : (
        <>
          <p className="text-sm text-surface-500 dark:text-surface-400 mb-4">
            <span className="font-semibold text-surface-900 dark:text-white">{saved.length}</span> saved listing{saved.length !== 1 && 's'}
          </p>

          <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
            {saved.map(item => (
              <div key={item.saved_id} className="card-hover overflow-hidden">
                <div className="h-40 bg-gradient-to-br from-surface-100 to-surface-200 dark:from-surface-800 dark:to-surface-700 relative flex items-center justify-center">
                  <MapPin className="h-10 w-10 text-surface-300 dark:text-surface-600" />
                </div>
                <div className="p-5">
                  <h3 className="font-semibold text-surface-900 dark:text-white text-sm mb-1">{item.title}</h3>
                  <p className="text-xs text-surface-500 dark:text-surface-400 flex items-center gap-1 mb-3">
                    <MapPin className="h-3 w-3" />{item.address}, {item.city}
                  </p>
                  <div className="flex items-center gap-3 text-xs text-surface-500 mb-3">
                    <span>{item.beds === 0 ? 'Studio' : `${item.beds} Bed`}</span>
                    <span>•</span><span>{item.baths} Bath</span>
                    {item.sqft && <><span>•</span><span>{item.sqft} sqft</span></>}
                  </div>
                  <div className="flex items-center justify-between pt-3 border-t border-surface-100 dark:border-surface-700">
                    <span className="text-lg font-bold text-surface-900 dark:text-white">${item.price?.toLocaleString()}<span className="text-xs font-normal text-surface-500">/mo</span></span>
                    <button onClick={() => handleRemove(item.saved_id)}
                      className="btn btn-secondary btn-sm text-xs text-red-500 hover:text-red-600">
                      <Trash2 className="h-3.5 w-3.5" /> Remove
                    </button>
                  </div>
                  {item.notes && (
                    <p className="text-xs text-surface-400 mt-2 italic">Note: {item.notes}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </TenantLayout>
  )
}

export default SavedListings
