import { useState, useEffect } from 'react'
import Layout from '../../components/Layout'
import { landlordAPI } from '../../services/api'
import toast from 'react-hot-toast'
import { 
  History, FileCheck, Users, Image, Shield, Loader, 
  ChevronDown, ChevronUp, RefreshCw, Search
} from 'lucide-react'

const VerificationHistory = () => {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [expandedId, setExpandedId] = useState(null)
  const [filterAction, setFilterAction] = useState('all')
  const [page, setPage] = useState(0)
  const pageSize = 20

  useEffect(() => {
    fetchHistory()
  }, [page])

  const fetchHistory = async () => {
    setLoading(true)
    try {
      const res = await landlordAPI.getVerificationHistory(page * pageSize, pageSize)
      setHistory(res.data.history || [])
    } catch (error) {
      toast.error('Failed to load history')
    } finally {
      setLoading(false)
    }
  }

  const getActionLabel = (action) => {
    const labels = {
      document_verified: 'Document Verification',
      tenant_verified: 'Tenant Verification',
      property_image_verified: 'Image Verification',
      property_images_verified: 'Bulk Image Verification',
      listing_images_verified: 'Listing Images Verification',
      cross_document_verified: 'Cross-Document Check',
      full_application_verified: 'Full Application Review'
    }
    return labels[action] || action.replace(/_/g, ' ')
  }

  const getActionIcon = (action) => {
    if (action.includes('document') || action.includes('cross')) return <FileCheck className="h-5 w-5" />
    if (action.includes('tenant') || action.includes('application')) return <Users className="h-5 w-5" />
    if (action.includes('image') || action.includes('listing')) return <Image className="h-5 w-5" />
    return <Shield className="h-5 w-5" />
  }

  const getActionColor = (action) => {
    if (action.includes('document')) return 'bg-blue-100 text-blue-600'
    if (action.includes('tenant') || action.includes('application')) return 'bg-purple-100 text-purple-600'
    if (action.includes('image')) return 'bg-green-100 text-green-600'
    return 'bg-gray-100 text-gray-600'
  }

  const filteredHistory = filterAction === 'all' 
    ? history 
    : history.filter(h => h.action.includes(filterAction))

  const uniqueActions = [...new Set(history.map(h => h.action))]

  return (
    <Layout title="Verification History">
      <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <History className="h-8 w-8 text-primary-600" />
            <div>
              <h3 className="text-2xl font-bold">Verification History</h3>
              <p className="text-gray-600">View all past document, tenant, and image verifications</p>
            </div>
          </div>
          <button onClick={fetchHistory} disabled={loading} className="btn-secondary">
            <RefreshCw className={`inline h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {/* Filter */}
        <div className="card">
          <div className="flex items-center space-x-3">
            <Search className="h-5 w-5 text-gray-400" />
            <select
              value={filterAction}
              onChange={(e) => setFilterAction(e.target.value)}
              className="input-field flex-1"
            >
              <option value="all">All Verification Types</option>
              <option value="document">Document Verifications</option>
              <option value="tenant">Tenant Verifications</option>
              <option value="image">Image Verifications</option>
              <option value="application">Full Applications</option>
              <option value="cross">Cross-Document Checks</option>
            </select>
          </div>
        </div>

        {/* History List */}
        {loading ? (
          <div className="flex justify-center p-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
          </div>
        ) : filteredHistory.length === 0 ? (
          <div className="card text-center py-12">
            <Shield className="h-16 w-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-xl font-bold text-gray-500 mb-2">No Verifications Yet</h3>
            <p className="text-gray-400">Your verification history will appear here once you start verifying documents or tenants.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredHistory.map((item) => (
              <div key={item.id} className="card hover:shadow-md transition-shadow">
                <div
                  className="flex items-center justify-between cursor-pointer"
                  onClick={() => setExpandedId(expandedId === item.id ? null : item.id)}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${getActionColor(item.action)}`}>
                      {getActionIcon(item.action)}
                    </div>
                    <div>
                      <p className="font-medium">{getActionLabel(item.action)}</p>
                      <p className="text-sm text-gray-500">{new Date(item.created_at).toLocaleString()}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    {item.details?.risk_level && (
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        item.details.risk_level === 'low' ? 'bg-green-100 text-green-700' :
                        item.details.risk_level === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                        item.details.risk_level === 'high' ? 'bg-orange-100 text-orange-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                        {item.details.risk_level}
                      </span>
                    )}
                    {item.details?.overall_risk != null && (
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        item.details.overall_risk < 0.3 ? 'bg-green-100 text-green-700' :
                        item.details.overall_risk < 0.6 ? 'bg-yellow-100 text-yellow-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                        {(item.details.overall_risk * 100).toFixed(0)}% risk
                      </span>
                    )}
                    {item.details?.risk_score != null && !item.details?.overall_risk && (
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        item.details.risk_score < 0.3 ? 'bg-green-100 text-green-700' :
                        item.details.risk_score < 0.6 ? 'bg-yellow-100 text-yellow-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                        {(item.details.risk_score * 100).toFixed(0)}% risk
                      </span>
                    )}
                    {expandedId === item.id ? (
                      <ChevronUp className="h-5 w-5 text-gray-400" />
                    ) : (
                      <ChevronDown className="h-5 w-5 text-gray-400" />
                    )}
                  </div>
                </div>

                {/* Expanded Details */}
                {expandedId === item.id && item.details && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <h5 className="text-sm font-medium text-gray-500 mb-2">Details</h5>
                    <div className="bg-gray-50 rounded-lg p-3">
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        {Object.entries(item.details).map(([key, value]) => (
                          <div key={key}>
                            <span className="text-gray-400 capitalize">{key.replace(/_/g, ' ')}:</span>{' '}
                            <span className="font-medium">
                              {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Pagination */}
        {history.length > 0 && (
          <div className="flex justify-center space-x-3">
            <button
              onClick={() => setPage(Math.max(0, page - 1))}
              disabled={page === 0}
              className="btn-secondary disabled:opacity-50"
            >
              Previous
            </button>
            <span className="px-4 py-2 text-gray-600">Page {page + 1}</span>
            <button
              onClick={() => setPage(page + 1)}
              disabled={history.length < pageSize}
              className="btn-secondary disabled:opacity-50"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </Layout>
  )
}

export default VerificationHistory
