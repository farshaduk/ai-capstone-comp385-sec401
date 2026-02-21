import { useState, useEffect } from 'react'
import LandlordLayout from '../../components/layouts/LandlordLayout'
import { landlordAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  Clock, FileCheck, Users, Image, Shield, Calendar, Loader, ChevronDown
} from 'lucide-react'

const VerificationHistoryNew = () => {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState(null)

  useEffect(() => { fetchHistory() }, [])

  const fetchHistory = async () => {
    try {
      const res = await landlordAPI.getVerificationHistory(0, 100)
      setHistory(res.data.history || res.data || [])
    } catch { toast.error('Failed to load verification history') }
    finally { setLoading(false) }
  }

  const getActionLabel = (a) => {
    const m = {
      document_verified: 'Document Verification', tenant_verified: 'Tenant Screening',
      property_image_verified: 'Image Verification', property_images_verified: 'Bulk Image Check',
      cross_document_verified: 'Cross-Document', full_application_verified: 'Full Application',
    }
    return m[a] || a
  }

  const getActionIcon = (a) => {
    if (a.includes('document')) return FileCheck
    if (a.includes('tenant') || a.includes('application')) return Users
    if (a.includes('image')) return Image
    return Shield
  }

  const getActionColor = (a) => {
    if (a.includes('document')) return 'bg-blue-500'
    if (a.includes('tenant') || a.includes('application')) return 'bg-purple-500'
    if (a.includes('image')) return 'bg-emerald-500'
    return 'bg-landlord-500'
  }

  return (
    <LandlordLayout title="Verification History" subtitle="Past verification records">
      <div className="space-y-4">
        {loading ? (
          <div className="flex justify-center py-16"><Loader className="h-8 w-8 animate-spin text-landlord-600" /></div>
        ) : history.length > 0 ? (
          history.map((item) => {
            const Icon = getActionIcon(item.action)
            const isExpanded = expanded === item.id
            return (
              <div key={item.id} className="card p-4">
                <div className="flex items-center justify-between cursor-pointer" onClick={() => setExpanded(isExpanded ? null : item.id)}>
                  <div className="flex items-center gap-3">
                    <div className={`w-9 h-9 ${getActionColor(item.action)} rounded-lg flex items-center justify-center`}>
                      <Icon className="h-4 w-4 text-white" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-surface-900 dark:text-white">{getActionLabel(item.action)}</p>
                      <p className="text-xs text-surface-500 dark:text-surface-400 flex items-center gap-1">
                        <Calendar className="h-3 w-3" /> {new Date(item.created_at).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {item.details?.risk_level && (
                      <span className={`badge ${
                        item.details.risk_level === 'low' ? 'badge-success' :
                        item.details.risk_level === 'medium' ? 'badge-warning' : 'badge-danger'
                      }`}>{item.details.risk_level}</span>
                    )}
                    <ChevronDown className={`h-4 w-4 text-surface-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                  </div>
                </div>

                {isExpanded && item.details && (
                  <div className="mt-4 pt-4 border-t border-surface-200 dark:border-surface-700">
                    <pre className="text-xs text-surface-600 dark:text-surface-400 bg-surface-50 dark:bg-surface-700 rounded-xl p-4 overflow-x-auto whitespace-pre-wrap">
                      {JSON.stringify(item.details, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )
          })
        ) : (
          <div className="card text-center py-16">
            <Clock className="h-16 w-16 text-surface-300 dark:text-surface-600 mx-auto mb-4" />
            <h3 className="text-lg font-display font-semibold mb-2 text-surface-600 dark:text-surface-300">No verification history</h3>
            <p className="text-surface-500 dark:text-surface-400">Start verifying documents and tenants to see your history here.</p>
          </div>
        )}
      </div>
    </LandlordLayout>
  )
}

export default VerificationHistoryNew
