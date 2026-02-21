import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import LandlordLayout from '../../components/layouts/LandlordLayout'
import { landlordAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  FileCheck, Users, Image, Shield, TrendingUp, ArrowRight,
  Loader, AlertTriangle, CheckCircle, BarChart3, Clock
} from 'lucide-react'

const LandlordDashboardNew = () => {
  const [stats, setStats] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => { fetchData() }, [])

  const fetchData = async () => {
    try {
      const [statsRes, historyRes] = await Promise.allSettled([
        landlordAPI.getDashboardStats(),
        landlordAPI.getVerificationHistory(0, 5),
      ])
      if (statsRes.status === 'fulfilled') setStats(statsRes.value.data.stats)
      if (historyRes.status === 'fulfilled') setHistory(historyRes.value.data.history || [])
    } catch {
      toast.error('Failed to load dashboard data')
    } finally {
      setLoading(false)
    }
  }

  const quickLinks = [
    { path: '/landlord/documents', icon: FileCheck, label: 'Verify Document', desc: 'Upload & verify tenant documents', color: 'bg-blue-500' },
    { path: '/landlord/tenants', icon: Users, label: 'Screen Tenant', desc: 'Full tenant application review', color: 'bg-purple-500' },
    { path: '/landlord/property-images', icon: Image, label: 'Verify Images', desc: 'Check property listing photos', color: 'bg-emerald-500' },
    { path: '/landlord/history', icon: TrendingUp, label: 'View History', desc: 'Past verification results', color: 'bg-amber-500' },
  ]

  const getActionLabel = (a) => {
    const m = {
      document_verified: 'Document Check', tenant_verified: 'Tenant Screen',
      property_image_verified: 'Image Check', property_images_verified: 'Bulk Image Check',
      cross_document_verified: 'Cross-Document', full_application_verified: 'Full Application',
    }
    return m[a] || a
  }

  const getActionIcon = (a) => {
    if (a.includes('document')) return <FileCheck className="h-4 w-4" />
    if (a.includes('tenant') || a.includes('application')) return <Users className="h-4 w-4" />
    if (a.includes('image')) return <Image className="h-4 w-4" />
    return <Shield className="h-4 w-4" />
  }

  if (loading) {
    return (
      <LandlordLayout title="Dashboard">
        <div className="flex justify-center py-20"><Loader className="h-8 w-8 animate-spin text-landlord-600" /></div>
      </LandlordLayout>
    )
  }

  return (
    <LandlordLayout title="Dashboard" subtitle="Verification overview and quick actions">
      <div className="space-y-6 animate-fade-in">
        {/* Stats */}
        {stats && (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { icon: FileCheck, label: 'Documents', value: stats.total_document_verifications, color: 'text-blue-600' },
              { icon: Users, label: 'Tenants', value: stats.total_tenant_verifications, color: 'text-purple-600' },
              { icon: Image, label: 'Images', value: stats.total_image_verifications, color: 'text-emerald-600' },
              { icon: Shield, label: 'Total', value: stats.total_verifications, color: 'text-landlord-600' },
            ].map((s, i) => (
              <div key={i} className="card p-5">
                <s.icon className={`h-7 w-7 ${s.color} mb-2`} />
                <p className="text-2xl font-display font-bold text-surface-900 dark:text-white">{s.value}</p>
                <p className="text-xs text-surface-500 dark:text-surface-400 font-medium">{s.label} Verified</p>
              </div>
            ))}
          </div>
        )}

        {/* Quick Actions */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {quickLinks.map((link) => (
            <Link key={link.path} to={link.path} className="card p-5 group hover:shadow-elevated transition-all duration-200">
              <div className={`w-10 h-10 ${link.color} rounded-xl flex items-center justify-center mb-3`}>
                <link.icon className="h-5 w-5 text-white" />
              </div>
              <h3 className="text-sm font-semibold text-surface-900 dark:text-white mb-1">{link.label}</h3>
              <p className="text-xs text-surface-500 dark:text-surface-400">{link.desc}</p>
              <ArrowRight className="h-4 w-4 text-surface-300 group-hover:text-landlord-600 mt-2 transition-colors" />
            </Link>
          ))}
        </div>

        {/* Recent Verifications */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white flex items-center gap-2">
              <Clock className="h-5 w-5 text-surface-400" /> Recent Verifications
            </h3>
            <Link to="/landlord/history" className="text-sm text-landlord-600 hover:text-landlord-700 font-medium flex items-center gap-1">
              View All <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
          {history.length > 0 ? (
            <div className="space-y-2">
              {history.map((item) => (
                <div key={item.id} className="flex items-center justify-between p-3 rounded-xl bg-surface-50 dark:bg-surface-700/50 hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-landlord-100 dark:bg-landlord-900/30 rounded-lg flex items-center justify-center text-landlord-600">
                      {getActionIcon(item.action)}
                    </div>
                    <div>
                      <p className="text-sm font-medium text-surface-900 dark:text-white">{getActionLabel(item.action)}</p>
                      <p className="text-xs text-surface-500 dark:text-surface-400">{new Date(item.created_at).toLocaleString()}</p>
                    </div>
                  </div>
                  {item.details?.risk_level && (
                    <span className={`badge ${
                      item.details.risk_level === 'low' ? 'badge-success' :
                      item.details.risk_level === 'medium' ? 'badge-warning' : 'badge-danger'
                    }`}>
                      {item.details.risk_level}
                    </span>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-surface-500 dark:text-surface-400">
              <Shield className="h-12 w-12 mx-auto mb-3 text-surface-300 dark:text-surface-600" />
              <p>No verifications yet. Start by verifying a document or tenant.</p>
            </div>
          )}
        </div>
      </div>
    </LandlordLayout>
  )
}

export default LandlordDashboardNew
