import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import Layout from '../../components/Layout'
import { landlordAPI } from '../../services/api'
import toast from 'react-hot-toast'
import { 
  LayoutDashboard, FileCheck, Users, Image, Shield, 
  TrendingUp, ArrowRight, Loader, AlertTriangle, CheckCircle 
} from 'lucide-react'

const LandlordDashboard = () => {
  const [stats, setStats] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const [statsRes, historyRes] = await Promise.allSettled([
        landlordAPI.getDashboardStats(),
        landlordAPI.getVerificationHistory(0, 5)
      ])
      if (statsRes.status === 'fulfilled') setStats(statsRes.value.data.stats)
      if (historyRes.status === 'fulfilled') setHistory(historyRes.value.data.history || [])
    } catch (error) {
      toast.error('Failed to load dashboard data')
    } finally {
      setLoading(false)
    }
  }

  const quickLinks = [
    { path: '/landlord/documents', icon: FileCheck, label: 'Verify Document', description: 'Upload and verify tenant documents', color: 'bg-blue-500' },
    { path: '/landlord/tenants', icon: Users, label: 'Verify Tenant', description: 'Full tenant application review', color: 'bg-purple-500' },
    { path: '/landlord/property-images', icon: Image, label: 'Verify Images', description: 'Check property listing photos', color: 'bg-green-500' },
    { path: '/landlord/history', icon: TrendingUp, label: 'View History', description: 'Past verification results', color: 'bg-orange-500' },
  ]

  const getActionLabel = (action) => {
    const labels = {
      document_verified: 'Document Verification',
      tenant_verified: 'Tenant Verification',
      property_image_verified: 'Image Verification',
      property_images_verified: 'Bulk Image Verification',
      cross_document_verified: 'Cross-Document Check',
      full_application_verified: 'Full Application Review'
    }
    return labels[action] || action
  }

  const getActionIcon = (action) => {
    if (action.includes('document')) return <FileCheck className="h-4 w-4" />
    if (action.includes('tenant') || action.includes('application')) return <Users className="h-4 w-4" />
    if (action.includes('image')) return <Image className="h-4 w-4" />
    return <Shield className="h-4 w-4" />
  }

  return (
    <Layout title="Landlord Dashboard">
      {loading ? (
        <div className="flex justify-center p-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      ) : (
        <div className="space-y-6 animate-fade-in">
          {/* Welcome Banner */}
          <div className="bg-gradient-to-r from-primary-600 to-blue-600 rounded-xl p-6 text-white">
            <h2 className="text-2xl font-bold mb-2">Landlord Verification Center</h2>
            <p className="opacity-80">Verify tenant documents, screen applications, and protect your property with AI-powered fraud detection.</p>
          </div>

          {/* Stats Cards */}
          {stats && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="card text-center">
                <FileCheck className="h-8 w-8 text-blue-600 mx-auto mb-2" />
                <p className="text-3xl font-bold">{stats.total_document_verifications}</p>
                <p className="text-sm text-gray-500">Documents Verified</p>
              </div>
              <div className="card text-center">
                <Users className="h-8 w-8 text-purple-600 mx-auto mb-2" />
                <p className="text-3xl font-bold">{stats.total_tenant_verifications}</p>
                <p className="text-sm text-gray-500">Tenants Screened</p>
              </div>
              <div className="card text-center">
                <Image className="h-8 w-8 text-green-600 mx-auto mb-2" />
                <p className="text-3xl font-bold">{stats.total_image_verifications}</p>
                <p className="text-sm text-gray-500">Images Analyzed</p>
              </div>
              <div className="card text-center">
                <Shield className="h-8 w-8 text-primary-600 mx-auto mb-2" />
                <p className="text-3xl font-bold">{stats.total_verifications}</p>
                <p className="text-sm text-gray-500">Total Verifications</p>
              </div>
            </div>
          )}

          {/* Quick Links */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {quickLinks.map((link) => {
              const Icon = link.icon
              return (
                <Link
                  key={link.path}
                  to={link.path}
                  className="card hover:shadow-lg transition-all group"
                >
                  <div className={`w-12 h-12 ${link.color} rounded-lg flex items-center justify-center mb-3`}>
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <h3 className="font-bold mb-1">{link.label}</h3>
                  <p className="text-sm text-gray-500">{link.description}</p>
                  <ArrowRight className="h-4 w-4 text-gray-400 group-hover:text-primary-600 mt-2 transition" />
                </Link>
              )
            })}
          </div>

          {/* Recent Activity */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold">Recent Verifications</h3>
              <Link to="/landlord/history" className="text-sm text-primary-600 hover:text-primary-700">
                View All <ArrowRight className="inline h-4 w-4" />
              </Link>
            </div>
            {history.length > 0 ? (
              <div className="space-y-3">
                {history.map((item) => (
                  <div key={item.id} className="flex items-center justify-between bg-gray-50 rounded-lg p-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center text-primary-600">
                        {getActionIcon(item.action)}
                      </div>
                      <div>
                        <p className="font-medium text-sm">{getActionLabel(item.action)}</p>
                        <p className="text-xs text-gray-500">{new Date(item.created_at).toLocaleString()}</p>
                      </div>
                    </div>
                    {item.details?.risk_level && (
                      <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                        item.details.risk_level === 'low' ? 'bg-green-100 text-green-700' :
                        item.details.risk_level === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                        {item.details.risk_level}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Shield className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                <p>No verifications yet. Start by verifying a document or tenant.</p>
              </div>
            )}
          </div>
        </div>
      )}
    </Layout>
  )
}

export default LandlordDashboard
