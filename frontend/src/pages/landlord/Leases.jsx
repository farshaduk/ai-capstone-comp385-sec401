import { useState, useEffect } from 'react'
import LandlordLayout from '../../components/layouts/LandlordLayout'
import { landlordAPI } from '../../services/api'
import { FileText, DollarSign, Calendar, User, Loader, CheckCircle, Clock, XCircle } from 'lucide-react'
import toast from 'react-hot-toast'

const statusConfig = {
  active: { label: 'Active', color: 'badge-success', icon: CheckCircle },
  pending_signature: { label: 'Pending Signature', color: 'badge-warning', icon: Clock },
  pending: { label: 'Pending', color: 'badge-warning', icon: Clock },
  expired: { label: 'Expired', color: 'badge-danger', icon: XCircle },
  expiring: { label: 'Expiring Soon', color: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400 rounded-full px-3 py-1 text-xs font-medium', icon: Clock },
  terminated: { label: 'Terminated', color: 'bg-surface-100 text-surface-600 dark:bg-surface-800 dark:text-surface-400 rounded-full px-3 py-1 text-xs font-medium', icon: XCircle },
}

const Leases = () => {
  const [data, setData] = useState({ leases: [], stats: {} })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const { data: res } = await landlordAPI.getLeases()
        setData(res)
      } catch (err) {
        console.error(err)
        toast.error('Failed to load leases')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const stats = data.stats || {}

  return (
    <LandlordLayout title="Leases" subtitle="Manage your tenant lease agreements">
      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {[
          { label: 'Total Leases', value: stats.total || 0, icon: FileText },
          { label: 'Active Leases', value: stats.active || 0, icon: CheckCircle },
          { label: 'Monthly Revenue', value: `$${(stats.monthly_revenue || 0).toLocaleString()}`, icon: DollarSign },
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

      {loading ? (
        <div className="card p-16 text-center"><Loader className="h-8 w-8 animate-spin text-landlord-500 mx-auto mb-3" /><p className="text-surface-500">Loading...</p></div>
      ) : data.leases.length === 0 ? (
        <div className="card p-16 text-center">
          <FileText className="h-12 w-12 text-surface-300 dark:text-surface-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-surface-900 dark:text-white mb-2">No leases yet</h3>
          <p className="text-sm text-surface-500">Approve applicants to create lease agreements.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {data.leases.map(lease => {
            const cfg = statusConfig[lease.status] || statusConfig.active
            const Icon = cfg.icon
            return (
              <div key={lease.id} className="card p-5">
                <div className="flex flex-col md:flex-row md:items-center gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-semibold text-surface-900 dark:text-white">{lease.property}</h4>
                      <span className={cfg.color + ' flex items-center gap-1'}>
                        <Icon className="h-3 w-3" />{cfg.label}
                      </span>
                    </div>
                    <p className="text-sm text-surface-500">{lease.address}</p>
                    <div className="flex items-center gap-4 text-xs text-surface-400 mt-2">
                      <span className="flex items-center gap-1"><User className="h-3 w-3" /> {lease.tenant}</span>
                      <span className="flex items-center gap-1"><Calendar className="h-3 w-3" /> {lease.start_date} — {lease.end_date}</span>
                    </div>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <p className="text-lg font-bold text-surface-900 dark:text-white">${lease.rent?.toLocaleString()}<span className="text-xs font-normal text-surface-500">/mo</span></p>
                    {lease.deposit > 0 && <p className="text-xs text-surface-400">Deposit: ${lease.deposit?.toLocaleString()}</p>}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </LandlordLayout>
  )
}

export default Leases
