import { useState, useEffect } from 'react'
import LandlordLayout from '../../components/layouts/LandlordLayout'
import { landlordAPI } from '../../services/api'
import { DollarSign, Loader, ArrowUp, ArrowDown, CheckCircle, Clock } from 'lucide-react'
import toast from 'react-hot-toast'

const LandlordPayments = () => {
  const [leaseData, setLeaseData] = useState({ leases: [], stats: {} })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const { data } = await landlordAPI.getLeases()
        setLeaseData(data)
      } catch (err) {
        console.error(err)
        toast.error('Failed to load payment data')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const stats = leaseData.stats || {}
  const activeLeases = (leaseData.leases || []).filter(l => l.status === 'active')

  return (
    <LandlordLayout title="Payments" subtitle="Track rental income from your properties">
      {/* Revenue Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        <div className="card p-5">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-green-100 dark:bg-green-900/30 rounded-xl">
              <DollarSign className="h-6 w-6 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-xs text-surface-500">Monthly Revenue</p>
              <p className="text-2xl font-bold text-surface-900 dark:text-white">${(stats.monthly_revenue || 0).toLocaleString()}</p>
            </div>
          </div>
        </div>
        <div className="card p-5">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-landlord-100 dark:bg-landlord-900/30 rounded-xl">
              <CheckCircle className="h-6 w-6 text-landlord-600 dark:text-landlord-400" />
            </div>
            <div>
              <p className="text-xs text-surface-500">Active Leases</p>
              <p className="text-2xl font-bold text-surface-900 dark:text-white">{stats.active || 0}</p>
            </div>
          </div>
        </div>
        <div className="card p-5">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-blue-100 dark:bg-blue-900/30 rounded-xl">
              <ArrowUp className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-xs text-surface-500">Annual Projected</p>
              <p className="text-2xl font-bold text-surface-900 dark:text-white">${((stats.monthly_revenue || 0) * 12).toLocaleString()}</p>
            </div>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="card p-16 text-center"><Loader className="h-8 w-8 animate-spin text-landlord-500 mx-auto mb-3" /><p className="text-surface-500">Loading...</p></div>
      ) : activeLeases.length === 0 ? (
        <div className="card p-16 text-center">
          <DollarSign className="h-12 w-12 text-surface-300 dark:text-surface-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-surface-900 dark:text-white mb-2">No active income</h3>
          <p className="text-sm text-surface-500">You have no active leases generating rental income.</p>
        </div>
      ) : (
        <>
          <h3 className="text-sm font-semibold text-surface-500 uppercase tracking-wider mb-3">Income by Property</h3>
          <div className="space-y-3">
            {activeLeases.map(lease => (
              <div key={lease.id} className="card p-4 flex items-center gap-4">
                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-surface-900 dark:text-white text-sm">{lease.property}</h4>
                  <p className="text-xs text-surface-500">{lease.tenant} • {lease.start_date} — {lease.end_date}</p>
                </div>
                <div className="text-right flex-shrink-0">
                  <p className="text-lg font-bold text-green-600 dark:text-green-400">${lease.rent?.toLocaleString()}</p>
                  <p className="text-xs text-surface-400">/month</p>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </LandlordLayout>
  )
}

export default LandlordPayments
