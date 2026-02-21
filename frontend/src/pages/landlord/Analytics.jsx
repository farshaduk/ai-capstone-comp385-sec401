import { useState, useEffect } from 'react'
import LandlordLayout from '../../components/layouts/LandlordLayout'
import { landlordAPI } from '../../services/api'
import { BarChart3, Eye, Users, DollarSign, Home, Loader, TrendingUp } from 'lucide-react'
import toast from 'react-hot-toast'

const Analytics = () => {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const { data: res } = await landlordAPI.getAnalytics()
        setData(res)
      } catch (err) {
        console.error(err)
        toast.error('Failed to load analytics')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return (
    <LandlordLayout title="Analytics" subtitle="Track your property performance">
      <div className="card p-16 text-center"><Loader className="h-8 w-8 animate-spin text-landlord-500 mx-auto mb-3" /><p className="text-surface-500">Loading analytics...</p></div>
    </LandlordLayout>
  )

  const stats = data?.stats || {}
  const perf = data?.listing_performance || []

  return (
    <LandlordLayout title="Analytics" subtitle="Track your property performance">
      {/* KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
        {[
          { label: 'Total Views', value: stats.total_views || 0, icon: Eye, color: 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400' },
          { label: 'Applications', value: stats.total_applications || 0, icon: Users, color: 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400' },
          { label: 'Monthly Revenue', value: `$${(stats.monthly_revenue || 0).toLocaleString()}`, icon: DollarSign, color: 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400' },
          { label: 'Occupancy Rate', value: `${stats.occupancy_rate || 0}%`, icon: Home, color: 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400' },
          { label: 'Active Listings', value: stats.active_listings || 0, icon: TrendingUp, color: 'bg-landlord-100 dark:bg-landlord-900/30 text-landlord-600 dark:text-landlord-400' },
        ].map((s, i) => (
          <div key={i} className="card p-4">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-xl ${s.color.split(' ').slice(0, 2).join(' ')}`}>
                <s.icon className={`h-5 w-5 ${s.color.split(' ').slice(2).join(' ')}`} />
              </div>
              <div>
                <p className="text-xs text-surface-500">{s.label}</p>
                <p className="text-lg font-bold text-surface-900 dark:text-white">{s.value}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Listing Performance Table */}
      <div className="card overflow-hidden">
        <div className="p-4 border-b border-surface-200 dark:border-surface-700">
          <h3 className="font-semibold text-surface-900 dark:text-white flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-landlord-500" /> Listing Performance
          </h3>
        </div>
        {perf.length === 0 ? (
          <div className="p-8 text-center text-surface-400 text-sm">No listing data available yet.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-200 dark:border-surface-700">
                  <th className="text-left px-4 py-3 text-xs font-medium text-surface-500 uppercase">Property</th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-surface-500 uppercase">Views</th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-surface-500 uppercase">Applications</th>
                  <th className="text-right px-4 py-3 text-xs font-medium text-surface-500 uppercase">Apply Rate</th>
                </tr>
              </thead>
              <tbody>
                {perf.map((p, i) => {
                  const conv = p.views > 0 ? ((p.applications / p.views) * 100).toFixed(1) : '0.0'
                  return (
                    <tr key={i} className="border-b border-surface-100 dark:border-surface-800 last:border-0">
                      <td className="px-4 py-3 font-medium text-surface-900 dark:text-white">{p.name}</td>
                      <td className="px-4 py-3 text-right text-surface-600 dark:text-surface-400">{p.views}</td>
                      <td className="px-4 py-3 text-right text-surface-600 dark:text-surface-400">{p.applications}</td>
                      <td className="px-4 py-3 text-right">
                        <span className={`text-xs font-medium ${parseFloat(conv) > 5 ? 'text-green-600' : 'text-surface-400'}`}>{conv}%</span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </LandlordLayout>
  )
}

export default Analytics
