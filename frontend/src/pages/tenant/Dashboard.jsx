import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import TenantLayout from '../../components/layouts/TenantLayout'
import { useAuthStore } from '../../store/authStore'
import { renterAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  Shield, Search, AlertTriangle, CheckCircle, TrendingUp,
  Clock, ArrowRight, FileText, BarChart3, Zap
} from 'lucide-react'

const TenantDashboard = () => {
  const { user } = useAuthStore()
  const [stats, setStats] = useState(null)
  const [recentAnalyses, setRecentAnalyses] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDashboard()
  }, [])

  const loadDashboard = async () => {
    try {
      const [statsRes, historyRes] = await Promise.all([
        renterAPI.getStats().catch(() => ({ data: null })),
        renterAPI.getHistory(0, 5).catch(() => ({ data: [] })),
      ])
      setStats(statsRes.data)
      setRecentAnalyses(Array.isArray(historyRes.data) ? historyRes.data : [])
    } catch (err) {
      // silently handle
    } finally {
      setLoading(false)
    }
  }

  const getRiskColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'very_low': case 'low': return 'text-accent-green bg-emerald-50 dark:bg-emerald-950/30'
      case 'medium': return 'text-accent-amber bg-amber-50 dark:bg-amber-950/30'
      case 'high': case 'very_high': return 'text-accent-red bg-red-50 dark:bg-red-950/30'
      default: return 'text-surface-500 bg-surface-50 dark:bg-surface-800'
    }
  }

  return (
    <TenantLayout title="Dashboard" subtitle={`Welcome back, ${user?.full_name?.split(' ')[0] || 'there'}`}>
      {/* Quick Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {[
          {
            label: 'Total Scans',
            value: stats?.total_analyses || 0,
            icon: Search,
            color: 'text-primary-600 dark:text-primary-400',
            bg: 'bg-primary-50 dark:bg-primary-950/30',
          },
          {
            label: 'Safe Listings',
            value: stats?.safe_count || 0,
            icon: CheckCircle,
            color: 'text-accent-green',
            bg: 'bg-emerald-50 dark:bg-emerald-950/30',
          },
          {
            label: 'Fraud Detected',
            value: stats?.fraud_count || 0,
            icon: AlertTriangle,
            color: 'text-accent-red',
            bg: 'bg-red-50 dark:bg-red-950/30',
          },
          {
            label: 'Scans Left',
            value: stats?.scans_remaining ?? user?.scans_remaining ?? 0,
            icon: Zap,
            color: 'text-accent-amber',
            bg: 'bg-amber-50 dark:bg-amber-950/30',
          },
        ].map((stat, i) => (
          <div key={i} className="card p-5 animate-fade-in-up" style={{ animationDelay: `${i * 0.1}s` }}>
            <div className="flex items-center justify-between mb-3">
              <div className={`p-2 rounded-xl ${stat.bg}`}>
                <stat.icon className={`h-5 w-5 ${stat.color}`} />
              </div>
            </div>
            <p className="stat-value">{stat.value}</p>
            <p className="stat-label mt-1">{stat.label}</p>
          </div>
        ))}
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Quick Analyze */}
        <div className="lg:col-span-2">
          <div className="card p-6 mb-6">
            <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary-500" />
              Quick Analyze
            </h3>
            <p className="text-sm text-surface-500 dark:text-surface-400 mb-4">
              Paste a rental listing URL or text to instantly analyze for fraud indicators.
            </p>
            <Link
              to="/tenant/analyze"
              className="btn btn-lg btn-primary w-full"
            >
              <Search className="h-5 w-5" />
              Start Analysis
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>

          {/* Recent Analyses */}
          <div className="card">
            <div className="flex items-center justify-between p-6 pb-4">
              <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white">
                Recent Analyses
              </h3>
              <Link to="/tenant/history" className="text-sm font-medium text-primary-600 dark:text-primary-400 hover:underline">
                View All
              </Link>
            </div>
            <div className="divide-y divide-surface-100 dark:divide-surface-800">
              {recentAnalyses.length > 0 ? (
                recentAnalyses.map((analysis, i) => (
                  <div key={analysis.id || i} className="px-6 py-4 hover:bg-surface-50 dark:hover:bg-surface-800/30 transition-colors">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-surface-900 dark:text-white truncate">
                          {analysis.listing_text?.substring(0, 80) || 'Analysis'}...
                        </p>
                        <div className="flex items-center gap-3 mt-1.5">
                          <span className={`badge text-xs ${getRiskColor(analysis.risk_level)}`}>
                            {analysis.risk_level?.replace('_', ' ') || 'N/A'}
                          </span>
                          <span className="text-xs text-surface-400 dark:text-surface-500 flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {new Date(analysis.created_at).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-lg font-bold text-surface-900 dark:text-white">
                          {analysis.risk_score || 0}
                        </p>
                        <p className="text-xs text-surface-400">score</p>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="px-6 py-12 text-center">
                  <Search className="h-10 w-10 text-surface-300 dark:text-surface-600 mx-auto mb-3" />
                  <p className="text-sm text-surface-500 dark:text-surface-400">No analyses yet</p>
                  <p className="text-xs text-surface-400 dark:text-surface-500 mt-1">Start analyzing listings to see results here</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Trust Score Card - computed from real analysis data */}
          <div className="card p-6 bg-gradient-to-br from-tenant-50 to-emerald-50 dark:from-tenant-950/20 dark:to-emerald-950/20 border-tenant-200 dark:border-tenant-800">
            <h3 className="text-sm font-semibold text-tenant-700 dark:text-tenant-300 mb-3">Your Trust Score</h3>
            {(() => {
              const total = stats?.total_analyses || 0
              const safe = stats?.safe_count || 0
              const fraud = stats?.fraud_count || 0
              // Trust score rewards engagement: scanning listings = due diligence.
              // Finding fraud is GOOD (tenant protected themselves), not a penalty.
              // Formula: base 40 + engagement bonus (up to 30 from scan count)
              //        + diligence bonus (up to 30 from % of scans that were verified safe or caught fraud)
              const engagementBonus = Math.min(30, total * 3)  // +3 per scan, cap at 30
              const verifiedRatio = total > 0 ? (safe + fraud) / total : 0  // both safe + fraud = verified
              const diligenceBonus = verifiedRatio * 30
              const trustScore = total === 0 ? 0 : Math.min(100, Math.max(0, Math.round(40 + engagementBonus + diligenceBonus)))
              return (
                <>
                  <div className="flex items-end gap-2 mb-3">
                    <span className="text-4xl font-display font-bold text-tenant-600 dark:text-tenant-400">{trustScore}</span>
                    <span className="text-sm text-tenant-500 dark:text-tenant-400 mb-1">/100</span>
                  </div>
                  <div className="w-full bg-tenant-200 dark:bg-tenant-800 rounded-full h-2 mb-3">
                    <div className="bg-tenant-500 h-2 rounded-full transition-all" style={{ width: `${trustScore}%` }} />
                  </div>
                  <p className="text-xs text-tenant-600 dark:text-tenant-400 flex items-center gap-1">
                    <TrendingUp className="h-3 w-3" />
                    Based on {total} scan{total !== 1 ? 's' : ''}
                  </p>
                </>
              )
            })()}
          </div>

          {/* Subscription */}
          <div className="card p-6">
            <h3 className="text-sm font-display font-bold text-surface-900 dark:text-white mb-3">Subscription</h3>
            <div className="flex items-center gap-3 mb-4">
              <div className="badge-primary capitalize">
                {user?.subscription_plan || 'Free'}
              </div>
            </div>
            <p className="text-sm text-surface-500 dark:text-surface-400 mb-4">
              {stats?.scans_remaining ?? user?.scans_remaining ?? 0} scans remaining this month
            </p>
            <Link to="/tenant/subscription" className="btn btn-sm btn-outline w-full">
              Upgrade Plan
            </Link>
          </div>

          {/* Quick Actions */}
          <div className="card p-6">
            <h3 className="text-sm font-display font-bold text-surface-900 dark:text-white mb-4">Quick Actions</h3>
            <div className="space-y-2">
              {[
                { to: '/tenant/analyze', icon: Search, label: 'Analyze Listing' },
                { to: '/tenant/history', icon: FileText, label: 'View History' },
                { to: '/tenant/profile', icon: BarChart3, label: 'Edit Profile' },
              ].map(action => (
                <Link
                  key={action.to}
                  to={action.to}
                  className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium
                           text-surface-600 hover:text-surface-900 hover:bg-surface-50
                           dark:text-surface-400 dark:hover:text-white dark:hover:bg-surface-800
                           transition-colors"
                >
                  <action.icon className="h-4 w-4" />
                  {action.label}
                  <ArrowRight className="h-3 w-3 ml-auto" />
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>
    </TenantLayout>
  )
}

export default TenantDashboard
