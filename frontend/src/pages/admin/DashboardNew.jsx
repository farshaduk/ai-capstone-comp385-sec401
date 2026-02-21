import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import AdminLayout from '../../components/layouts/AdminLayout'
import { adminAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  Users, Database, Brain, AlertTriangle, Activity,
  MessageSquare, Sparkles, RefreshCw, TrendingUp, Target, Zap,
  ArrowRight, BarChart3, Loader, Shield
} from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const AdminDashboardNew = () => {
  const [stats, setStats] = useState(null)
  const [feedbackStats, setFeedbackStats] = useState(null)
  const [autoLearnInsights, setAutoLearnInsights] = useState(null)
  const [loading, setLoading] = useState(true)
  const [autoLearnTriggering, setAutoLearnTriggering] = useState(false)

  useEffect(() => {
    Promise.allSettled([fetchStats(), fetchAutoLearnInsights()])
  }, [])

  const fetchStats = async () => {
    try {
      const [s, f] = await Promise.all([adminAPI.getDashboard(), adminAPI.getFeedbackStats()])
      setStats(s.data)
      setFeedbackStats(f.data)
    } catch { console.error('Failed to fetch stats') }
    finally { setLoading(false) }
  }

  const fetchAutoLearnInsights = async () => {
    try {
      const r = await adminAPI.getAutoLearnInsights()
      setAutoLearnInsights(r.data)
    } catch { /* optional */ }
  }

  const triggerAutoLearn = async () => {
    setAutoLearnTriggering(true)
    try {
      const r = await adminAPI.triggerAutoLearn()
      const runResult = r.data
      toast.success(
        runResult.status === 'success'
          ? `Auto-learning completed! ${runResult.patterns_found} patterns found.`
          : runResult.message || 'Auto-learning completed!'
      )
      // Re-fetch stats so autoLearnInsights has the consistent stats shape
      await fetchAutoLearnInsights()
    } catch (e) {
      toast.error('Auto-learning failed: ' + (e.response?.data?.detail || e.message))
    } finally { setAutoLearnTriggering(false) }
  }

  if (loading) {
    return (
      <AdminLayout title="Dashboard">
        <div className="flex justify-center py-20"><Loader className="h-8 w-8 animate-spin text-primary-600" /></div>
      </AdminLayout>
    )
  }

  const chartData = [
    { name: 'Users', value: stats?.total_users || 0 },
    { name: 'Analyses', value: stats?.total_analyses || 0 },
    { name: 'High Risk', value: stats?.high_risk_analyses || 0 },
  ]

  return (
    <AdminLayout title="Dashboard" subtitle="System overview and analytics">
      <div className="space-y-6 animate-fade-in">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {[
            { icon: Users, label: 'Total Users', value: stats?.total_users || 0, color: 'text-blue-600', bg: 'bg-blue-50 dark:bg-blue-900/20' },
            { icon: Activity, label: 'Risk Analyses', value: stats?.total_analyses || 0, color: 'text-emerald-600', bg: 'bg-emerald-50 dark:bg-emerald-900/20' },
            { icon: AlertTriangle, label: 'High Risk', value: stats?.high_risk_analyses || 0, color: 'text-red-600', bg: 'bg-red-50 dark:bg-red-900/20' },
          ].map((s, i) => (
            <div key={i} className={`card p-5 ${s.bg}`}>
              <s.icon className={`h-7 w-7 ${s.color} mb-2`} />
              <p className={`text-2xl font-display font-bold ${s.color}`}>{s.value}</p>
              <p className="text-xs text-surface-500 dark:text-surface-400 font-medium">{s.label}</p>
            </div>
          ))}
        </div>

        {/* Chart + Feedback Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Chart */}
          <div className="card p-6">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-surface-400" /> System Overview
            </h3>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip contentStyle={{ borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }} />
                <Bar dataKey="value" fill="#6366f1" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Feedback */}
          {feedbackStats && (
            <div className="card p-6">
              <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
                <MessageSquare className="h-5 w-5 text-purple-500" /> User Feedback
                {(feedbackStats.pending_review ?? 0) > 0 && (
                  <Link to="/admin/feedback-review" className="ml-auto inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300 hover:bg-amber-200 dark:hover:bg-amber-900/50 transition-colors">
                    <AlertTriangle className="h-3 w-3" /> {feedbackStats.pending_review} pending review
                  </Link>
                )}
              </h3>
              <div className="grid grid-cols-2 gap-3 mb-4">
                {[
                  { label: 'Total', value: feedbackStats.total_feedback, bg: 'bg-surface-50 dark:bg-surface-700' },
                  { label: 'Safe', value: feedbackStats.safe_reports, bg: 'bg-emerald-50 dark:bg-emerald-900/20', color: 'text-emerald-600' },
                  { label: 'Fraud', value: feedbackStats.fraud_reports, bg: 'bg-red-50 dark:bg-red-900/20', color: 'text-red-600' },
                  { label: 'Unsure', value: feedbackStats.unsure_reports, bg: 'bg-amber-50 dark:bg-amber-900/20', color: 'text-amber-600' },
                ].map((f, i) => (
                  <div key={i} className={`${f.bg} rounded-xl p-3`}>
                    <p className={`text-xl font-display font-bold ${f.color || 'text-surface-900 dark:text-white'}`}>{f.value}</p>
                    <p className="text-xs text-surface-500">{f.label}</p>
                  </div>
                ))}
              </div>
              {feedbackStats.total_feedback > 0 && (
                <div className="p-3 rounded-xl bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800">
                  <p className="text-xs text-primary-700 dark:text-primary-300">
                    <strong>Fraud Rate:</strong> {feedbackStats.fraud_confirmation_rate.toFixed(1)}% of listings flagged as fraud
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Auto-Learning */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-amber-500" /> AI Auto-Learning Engine
            </h3>
            <div className="flex gap-2">
              <button onClick={fetchAutoLearnInsights} className="btn btn-sm btn-secondary">
                <RefreshCw className="h-3.5 w-3.5" /> Refresh
              </button>
              <button onClick={triggerAutoLearn}
                disabled={autoLearnTriggering || (feedbackStats?.total_feedback || 0) < 5}
                className="btn btn-sm btn-primary">
                <Zap className={`h-3.5 w-3.5 ${autoLearnTriggering ? 'animate-pulse' : ''}`} />
                {autoLearnTriggering ? 'Learning...' : 'Trigger Auto-Learn'}
              </button>
            </div>
          </div>

          {(feedbackStats?.total_feedback || 0) < 5 && (
            <div className="p-3 rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 mb-4">
              <p className="text-xs text-amber-700 dark:text-amber-300 flex items-center gap-2">
                <AlertTriangle className="h-3.5 w-3.5" />
                Need 5+ feedback entries for auto-learning. Currently: {feedbackStats?.total_feedback || 0}
              </p>
            </div>
          )}

          {autoLearnInsights && (
            <div className="grid grid-cols-3 gap-4">
              {[
                { icon: Target, label: 'Patterns', value: autoLearnInsights.total_patterns ?? 0, color: 'text-amber-600', bg: 'bg-amber-50 dark:bg-amber-900/20' },
                { icon: TrendingUp, label: 'Calibrated', value: autoLearnInsights.calibrated_weights ?? 0, color: 'text-blue-600', bg: 'bg-blue-50 dark:bg-blue-900/20' },
                { icon: Shield, label: 'Keywords', value: autoLearnInsights.fraud_keywords ?? 0, color: 'text-purple-600', bg: 'bg-purple-50 dark:bg-purple-900/20' },
              ].map((s, i) => (
                <div key={i} className={`${s.bg} rounded-xl p-4`}>
                  <s.icon className={`h-5 w-5 ${s.color} mb-1`} />
                  <p className={`text-xl font-display font-bold ${s.color}`}>{s.value}</p>
                  <p className="text-xs text-surface-500">{s.label}</p>
                </div>
              ))}
              {autoLearnInsights.last_updated && (
                <div className="col-span-3">
                  <p className="text-xs text-surface-400 text-right">
                    Last updated: {new Date(autoLearnInsights.last_updated).toLocaleString()}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { path: '/admin/datasets', icon: Database, label: 'Datasets', desc: 'Upload & manage training data', color: 'bg-purple-500' },
            { path: '/admin/trained-models', icon: Brain, label: 'Models', desc: 'Train & deploy ML models', color: 'bg-primary-500' },
            { path: '/admin/users', icon: Users, label: 'Users', desc: 'Manage platform users', color: 'bg-blue-500' },
            { path: '/admin/audit-logs', icon: Activity, label: 'Audit Logs', desc: 'Security & activity logs', color: 'bg-emerald-500' },
          ].map((link) => (
            <Link key={link.path} to={link.path} className="card p-5 group hover:shadow-elevated transition-all duration-200">
              <div className={`w-10 h-10 ${link.color} rounded-xl flex items-center justify-center mb-3`}>
                <link.icon className="h-5 w-5 text-white" />
              </div>
              <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-1">{link.label}</h4>
              <p className="text-xs text-surface-500 dark:text-surface-400">{link.desc}</p>
              <ArrowRight className="h-4 w-4 text-surface-300 group-hover:text-primary-600 mt-2 transition-colors" />
            </Link>
          ))}
        </div>
      </div>
    </AdminLayout>
  )
}

export default AdminDashboardNew
