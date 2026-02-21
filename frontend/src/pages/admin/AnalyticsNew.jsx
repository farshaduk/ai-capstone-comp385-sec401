import { useState, useEffect } from 'react'
import AdminLayout from '../../components/layouts/AdminLayout'
import { adminAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  BarChart3, TrendingUp, AlertTriangle, Users, Activity,
  Target, Loader, RefreshCw, Calendar, Shield, Brain,
  PieChart, ArrowUpRight, ArrowDownRight, Minus, MessageSquare
} from 'lucide-react'
import {
  BarChart, Bar, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart as RPieChart, Pie, Cell, Legend
} from 'recharts'

const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899']

const AdminAnalyticsNew = () => {
  const [overview, setOverview] = useState(null)
  const [modelAccuracy, setModelAccuracy] = useState(null)
  const [topIndicators, setTopIndicators] = useState(null)
  const [loading, setLoading] = useState(true)
  const [days, setDays] = useState(30)

  useEffect(() => { fetchAll() }, [days])

  const fetchAll = async () => {
    setLoading(true)
    try {
      const [o, m, t] = await Promise.allSettled([
        adminAPI.getAnalyticsOverview(days),
        adminAPI.getModelAccuracy(),
        adminAPI.getTopIndicators()
      ])
      if (o.status === 'fulfilled') setOverview(o.value.data)
      if (m.status === 'fulfilled') setModelAccuracy(m.value.data)
      if (t.status === 'fulfilled') setTopIndicators(t.value.data)
    } catch { toast.error('Failed to load analytics') }
    finally { setLoading(false) }
  }

  if (loading) {
    return (
      <AdminLayout title="Analytics">
        <div className="flex justify-center py-20"><Loader className="h-8 w-8 animate-spin text-primary-600" /></div>
      </AdminLayout>
    )
  }

  const summary = overview?.summary || {}

  return (
    <AdminLayout title="Analytics" subtitle="Deep insights and trend analysis">
      <div className="space-y-6 animate-fade-in">
        {/* Period Selector + Refresh */}
        <div className="flex items-center justify-between">
          <div className="flex gap-2">
            {[7, 14, 30, 90].map((d) => (
              <button key={d} onClick={() => setDays(d)}
                className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
                  days === d
                    ? 'bg-primary-600 text-white shadow-sm'
                    : 'bg-surface-100 dark:bg-surface-700 text-surface-600 dark:text-surface-300 hover:bg-surface-200 dark:hover:bg-surface-600'
                }`}>
                {d}d
              </button>
            ))}
          </div>
          <button onClick={fetchAll} className="btn btn-sm btn-secondary">
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </button>
        </div>

        {/* Summary Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            { icon: Activity, label: 'Analyses', value: summary.total_analyses || 0, color: 'text-blue-600', bg: 'bg-blue-50 dark:bg-blue-900/20' },
            { icon: AlertTriangle, label: 'Flagged', value: summary.total_flagged || 0, color: 'text-red-600', bg: 'bg-red-50 dark:bg-red-900/20' },
            { icon: Target, label: 'Fraud Rate', value: `${summary.fraud_rate || 0}%`, color: 'text-amber-600', bg: 'bg-amber-50 dark:bg-amber-900/20' },
            { icon: Users, label: 'New Users', value: summary.new_users || 0, color: 'text-emerald-600', bg: 'bg-emerald-50 dark:bg-emerald-900/20' },
            { icon: MessageSquare, label: 'Feedback', value: summary.total_feedback || 0, color: 'text-purple-600', bg: 'bg-purple-50 dark:bg-purple-900/20' },
            { icon: TrendingUp, label: 'Avg Risk', value: summary.avg_risk_score || 0, color: 'text-primary-600', bg: 'bg-primary-50 dark:bg-primary-900/20' },
          ].map((s, i) => (
            <div key={i} className={`card p-5 ${s.bg}`}>
              <s.icon className={`h-7 w-7 ${s.color} mb-2`} />
              <p className={`text-2xl font-display font-bold ${s.color}`}>{s.value}</p>
              <p className="text-xs text-surface-500 dark:text-surface-400 font-medium">{s.label}</p>
            </div>
          ))}
        </div>

        {/* Analysis Volume Trend + Fraud Rate Trend */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Analysis Volume */}
          <div className="card p-6">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-surface-400" /> Analysis Volume
            </h3>
            {(overview?.analysis_trend?.length || 0) > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={overview.analysis_trend}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(d) => d.slice(5)} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }} />
                  <Area type="monotone" dataKey="count" fill="#6366f1" fillOpacity={0.15} stroke="#6366f1" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[240px] text-surface-400 text-sm">
                No analysis data for this period
              </div>
            )}
          </div>

          {/* Fraud Rate Trend */}
          <div className="card p-6">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-400" /> Fraud Detection Rate
            </h3>
            {(overview?.fraud_rate_trend?.length || 0) > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={overview.fraud_rate_trend}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(d) => d.slice(5)} />
                  <YAxis tick={{ fontSize: 12 }} unit="%" />
                  <Tooltip contentStyle={{ borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}
                    formatter={(val) => [`${val}%`, 'Fraud Rate']} />
                  <Line type="monotone" dataKey="rate" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[240px] text-surface-400 text-sm">
                No fraud rate data for this period
              </div>
            )}
          </div>
        </div>

        {/* Risk Score Distribution + Risk Level Breakdown */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Score Distribution */}
          <div className="card p-6">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <PieChart className="h-5 w-5 text-purple-400" /> Risk Score Distribution
            </h3>
            {(overview?.score_distribution?.length || 0) > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={overview.score_distribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }} />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {overview.score_distribution.map((entry, idx) => (
                      <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[240px] text-surface-400 text-sm">
                No score data for this period
              </div>
            )}
          </div>

          {/* Risk Level Breakdown */}
          <div className="card p-6">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <Shield className="h-5 w-5 text-amber-400" /> Risk Level Breakdown
            </h3>
            {(overview?.risk_level_breakdown?.length || 0) > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <RPieChart>
                  <Pie data={overview.risk_level_breakdown} dataKey="count" nameKey="level"
                    cx="50%" cy="50%" outerRadius={85} innerRadius={40} paddingAngle={3}
                    label={({ level, count }) => `${level}: ${count}`}>
                    {overview.risk_level_breakdown.map((entry, idx) => (
                      <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }} />
                  <Legend />
                </RPieChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[240px] text-surface-400 text-sm">
                No risk level data for this period
              </div>
            )}
          </div>
        </div>

        {/* User Registration Trend + Role Distribution */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* User Trend */}
          <div className="card p-6">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <Users className="h-5 w-5 text-blue-400" /> User Registrations
            </h3>
            {(overview?.user_trend?.length || 0) > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={overview.user_trend}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(d) => d.slice(5)} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }} />
                  <Area type="monotone" dataKey="count" fill="#10b981" fillOpacity={0.15} stroke="#10b981" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[240px] text-surface-400 text-sm">
                No user registration data for this period
              </div>
            )}
          </div>

          {/* Role Distribution */}
          <div className="card p-6">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <Shield className="h-5 w-5 text-emerald-400" /> User Roles
            </h3>
            {(overview?.role_distribution?.length || 0) > 0 ? (
              <div className="space-y-3 pt-2">
                {overview.role_distribution.map((r, i) => {
                  const total = overview.role_distribution.reduce((s, x) => s + x.count, 0)
                  const pct = total > 0 ? Math.round((r.count / total) * 100) : 0
                  return (
                    <div key={i}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium text-surface-700 dark:text-surface-300 capitalize">{r.role}</span>
                        <span className="text-sm font-bold text-surface-900 dark:text-white">{r.count} <span className="text-xs text-surface-400 font-normal">({pct}%)</span></span>
                      </div>
                      <div className="h-2.5 bg-surface-100 dark:bg-surface-700 rounded-full overflow-hidden">
                        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: COLORS[i % COLORS.length] }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="flex items-center justify-center h-[240px] text-surface-400 text-sm">
                No user role data
              </div>
            )}
          </div>
        </div>

        {/* Feedback Trend */}
        {(overview?.feedback_trend?.length || 0) > 0 && (
          <div className="card p-6">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <MessageSquare className="h-5 w-5 text-purple-400" /> Feedback Trend
            </h3>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={overview.feedback_trend}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(d) => d.slice(5)} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip contentStyle={{ borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }} />
                <Bar dataKey="safe" fill="#10b981" radius={[4, 4, 0, 0]} stackId="fb" name="Safe" />
                <Bar dataKey="fraud" fill="#ef4444" radius={[4, 4, 0, 0]} stackId="fb" name="Fraud" />
                <Bar dataKey="unsure" fill="#f59e0b" radius={[4, 4, 0, 0]} stackId="fb" name="Unsure" />
                <Legend />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Model Accuracy + Top Indicators */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Model Accuracy */}
          <div className="card p-6">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary-400" /> Model Accuracy
            </h3>
            {modelAccuracy?.metrics ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: 'Accuracy', value: modelAccuracy.interpretation.accuracy_pct, color: 'text-blue-600', bg: 'bg-blue-50 dark:bg-blue-900/20' },
                    { label: 'Precision', value: modelAccuracy.interpretation.precision_pct, color: 'text-emerald-600', bg: 'bg-emerald-50 dark:bg-emerald-900/20' },
                    { label: 'Recall', value: modelAccuracy.interpretation.recall_pct, color: 'text-amber-600', bg: 'bg-amber-50 dark:bg-amber-900/20' },
                    { label: 'F1 Score', value: modelAccuracy.interpretation.f1_pct, color: 'text-purple-600', bg: 'bg-purple-50 dark:bg-purple-900/20' },
                  ].map((m, i) => (
                    <div key={i} className={`${m.bg} rounded-xl p-3`}>
                      <p className={`text-xl font-display font-bold ${m.color}`}>{m.value}</p>
                      <p className="text-xs text-surface-500">{m.label}</p>
                    </div>
                  ))}
                </div>
                {/* Confusion Matrix */}
                <div className="p-3 rounded-xl bg-surface-50 dark:bg-surface-700/50">
                  <p className="text-xs font-semibold text-surface-600 dark:text-surface-300 mb-2">Confusion Matrix</p>
                  <div className="grid grid-cols-2 gap-2 text-center text-xs">
                    <div className="bg-emerald-50 dark:bg-emerald-900/30 rounded-lg p-2">
                      <p className="font-bold text-emerald-600">{modelAccuracy.confusion_matrix.true_positive}</p>
                      <p className="text-surface-500">True Pos</p>
                    </div>
                    <div className="bg-red-50 dark:bg-red-900/30 rounded-lg p-2">
                      <p className="font-bold text-red-600">{modelAccuracy.confusion_matrix.false_positive}</p>
                      <p className="text-surface-500">False Pos</p>
                    </div>
                    <div className="bg-amber-50 dark:bg-amber-900/30 rounded-lg p-2">
                      <p className="font-bold text-amber-600">{modelAccuracy.confusion_matrix.false_negative}</p>
                      <p className="text-surface-500">False Neg</p>
                    </div>
                    <div className="bg-blue-50 dark:bg-blue-900/30 rounded-lg p-2">
                      <p className="font-bold text-blue-600">{modelAccuracy.confusion_matrix.true_negative}</p>
                      <p className="text-surface-500">True Neg</p>
                    </div>
                  </div>
                </div>
                <p className="text-xs text-surface-400">Based on {modelAccuracy.sample_count} confirmed feedback samples</p>
              </div>
            ) : (
              <div className="p-4 rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
                <p className="text-xs text-amber-700 dark:text-amber-300 flex items-center gap-2">
                  <AlertTriangle className="h-3.5 w-3.5" />
                  {modelAccuracy?.message || 'No confirmed feedback data to calculate accuracy'}
                </p>
              </div>
            )}
          </div>

          {/* Top Fraud Indicators */}
          <div className="card p-6">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <Target className="h-5 w-5 text-red-400" /> Top Fraud Indicators
            </h3>
            {(topIndicators?.top_indicators?.length || 0) > 0 ? (
              <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
                {topIndicators.top_indicators.map((ind, i) => {
                  const maxCount = topIndicators.top_indicators[0]?.count || 1
                  const pct = Math.round((ind.count / maxCount) * 100)
                  return (
                    <div key={i}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-medium text-surface-700 dark:text-surface-300 truncate max-w-[180px]" title={ind.indicator}>
                          {ind.indicator}
                        </span>
                        <span className="text-xs font-bold text-surface-900 dark:text-white ml-2">{ind.count}</span>
                      </div>
                      <div className="h-1.5 bg-surface-100 dark:bg-surface-700 rounded-full overflow-hidden">
                        <div className="h-full bg-red-500 rounded-full transition-all duration-500" style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  )
                })}
                <p className="text-xs text-surface-400 pt-2">
                  Across {topIndicators.total_flagged_analyses} flagged analyses
                </p>
              </div>
            ) : (
              <div className="flex items-center justify-center h-[240px] text-surface-400 text-sm">
                No flagged analyses to extract indicators
              </div>
            )}
          </div>
        </div>
      </div>
    </AdminLayout>
  )
}

export default AdminAnalyticsNew
