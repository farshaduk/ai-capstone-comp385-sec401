import { useState, useEffect, useCallback } from 'react'
import AdminLayout from '../../components/layouts/AdminLayout'
import { adminAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  Activity, Server, Database, HardDrive, Cpu, Brain,
  AlertTriangle, CheckCircle, XCircle, Clock, RefreshCw,
  Loader, Shield, Zap, Package, MemoryStick, Gauge,
  AlertOctagon, Info, User, Calendar
} from 'lucide-react'

const AdminMonitoringNew = () => {
  const [health, setHealth] = useState(null)
  const [enginesHealth, setEnginesHealth] = useState(null)
  const [errors, setErrors] = useState(null)
  const [activity, setActivity] = useState(null)
  const [deps, setDeps] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('health')

  useEffect(() => { fetchAll() }, [])

  const fetchAll = async () => {
    setLoading(true)
    try {
      const [h, e, er, a, d] = await Promise.allSettled([
        adminAPI.getSystemHealth(),
        adminAPI.getAIEnginesHealth(),
        adminAPI.getRecentErrors(),
        adminAPI.getActivityFeed(),
        adminAPI.getDependencyVersions()
      ])
      if (h.status === 'fulfilled') setHealth(h.value.data)
      if (e.status === 'fulfilled') setEnginesHealth(e.value.data)
      if (er.status === 'fulfilled') setErrors(er.value.data)
      if (a.status === 'fulfilled') setActivity(a.value.data)
      if (d.status === 'fulfilled') setDeps(d.value.data)
    } catch { toast.error('Failed to load monitoring data') }
    finally { setLoading(false) }
  }

  const tabs = [
    { key: 'health', label: 'System Health', icon: Server },
    { key: 'engines', label: 'AI Engines', icon: Brain },
    { key: 'errors', label: 'Error Log', icon: AlertOctagon },
    { key: 'activity', label: 'Activity Feed', icon: Activity },
    { key: 'dependencies', label: 'Dependencies', icon: Package },
  ]

  if (loading) {
    return (
      <AdminLayout title="Monitoring">
        <div className="flex justify-center py-20"><Loader className="h-8 w-8 animate-spin text-primary-600" /></div>
      </AdminLayout>
    )
  }

  return (
    <AdminLayout title="Monitoring" subtitle="System health and operational status">
      <div className="space-y-6 animate-fade-in">
        {/* Top Status Bar */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            {
              icon: Server,
              label: 'System Status',
              value: health?.status === 'healthy' ? 'Healthy' : 'Degraded',
              color: health?.status === 'healthy' ? 'text-emerald-600' : 'text-red-600',
              bg: health?.status === 'healthy' ? 'bg-emerald-50 dark:bg-emerald-900/20' : 'bg-red-50 dark:bg-red-900/20'
            },
            {
              icon: Brain,
              label: 'AI Engines',
              value: `${enginesHealth?.available ?? 0}/${enginesHealth?.total_engines ?? 0}`,
              color: 'text-primary-600',
              bg: 'bg-primary-50 dark:bg-primary-900/20'
            },
            {
              icon: Cpu,
              label: 'CPU Usage',
              value: `${health?.server?.cpu_percent ?? 0}%`,
              color: (health?.server?.cpu_percent ?? 0) > 80 ? 'text-red-600' : 'text-blue-600',
              bg: (health?.server?.cpu_percent ?? 0) > 80 ? 'bg-red-50 dark:bg-red-900/20' : 'bg-blue-50 dark:bg-blue-900/20'
            },
            {
              icon: MemoryStick,
              label: 'Memory',
              value: `${health?.server?.memory_percent ?? 0}%`,
              color: (health?.server?.memory_percent ?? 0) > 85 ? 'text-red-600' : 'text-purple-600',
              bg: (health?.server?.memory_percent ?? 0) > 85 ? 'bg-red-50 dark:bg-red-900/20' : 'bg-purple-50 dark:bg-purple-900/20'
            },
          ].map((s, i) => (
            <div key={i} className={`card p-5 ${s.bg}`}>
              <s.icon className={`h-7 w-7 ${s.color} mb-2`} />
              <p className={`text-2xl font-display font-bold ${s.color}`}>{s.value}</p>
              <p className="text-xs text-surface-500 dark:text-surface-400 font-medium">{s.label}</p>
            </div>
          ))}
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-1 bg-surface-100 dark:bg-surface-800 rounded-xl p-1 overflow-x-auto">
          {tabs.map((tab) => (
            <button key={tab.key} onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${
                activeTab === tab.key
                  ? 'bg-white dark:bg-surface-700 text-primary-600 shadow-sm'
                  : 'text-surface-500 dark:text-surface-400 hover:text-surface-700 dark:hover:text-surface-200'
              }`}>
              <tab.icon className="h-3.5 w-3.5" />
              {tab.label}
            </button>
          ))}
          <div className="flex-1" />
          <button onClick={fetchAll} className="flex items-center gap-1 px-3 py-2 rounded-lg text-xs font-medium text-surface-500 hover:text-primary-600 transition-colors">
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'health' && <SystemHealthTab health={health} />}
        {activeTab === 'engines' && <AIEnginesHealthTab data={enginesHealth} />}
        {activeTab === 'errors' && <ErrorLogTab data={errors} />}
        {activeTab === 'activity' && <ActivityFeedTab data={activity} />}
        {activeTab === 'dependencies' && <DependenciesTab data={deps} />}
      </div>
    </AdminLayout>
  )
}


// ============================================================
// Tab: System Health
// ============================================================
const SystemHealthTab = ({ health }) => {
  if (!health) return <EmptyState message="System health data unavailable" />

  const server = health.server || {}
  const db = health.database || {}
  const storage = health.storage || {}

  return (
    <div className="space-y-6">
      {/* Server Resources */}
      <div className="card p-6">
        <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
          <Server className="h-5 w-5 text-blue-400" /> Server Resources
        </h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <ResourceGauge label="CPU" value={server.cpu_percent ?? 0} unit="%" warning={70} critical={90} />
          <ResourceGauge label="Memory" value={server.memory_percent ?? 0} unit="%" warning={75} critical={90}
            subtitle={`${server.memory_used_mb ?? 0} / ${server.memory_total_mb ?? 0} MB`} />
          <ResourceGauge label="Disk" value={server.disk_percent ?? 0} unit="%" warning={80} critical={95}
            subtitle={`${server.disk_used_gb ?? 0} / ${server.disk_total_gb ?? 0} GB`} />
          <div className="bg-surface-50 dark:bg-surface-700/50 rounded-xl p-4">
            <p className="text-xs text-surface-500 mb-1">Process Memory</p>
            <p className="text-xl font-display font-bold text-surface-900 dark:text-white">{server.process_memory_mb ?? 0} <span className="text-xs font-normal text-surface-400">MB</span></p>
          </div>
        </div>
      </div>

      {/* Server Info */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card p-6">
          <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
            <Info className="h-5 w-5 text-surface-400" /> Server Info
          </h3>
          <div className="space-y-2">
            {[
              ['Platform', `${server.platform} ${server.platform_version || ''}`],
              ['Python', server.python_version],
              ['Hostname', server.hostname],
              ['CPUs', server.cpu_count],
              ['Uptime', `${server.uptime_hours ?? 0} hours`],
            ].map(([k, v]) => (
              <div key={k} className="flex items-center justify-between py-1.5 border-b border-surface-100 dark:border-surface-700 last:border-0">
                <span className="text-xs text-surface-500">{k}</span>
                <span className="text-xs font-medium text-surface-900 dark:text-white">{v}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Database Stats */}
        <div className="card p-6">
          <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
            <Database className="h-5 w-5 text-purple-400" /> Database
          </h3>
          <div className="space-y-2">
            <div className="flex items-center justify-between py-1.5 border-b border-surface-100 dark:border-surface-700">
              <span className="text-xs text-surface-500">Engine</span>
              <span className="text-xs font-medium text-surface-900 dark:text-white">{db.engine}</span>
            </div>
            <div className="flex items-center justify-between py-1.5 border-b border-surface-100 dark:border-surface-700">
              <span className="text-xs text-surface-500">File Size</span>
              <span className="text-xs font-medium text-surface-900 dark:text-white">{db.file_size_mb} MB</span>
            </div>
            <div className="flex items-center justify-between py-1.5 border-b border-surface-100 dark:border-surface-700">
              <span className="text-xs text-surface-500">Total Records</span>
              <span className="text-xs font-medium text-surface-900 dark:text-white">{db.total_records?.toLocaleString()}</span>
            </div>
            {db.tables && Object.entries(db.tables).map(([table, count]) => (
              <div key={table} className="flex items-center justify-between py-1.5 border-b border-surface-100 dark:border-surface-700 last:border-0">
                <span className="text-xs text-surface-400 pl-3">{table}</span>
                <span className="text-xs text-surface-600 dark:text-surface-300">{count?.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Storage */}
      <div className="card p-6">
        <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
          <HardDrive className="h-5 w-5 text-amber-400" /> Storage Usage
        </h3>
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
          {[
            { label: 'Models', value: storage.models_mb, color: 'bg-primary-500' },
            { label: 'Data', value: storage.data_mb, color: 'bg-emerald-500' },
            { label: 'Uploads', value: storage.uploads_mb, color: 'bg-amber-500' },
            { label: 'Database', value: storage.database_mb, color: 'bg-purple-500' },
            { label: 'Total', value: storage.total_mb, color: 'bg-red-500' },
          ].map((s) => (
            <div key={s.label} className="bg-surface-50 dark:bg-surface-700/50 rounded-xl p-3">
              <div className={`w-2 h-2 ${s.color} rounded-full mb-2`} />
              <p className="text-lg font-display font-bold text-surface-900 dark:text-white">{s.value ?? 0} <span className="text-xs font-normal text-surface-400">MB</span></p>
              <p className="text-xs text-surface-500">{s.label}</p>
            </div>
          ))}
        </div>
      </div>

      {health.timestamp && (
        <p className="text-xs text-surface-400 text-right">
          Last checked: {new Date(health.timestamp).toLocaleString()}
        </p>
      )}
    </div>
  )
}


// ============================================================
// Tab: AI Engines Health
// ============================================================
const AIEnginesHealthTab = ({ data }) => {
  if (!data) return <EmptyState message="AI engines health data unavailable" />

  const statusColors = {
    available: 'text-emerald-600 bg-emerald-50 dark:bg-emerald-900/20',
    ready: 'text-emerald-600 bg-emerald-50 dark:bg-emerald-900/20',
    loaded_not_trained: 'text-amber-600 bg-amber-50 dark:bg-amber-900/20',
    error: 'text-red-600 bg-red-50 dark:bg-red-900/20',
  }

  const statusIcons = {
    available: CheckCircle,
    ready: CheckCircle,
    loaded_not_trained: AlertTriangle,
    error: XCircle,
  }

  return (
    <div className="space-y-4">
      {/* Health Summary */}
      <div className="card p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
            data.health_percentage >= 80 ? 'bg-emerald-50 dark:bg-emerald-900/20' : 'bg-amber-50 dark:bg-amber-900/20'
          }`}>
            <Gauge className={`h-5 w-5 ${data.health_percentage >= 80 ? 'text-emerald-600' : 'text-amber-600'}`} />
          </div>
          <div>
            <p className="text-sm font-semibold text-surface-900 dark:text-white">Overall Health: {data.health_percentage}%</p>
            <p className="text-xs text-surface-500">{data.available} of {data.total_engines} engines operational</p>
          </div>
        </div>
        <p className="text-xs text-surface-400">{data.timestamp ? new Date(data.timestamp).toLocaleString() : ''}</p>
      </div>

      {/* Engine Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {(data.engines || []).map((engine, i) => {
          const StatusIcon = statusIcons[engine.status] || AlertTriangle
          const colorClass = statusColors[engine.status] || 'text-surface-500 bg-surface-50 dark:bg-surface-700'
          return (
            <div key={i} className="card p-4">
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${colorClass}`}>
                    <StatusIcon className="h-4 w-4" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-surface-900 dark:text-white">{engine.name}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${colorClass}`}>
                        {engine.status}
                      </span>
                      {engine.is_real_ai && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-primary-50 dark:bg-primary-900/20 text-primary-600">
                          Real AI
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-xs text-surface-500">{engine.latency_ms} ms</p>
                </div>
              </div>
              {engine.error && (
                <div className="mt-2 p-2 rounded-lg bg-red-50 dark:bg-red-900/20">
                  <p className="text-xs text-red-600 dark:text-red-400 line-clamp-2">{engine.error}</p>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}


// ============================================================
// Tab: Error Log
// ============================================================
const ErrorLogTab = ({ data }) => {
  if (!data) return <EmptyState message="Error log data unavailable" />

  return (
    <div className="space-y-3">
      <div className="card p-4">
        <p className="text-sm font-semibold text-surface-900 dark:text-white flex items-center gap-2">
          <AlertOctagon className="h-4 w-4 text-red-500" />
          {data.total_errors} error{data.total_errors !== 1 ? 's' : ''} found
        </p>
      </div>

      {data.total_errors > 0 ? (
        <div className="space-y-2">
          {data.errors.map((err) => (
            <div key={err.id} className="card p-4 flex items-start gap-3">
              <div className="w-8 h-8 bg-red-100 dark:bg-red-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                <AlertTriangle className="h-4 w-4 text-red-600" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-surface-900 dark:text-white">{err.action}</p>
                  <span className="text-xs text-surface-500 flex items-center gap-1">
                    <Calendar className="h-3 w-3" /> {err.created_at ? new Date(err.created_at).toLocaleString() : ''}
                  </span>
                </div>
                {err.entity_type && (
                  <p className="text-xs text-surface-500 mt-0.5">Entity: {err.entity_type}</p>
                )}
                {err.details && (
                  <p className="text-xs text-surface-400 mt-1 line-clamp-2">
                    {typeof err.details === 'string' ? err.details : JSON.stringify(err.details)}
                  </p>
                )}
                <div className="flex items-center gap-3 mt-1">
                  {err.user_id && <span className="text-xs text-surface-400">User ID: {err.user_id}</span>}
                  {err.ip_address && <span className="text-xs text-surface-400">IP: {err.ip_address}</span>}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="card text-center py-16">
          <CheckCircle className="h-16 w-16 text-emerald-300 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-surface-600 mb-2">No Errors</h3>
          <p className="text-sm text-surface-400">System is running without detected errors</p>
        </div>
      )}
    </div>
  )
}


// ============================================================
// Tab: Activity Feed
// ============================================================
const ActivityFeedTab = ({ data }) => {
  if (!data) return <EmptyState message="Activity feed unavailable" />

  const actionColors = {
    login: 'bg-blue-100 dark:bg-blue-900/30 text-blue-600',
    register: 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600',
    analyze: 'bg-purple-100 dark:bg-purple-900/30 text-purple-600',
    feedback: 'bg-amber-100 dark:bg-amber-900/30 text-amber-600',
    login_failed: 'bg-red-100 dark:bg-red-900/30 text-red-600',
  }

  const getActionColor = (action) => {
    for (const [key, cls] of Object.entries(actionColors)) {
      if (action?.toLowerCase().includes(key)) return cls
    }
    return 'bg-surface-100 dark:bg-surface-700 text-surface-600'
  }

  return (
    <div className="space-y-2">
      <div className="card p-4">
        <p className="text-sm font-semibold text-surface-900 dark:text-white flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary-500" />
          {data.total} recent activities
        </p>
      </div>

      {(data.activities || []).length > 0 ? (
        <div className="space-y-2">
          {data.activities.map((act) => (
            <div key={act.id} className="card p-4 flex items-start gap-3">
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${getActionColor(act.action)}`}>
                <Zap className="h-4 w-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-surface-900 dark:text-white">{act.action}</p>
                  <span className="text-xs text-surface-500 flex items-center gap-1">
                    <Clock className="h-3 w-3" /> {act.created_at ? new Date(act.created_at).toLocaleString() : ''}
                  </span>
                </div>
                <div className="flex items-center gap-3 mt-1">
                  {act.user_email && (
                    <span className="text-xs text-surface-500 flex items-center gap-1">
                      <User className="h-3 w-3" /> {act.user_email}
                    </span>
                  )}
                  {act.entity_type && (
                    <span className="text-xs text-surface-400">{act.entity_type}</span>
                  )}
                </div>
                {act.details && (
                  <p className="text-xs text-surface-400 mt-1 line-clamp-1">
                    {typeof act.details === 'string' ? act.details : JSON.stringify(act.details)}
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="card text-center py-16">
          <Activity className="h-16 w-16 text-surface-300 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-surface-600 mb-2">No Activity</h3>
        </div>
      )}
    </div>
  )
}


// ============================================================
// Tab: Dependencies
// ============================================================
const DependenciesTab = ({ data }) => {
  if (!data) return <EmptyState message="Dependency data unavailable" />

  return (
    <div className="space-y-4">
      {/* Python & System */}
      <div className="card p-6">
        <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
          <Info className="h-5 w-5 text-surface-400" /> Runtime Environment
        </h3>
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: 'Python', value: data.python_version },
            { label: 'System', value: data.system },
            { label: 'Architecture', value: data.architecture },
          ].map((item) => (
            <div key={item.label} className="bg-surface-50 dark:bg-surface-700/50 rounded-xl p-3">
              <p className="text-xs text-surface-500 mb-1">{item.label}</p>
              <p className="text-sm font-semibold text-surface-900 dark:text-white">{item.value || 'N/A'}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Package List */}
      <div className="card p-6">
        <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
          <Package className="h-5 w-5 text-purple-400" /> Installed Packages
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
          {Object.entries(data.dependencies || {}).map(([pkg, version]) => (
            <div key={pkg} className="flex items-center justify-between py-2 px-3 rounded-lg bg-surface-50 dark:bg-surface-700/50">
              <div className="flex items-center gap-2">
                {version ? (
                  <CheckCircle className="h-3.5 w-3.5 text-emerald-500" />
                ) : (
                  <XCircle className="h-3.5 w-3.5 text-red-400" />
                )}
                <span className="text-sm font-medium text-surface-900 dark:text-white">{pkg}</span>
              </div>
              <span className={`text-xs font-mono ${version ? 'text-surface-600 dark:text-surface-300' : 'text-red-400'}`}>
                {version || 'not installed'}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}


// ============================================================
// Shared Components
// ============================================================
const ResourceGauge = ({ label, value, unit, warning, critical, subtitle }) => {
  const color = value >= critical ? 'text-red-600' : value >= warning ? 'text-amber-600' : 'text-emerald-600'
  const barColor = value >= critical ? 'bg-red-500' : value >= warning ? 'bg-amber-500' : 'bg-emerald-500'

  return (
    <div className="bg-surface-50 dark:bg-surface-700/50 rounded-xl p-4">
      <p className="text-xs text-surface-500 mb-1">{label}</p>
      <p className={`text-xl font-display font-bold ${color}`}>{value}{unit}</p>
      <div className="h-1.5 bg-surface-200 dark:bg-surface-600 rounded-full overflow-hidden mt-2">
        <div className={`h-full ${barColor} rounded-full transition-all duration-500`} style={{ width: `${Math.min(value, 100)}%` }} />
      </div>
      {subtitle && <p className="text-[10px] text-surface-400 mt-1">{subtitle}</p>}
    </div>
  )
}

const EmptyState = ({ message }) => (
  <div className="card text-center py-16">
    <AlertTriangle className="h-16 w-16 text-surface-300 mx-auto mb-4" />
    <h3 className="text-lg font-semibold text-surface-600 mb-2">{message}</h3>
  </div>
)

export default AdminMonitoringNew
