import { useState, useEffect } from 'react'
import AdminLayout from '../../components/layouts/AdminLayout'
import { adminAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  Settings, Loader, Sliders, Shield, Server,
  Save, RotateCcw, Info, AlertTriangle, Lock,
  Clock, Database, Cpu, HardDrive, CheckCircle
} from 'lucide-react'

const AdminSettings = () => {
  const [settings, setSettings] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [tab, setTab] = useState('risk')
  const [riskForm, setRiskForm] = useState(null)
  const [hasChanges, setHasChanges] = useState(false)

  useEffect(() => { fetchSettings() }, [])

  const fetchSettings = async () => {
    try {
      const res = await adminAPI.getSettings()
      setSettings(res.data)
      setRiskForm({
        risk_base_thresholds: [...res.data.risk_tuning.risk_base_thresholds],
        risk_severity_shift_coefficient: res.data.risk_tuning.risk_severity_shift_coefficient,
        risk_confidence_shift_coefficient: res.data.risk_tuning.risk_confidence_shift_coefficient,
        risk_max_threshold_shift: res.data.risk_tuning.risk_max_threshold_shift,
        risk_severity_baseline: res.data.risk_tuning.risk_severity_baseline,
        risk_confidence_baseline: res.data.risk_tuning.risk_confidence_baseline,
      })
      setHasChanges(false)
    } catch { toast.error('Failed to load settings') }
    finally { setLoading(false) }
  }

  const handleRiskChange = (key, value) => {
    setRiskForm(prev => ({ ...prev, [key]: value }))
    setHasChanges(true)
  }

  const handleThresholdChange = (index, value) => {
    const newThresholds = [...riskForm.risk_base_thresholds]
    newThresholds[index] = parseFloat(value) || 0
    setRiskForm(prev => ({ ...prev, risk_base_thresholds: newThresholds }))
    setHasChanges(true)
  }

  const handleSaveRisk = async () => {
    // Validate ascending order
    const t = riskForm.risk_base_thresholds
    for (let i = 0; i < t.length - 1; i++) {
      if (t[i] >= t[i + 1]) {
        toast.error('Thresholds must be in ascending order')
        return
      }
    }
    setSaving(true)
    try {
      await adminAPI.updateRiskTuning(riskForm)
      toast.success('Risk tuning settings saved')
      await fetchSettings()
    } catch (e) {
      toast.error(e.response?.data?.detail || 'Failed to save settings')
    } finally { setSaving(false) }
  }

  const handleReset = () => {
    if (!settings) return
    setRiskForm({
      risk_base_thresholds: [...settings.risk_tuning.risk_base_thresholds],
      risk_severity_shift_coefficient: settings.risk_tuning.risk_severity_shift_coefficient,
      risk_confidence_shift_coefficient: settings.risk_tuning.risk_confidence_shift_coefficient,
      risk_max_threshold_shift: settings.risk_tuning.risk_max_threshold_shift,
      risk_severity_baseline: settings.risk_tuning.risk_severity_baseline,
      risk_confidence_baseline: settings.risk_tuning.risk_confidence_baseline,
    })
    setHasChanges(false)
    toast('Reset to saved values', { icon: '↩️' })
  }

  if (loading) {
    return (
      <AdminLayout title="Settings" subtitle="System configuration">
        <div className="flex justify-center py-20"><Loader className="h-8 w-8 animate-spin text-primary-600" /></div>
      </AdminLayout>
    )
  }

  const tabs = [
    { id: 'risk', label: 'Risk Tuning', icon: Sliders },
    { id: 'auth', label: 'Session & Auth', icon: Shield },
    { id: 'system', label: 'System Info', icon: Server },
  ]

  const thresholdLabels = [
    { label: 'Very Low → Low', color: 'text-emerald-600' },
    { label: 'Low → Medium', color: 'text-yellow-600' },
    { label: 'Medium → High', color: 'text-orange-600' },
    { label: 'High → Very High', color: 'text-red-600' },
  ]

  const coefficientFields = [
    {
      key: 'risk_severity_shift_coefficient',
      label: 'Severity Shift Coefficient',
      icon: AlertTriangle,
      color: 'text-orange-600',
      bg: 'bg-orange-50 dark:bg-orange-900/20',
      min: 0.05, max: 0.30, step: 0.01,
    },
    {
      key: 'risk_confidence_shift_coefficient',
      label: 'Confidence Shift Coefficient',
      icon: CheckCircle,
      color: 'text-blue-600',
      bg: 'bg-blue-50 dark:bg-blue-900/20',
      min: 0.05, max: 0.25, step: 0.01,
    },
    {
      key: 'risk_max_threshold_shift',
      label: 'Max Threshold Shift',
      icon: Sliders,
      color: 'text-purple-600',
      bg: 'bg-purple-50 dark:bg-purple-900/20',
      min: 0.10, max: 0.30, step: 0.01,
    },
    {
      key: 'risk_severity_baseline',
      label: 'Severity Baseline',
      icon: AlertTriangle,
      color: 'text-amber-600',
      bg: 'bg-amber-50 dark:bg-amber-900/20',
      min: 0.20, max: 0.60, step: 0.05,
    },
    {
      key: 'risk_confidence_baseline',
      label: 'Confidence Baseline',
      icon: CheckCircle,
      color: 'text-cyan-600',
      bg: 'bg-cyan-50 dark:bg-cyan-900/20',
      min: 0.50, max: 0.90, step: 0.05,
    },
  ]

  return (
    <AdminLayout title="Settings" subtitle="System configuration and risk tuning">
      <div className="space-y-6 animate-fade-in">
        {/* Tabs */}
        <div className="flex items-center gap-2">
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors flex items-center gap-2 ${tab === t.id
                ? 'bg-primary-600 text-white'
                : 'bg-surface-100 dark:bg-surface-800 text-surface-600 dark:text-surface-400 hover:bg-surface-200 dark:hover:bg-surface-700'}`}>
              <t.icon className="h-4 w-4" /> {t.label}
            </button>
          ))}
        </div>

        {/* ── Tab 1: Risk Tuning ── */}
        {tab === 'risk' && riskForm && (
          <div className="space-y-6">
            {/* Info banner */}
            <div className="card p-4 border-l-4 border-primary-500 bg-primary-50/50 dark:bg-primary-900/10">
              <div className="flex items-start gap-3">
                <Info className="h-5 w-5 text-primary-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-surface-900 dark:text-white">Risk Level Determination</p>
                  <p className="text-xs text-surface-500 dark:text-surface-400 mt-1">
                    These settings control how the system determines fraud risk levels. The final composite score is mapped to risk levels
                    using adaptive thresholds that shift based on indicator severity and model confidence.
                  </p>
                </div>
              </div>
            </div>

            {/* Base Thresholds */}
            <div className="card p-6">
              <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-1 flex items-center gap-2">
                <Sliders className="h-5 w-5 text-surface-400" /> Base Risk Thresholds
              </h3>
              <p className="text-xs text-surface-500 dark:text-surface-400 mb-5">
                Boundaries between risk levels. Lower values = stricter (more listings flagged as risky).
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {riskForm.risk_base_thresholds.map((val, i) => (
                  <div key={i} className="space-y-2">
                    <label className="text-xs font-medium text-surface-600 dark:text-surface-300 flex items-center gap-1">
                      <span className={`w-2 h-2 rounded-full ${i === 0 ? 'bg-emerald-500' : i === 1 ? 'bg-yellow-500' : i === 2 ? 'bg-orange-500' : 'bg-red-500'}`} />
                      {thresholdLabels[i].label}
                    </label>
                    <input
                      type="range" min="0.05" max="0.95" step="0.01"
                      value={val}
                      onChange={(e) => handleThresholdChange(i, e.target.value)}
                      className="w-full accent-primary-600"
                    />
                    <div className="flex items-center gap-2">
                      <input
                        type="number" min="0.05" max="0.95" step="0.01"
                        value={val}
                        onChange={(e) => handleThresholdChange(i, e.target.value)}
                        className="input-field text-center text-sm font-mono w-full"
                      />
                    </div>
                  </div>
                ))}
              </div>
              {/* Visual threshold bar */}
              <div className="mt-5">
                <div className="h-3 rounded-full overflow-hidden flex">
                  <div style={{ width: `${riskForm.risk_base_thresholds[0] * 100}%` }} className="bg-emerald-400" title="Very Low" />
                  <div style={{ width: `${(riskForm.risk_base_thresholds[1] - riskForm.risk_base_thresholds[0]) * 100}%` }} className="bg-yellow-400" title="Low" />
                  <div style={{ width: `${(riskForm.risk_base_thresholds[2] - riskForm.risk_base_thresholds[1]) * 100}%` }} className="bg-orange-400" title="Medium" />
                  <div style={{ width: `${(riskForm.risk_base_thresholds[3] - riskForm.risk_base_thresholds[2]) * 100}%` }} className="bg-red-400" title="High" />
                  <div style={{ width: `${(1 - riskForm.risk_base_thresholds[3]) * 100}%` }} className="bg-red-700" title="Very High" />
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-[10px] text-emerald-600 font-medium">Very Low</span>
                  <span className="text-[10px] text-yellow-600 font-medium">Low</span>
                  <span className="text-[10px] text-orange-600 font-medium">Medium</span>
                  <span className="text-[10px] text-red-600 font-medium">High</span>
                  <span className="text-[10px] text-red-800 font-medium">Very High</span>
                </div>
              </div>
            </div>

            {/* Coefficients */}
            <div className="card p-6">
              <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-1 flex items-center gap-2">
                <Settings className="h-5 w-5 text-surface-400" /> Shift Coefficients & Baselines
              </h3>
              <p className="text-xs text-surface-500 dark:text-surface-400 mb-5">
                Control how severity and confidence dynamically shift the thresholds above.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {coefficientFields.map((field) => (
                  <div key={field.key} className={`rounded-xl p-4 ${field.bg}`}>
                    <div className="flex items-center gap-2 mb-3">
                      <field.icon className={`h-4 w-4 ${field.color}`} />
                      <label className="text-sm font-medium text-surface-900 dark:text-white">{field.label}</label>
                    </div>
                    <input
                      type="range" min={field.min} max={field.max} step={field.step}
                      value={riskForm[field.key]}
                      onChange={(e) => handleRiskChange(field.key, parseFloat(e.target.value))}
                      className="w-full accent-primary-600 mb-2"
                    />
                    <div className="flex items-center justify-between">
                      <input
                        type="number" min={field.min} max={field.max} step={field.step}
                        value={riskForm[field.key]}
                        onChange={(e) => handleRiskChange(field.key, parseFloat(e.target.value) || 0)}
                        className="input-field text-center text-sm font-mono w-20"
                      />
                      <p className="text-[10px] text-surface-400">{settings?.risk_tuning?.descriptions?.[field.key]?.split('.')[0] || ''}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Save / Reset buttons */}
            <div className="flex items-center gap-3">
              <button onClick={handleSaveRisk} disabled={saving || !hasChanges}
                className="btn btn-primary btn-md disabled:opacity-50">
                {saving ? <Loader className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                Save Risk Settings
              </button>
              <button onClick={handleReset} disabled={!hasChanges}
                className="btn btn-ghost btn-md disabled:opacity-50">
                <RotateCcw className="h-4 w-4" /> Reset
              </button>
              {hasChanges && (
                <span className="text-xs text-amber-600 flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3" /> Unsaved changes
                </span>
              )}
            </div>
          </div>
        )}

        {/* ── Tab 2: Session & Auth ── */}
        {tab === 'auth' && settings && (
          <div className="space-y-6">
            <div className="card p-4 border-l-4 border-blue-500 bg-blue-50/50 dark:bg-blue-900/10">
              <div className="flex items-start gap-3">
                <Info className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <p className="text-xs text-surface-500 dark:text-surface-400">
                  These values are set in <code className="text-xs bg-surface-200 dark:bg-surface-700 px-1 py-0.5 rounded">config.py</code> or <code className="text-xs bg-surface-200 dark:bg-surface-700 px-1 py-0.5 rounded">.env</code>. Changes require a backend restart.
                </p>
              </div>
            </div>

            <div className="card p-0 overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-surface-200 dark:border-surface-700 bg-surface-50 dark:bg-surface-800">
                    {['Setting', 'Value', 'Description'].map(h => (
                      <th key={h} className="text-left py-3 px-4 text-xs font-medium text-surface-500 dark:text-surface-400 uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    {
                      icon: Clock, label: 'Token Expiry',
                      value: `${settings.session_auth.access_token_expire_minutes} minutes`,
                      desc: 'How long a JWT access token remains valid before the user must re-authenticate'
                    },
                    {
                      icon: Lock, label: 'Algorithm',
                      value: settings.session_auth.algorithm,
                      desc: 'JWT signing algorithm used for authentication tokens'
                    },
                    {
                      icon: Server, label: 'Environment',
                      value: settings.session_auth.environment,
                      desc: 'Current deployment environment (development / staging / production)'
                    },
                  ].map((row, i) => (
                    <tr key={i} className="border-b border-surface-100 dark:border-surface-700/50">
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <row.icon className="h-4 w-4 text-blue-500" />
                          <span className="text-sm font-medium text-surface-900 dark:text-white">{row.label}</span>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className="font-mono text-sm text-primary-600 dark:text-primary-400 bg-primary-50 dark:bg-primary-900/20 px-2 py-1 rounded">
                          {row.value}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-xs text-surface-500 dark:text-surface-400">{row.desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── Tab 3: System Info ── */}
        {tab === 'system' && settings && (
          <div className="space-y-6">
            {/* System Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { icon: Cpu, label: 'CPU Cores', value: settings.system_info.cpu_count, color: 'text-blue-600', bg: 'bg-blue-50 dark:bg-blue-900/20' },
                { icon: HardDrive, label: 'Memory', value: `${settings.system_info.memory_total_gb} GB`, color: 'text-purple-600', bg: 'bg-purple-50 dark:bg-purple-900/20' },
                { icon: Database, label: 'DB Size', value: `${settings.system_info.database_size_mb} MB`, color: 'text-emerald-600', bg: 'bg-emerald-50 dark:bg-emerald-900/20' },
                { icon: Clock, label: 'Uptime', value: `${settings.system_info.uptime_hours} hrs`, color: 'text-amber-600', bg: 'bg-amber-50 dark:bg-amber-900/20' },
              ].map((s, i) => (
                <div key={i} className={`card p-5 ${s.bg}`}>
                  <s.icon className={`h-7 w-7 ${s.color} mb-2`} />
                  <p className={`text-2xl font-display font-bold ${s.color}`}>{s.value}</p>
                  <p className="text-xs text-surface-500 dark:text-surface-400 font-medium">{s.label}</p>
                </div>
              ))}
            </div>

            {/* Details Table */}
            <div className="card p-0 overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-surface-200 dark:border-surface-700 bg-surface-50 dark:bg-surface-800">
                    {['Property', 'Value'].map(h => (
                      <th key={h} className="text-left py-3 px-4 text-xs font-medium text-surface-500 dark:text-surface-400 uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    { label: 'Python Version', value: settings.system_info.python_version },
                    { label: 'Platform', value: settings.system_info.platform },
                    { label: 'OS', value: settings.system_info.os },
                    { label: 'Architecture', value: settings.system_info.architecture },
                    { label: 'Database Engine', value: settings.system_info.database_engine },
                    { label: 'Database Path', value: settings.system_info.database_path },
                    { label: 'Backend Directory', value: settings.system_info.backend_dir },
                  ].map((row, i) => (
                    <tr key={i} className="border-b border-surface-100 dark:border-surface-700/50">
                      <td className="py-3 px-4 text-sm font-medium text-surface-900 dark:text-white">{row.label}</td>
                      <td className="py-3 px-4 font-mono text-xs text-surface-600 dark:text-surface-300 break-all">{row.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </AdminLayout>
  )
}

export default AdminSettings
