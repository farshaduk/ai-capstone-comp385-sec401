import { useState, useEffect } from 'react'
import AdminLayout from '../../components/layouts/AdminLayout'
import { adminAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  CreditCard, Loader, Plus, Edit3, Trash2, Check, X, Zap, Crown, Rocket, Shield, Save
} from 'lucide-react'

const AdminPlans = () => {
  const [plans, setPlans] = useState([])
  const [loading, setLoading] = useState(true)

  // Create / Edit modal
  const [showForm, setShowForm] = useState(false)
  const [editingPlan, setEditingPlan] = useState(null)
  const [form, setForm] = useState({
    name: '', display_name: '', price: 0, scans_per_month: 10,
    features: {
      text_analysis: true, url_analysis: false, image_analysis: false,
      message_analysis: false, xai_explanations: false, export_reports: false,
      priority_support: false, api_access: false,
    },
  })
  const [saving, setSaving] = useState(false)

  useEffect(() => { fetchPlans() }, [])

  const fetchPlans = async () => {
    try {
      const res = await adminAPI.getSubscriptionPlans()
      setPlans(res.data)
    } catch { toast.error('Failed to fetch plans') }
    finally { setLoading(false) }
  }

  const handleDelete = async (id) => {
    if (!confirm('Delete this plan?')) return
    try {
      await adminAPI.deleteSubscriptionPlan(id)
      toast.success('Plan deleted')
      fetchPlans()
    } catch { toast.error('Failed to delete plan') }
  }

  const openCreateForm = () => {
    setEditingPlan(null)
    setForm({
      name: '', display_name: '', price: 0, scans_per_month: 10,
      features: {
        text_analysis: true, url_analysis: false, image_analysis: false,
        message_analysis: false, xai_explanations: false, export_reports: false,
        priority_support: false, api_access: false,
      },
    })
    setShowForm(true)
  }

  const openEditForm = (plan) => {
    setEditingPlan(plan)
    setForm({
      name: plan.name || '',
      display_name: plan.display_name || '',
      price: plan.price || 0,
      scans_per_month: plan.scans_per_month || 10,
      features: plan.features || {},
    })
    setShowForm(true)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.name.trim() || !form.display_name.trim()) return toast.error('Name and display name required')
    setSaving(true)
    try {
      if (editingPlan) {
        await adminAPI.updateSubscriptionPlan(editingPlan.id, form)
        toast.success('Plan updated!')
      } else {
        await adminAPI.createSubscriptionPlan(form)
        toast.success('Plan created!')
      }
      setShowForm(false)
      fetchPlans()
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to save plan')
    } finally { setSaving(false) }
  }

  const toggleFeature = (key) => {
    setForm({ ...form, features: { ...form.features, [key]: !form.features[key] } })
  }

  const getPlanIcon = (name) => {
    const icons = { free: Zap, basic: Shield, premium: Crown, enterprise: Rocket }
    return icons[name] || CreditCard
  }

  const featureLabels = {
    text_analysis: 'Text Analysis',
    url_analysis: 'URL Analysis',
    image_analysis: 'Image Verification',
    message_analysis: 'Message Analysis',
    xai_explanations: 'XAI Explanations',
    export_reports: 'Export Reports',
    priority_support: 'Priority Support',
    api_access: 'API Access',
  }

  return (
    <AdminLayout title="Subscription Plans" subtitle="Configure pricing tiers and features">
      <div className="space-y-6">
        {/* Actions */}
        <div className="flex justify-end">
          <button onClick={openCreateForm} className="btn btn-primary">
            <Plus className="h-4 w-4" /> Create Plan
          </button>
        </div>

        {/* Plans Grid */}
        {loading ? (
          <div className="flex justify-center py-16"><Loader className="h-8 w-8 animate-spin text-primary-600" /></div>
        ) : plans.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {plans.map((plan) => {
              const Icon = getPlanIcon(plan.name)
              return (
                <div key={plan.id} className="card p-5">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-primary-100 dark:bg-primary-900/30 rounded-xl flex items-center justify-center">
                        <Icon className="h-5 w-5 text-primary-600" />
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-surface-900 dark:text-white">{plan.display_name}</p>
                        <p className="text-xs text-surface-500">{plan.name}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <button onClick={() => openEditForm(plan)} className="btn btn-sm btn-ghost text-primary-600">
                        <Edit3 className="h-3.5 w-3.5" />
                      </button>
                      <button onClick={() => handleDelete(plan.id)} className="btn btn-sm btn-ghost text-red-500">
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                  <div className="mb-3">
                    <span className="text-2xl font-display font-bold text-surface-900 dark:text-white">${plan.price}</span>
                    <span className="text-sm text-surface-500">/mo</span>
                  </div>
                  <p className="text-xs text-surface-500 mb-3">
                    {plan.scans_per_month === -1 ? 'Unlimited' : plan.scans_per_month} scans/month
                  </p>
                  {plan.features && (
                    <div className="space-y-1">
                      {Object.entries(plan.features).filter(([, v]) => v).map(([key]) => (
                        <div key={key} className="flex items-center gap-2 text-xs text-surface-600 dark:text-surface-300">
                          <Check className="h-3 w-3 text-emerald-500" /> {featureLabels[key] || key.replace(/_/g, ' ')}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        ) : (
          <div className="card text-center py-16">
            <CreditCard className="h-16 w-16 text-surface-300 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-surface-600 mb-2">No plans configured</h3>
            <p className="text-surface-500 mb-4">Create your first subscription plan</p>
            <button onClick={openCreateForm} className="btn btn-primary"><Plus className="h-4 w-4" /> Create Plan</button>
          </div>
        )}
      </div>

      {/* Create / Edit Modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-surface-800 rounded-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto p-6 shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white">
                {editingPlan ? 'Edit Plan' : 'Create Plan'}
              </h3>
              <button onClick={() => setShowForm(false)} className="btn btn-ghost btn-sm"><X className="h-5 w-5" /></button>
            </div>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="input-label">Plan Name (slug)</label>
                  <input type="text" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })}
                    className="input-field" placeholder="e.g., premium" required disabled={!!editingPlan} />
                </div>
                <div>
                  <label className="input-label">Display Name</label>
                  <input type="text" value={form.display_name} onChange={e => setForm({ ...form, display_name: e.target.value })}
                    className="input-field" placeholder="e.g., Premium Plan" required />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="input-label">Price ($/month)</label>
                  <input type="number" min={0} step={0.01} value={form.price}
                    onChange={e => setForm({ ...form, price: parseFloat(e.target.value) || 0 })}
                    className="input-field" />
                </div>
                <div>
                  <label className="input-label">Scans/Month (-1 = unlimited)</label>
                  <input type="number" min={-1} value={form.scans_per_month}
                    onChange={e => setForm({ ...form, scans_per_month: parseInt(e.target.value) || 0 })}
                    className="input-field" />
                </div>
              </div>

              <div>
                <label className="input-label mb-3">Included Features</label>
                <div className="space-y-2">
                  {Object.entries(featureLabels).map(([key, label]) => (
                    <button key={key} type="button" onClick={() => toggleFeature(key)}
                      className={`w-full flex items-center justify-between p-3 rounded-xl border transition-colors ${
                        form.features[key]
                          ? 'border-primary-300 bg-primary-50 dark:bg-primary-900/20 dark:border-primary-700'
                          : 'border-surface-200 dark:border-surface-700 hover:bg-surface-50 dark:hover:bg-surface-700/50'
                      }`}>
                      <span className="text-sm text-surface-700 dark:text-surface-300">{label}</span>
                      <div className={`w-5 h-5 rounded-md flex items-center justify-center ${
                        form.features[key] ? 'bg-primary-600 text-white' : 'bg-surface-200 dark:bg-surface-600'
                      }`}>
                        {form.features[key] && <Check className="h-3 w-3" />}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex gap-2 pt-2">
                <button type="submit" disabled={saving} className="btn btn-primary flex-1">
                  {saving ? <><Loader className="h-4 w-4 animate-spin" /> Saving...</> : <><Save className="h-4 w-4" /> {editingPlan ? 'Update' : 'Create'} Plan</>}
                </button>
                <button type="button" onClick={() => setShowForm(false)} className="btn btn-secondary">Cancel</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </AdminLayout>
  )
}

export default AdminPlans
