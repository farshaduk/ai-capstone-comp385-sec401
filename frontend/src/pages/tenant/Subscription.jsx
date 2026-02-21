import { useState, useEffect } from 'react'
import TenantLayout from '../../components/layouts/TenantLayout'
import { renterAPI } from '../../services/api'
import { useAuthStore } from '../../store/authStore'
import toast from 'react-hot-toast'
import { Check, Zap, Crown, Rocket, CreditCard, Clock, X, Loader, Shield } from 'lucide-react'

const TenantSubscription = () => {
  const [plans, setPlans] = useState([])
  const [loading, setLoading] = useState(true)
  const [paymentHistory, setPaymentHistory] = useState([])
  const [showPaymentModal, setShowPaymentModal] = useState(false)
  const [selectedPlan, setSelectedPlan] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [paymentForm, setPaymentForm] = useState({
    card_number: '', expiry_month: '', expiry_year: '', cvv: '', cardholder_name: ''
  })
  const { user, updateUser } = useAuthStore()

  useEffect(() => {
    fetchPlans()
    fetchPaymentHistory()
  }, [])

  const fetchPlans = async () => {
    try {
      const response = await renterAPI.getPlans()
      setPlans(response.data)
    } catch (error) {
      toast.error('Failed to fetch plans')
    } finally {
      setLoading(false)
    }
  }

  const fetchPaymentHistory = async () => {
    try {
      const response = await renterAPI.getPaymentHistory()
      setPaymentHistory(response.data)
    } catch { /* optional */ }
  }

  const handleUpgrade = (plan) => {
    if (user.subscription_plan === plan.name) return toast.error('Already on this plan')
    if (plan.price === 0) return handleFreePlanUpgrade(plan.name)
    setSelectedPlan(plan)
    setShowPaymentModal(true)
  }

  const handleFreePlanUpgrade = async (planName) => {
    try {
      const response = await renterAPI.upgradePlan(planName)
      updateUser(response.data)
      toast.success(`Switched to ${planName} plan!`)
    } catch { toast.error('Failed to change subscription') }
  }

  const handlePayment = async (e) => {
    e.preventDefault()
    setProcessing(true)
    try {
      const response = await renterAPI.processPayment({
        plan_name: selectedPlan.name,
        card_number: paymentForm.card_number.replace(/\s/g, ''),
        expiry_month: parseInt(paymentForm.expiry_month),
        expiry_year: parseInt(paymentForm.expiry_year),
        cvv: paymentForm.cvv,
        cardholder_name: paymentForm.cardholder_name,
      })
      toast.success(response.data.message)
      updateUser({ ...user, subscription_plan: selectedPlan.name, scans_remaining: selectedPlan.scans_per_month })
      setShowPaymentModal(false)
      setSelectedPlan(null)
      setPaymentForm({ card_number: '', expiry_month: '', expiry_year: '', cvv: '', cardholder_name: '' })
      fetchPaymentHistory()
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Payment failed')
    } finally {
      setProcessing(false)
    }
  }

  const formatCardNumber = (v) => {
    const clean = v.replace(/\D/g, '').substring(0, 16)
    return clean.replace(/(.{4})/g, '$1 ').trim()
  }

  const planConfig = {
    free: { icon: Zap, gradient: 'from-surface-400 to-surface-500' },
    basic: { icon: Shield, gradient: 'from-blue-500 to-blue-600' },
    premium: { icon: Crown, gradient: 'from-purple-500 to-purple-600' },
    enterprise: { icon: Rocket, gradient: 'from-primary-500 to-primary-700' },
  }

  return (
    <TenantLayout title="Subscription" subtitle="Manage your plan and billing">
      <div className="space-y-8">
        {/* Current Plan */}
        <div className="card p-6 bg-gradient-to-r from-primary-600 to-primary-800">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-primary-200 text-sm mb-1">Current Plan</p>
              <p className="text-2xl font-display font-bold text-white capitalize">{user?.subscription_plan}</p>
              <p className="text-primary-200 text-sm mt-1">{user?.scans_remaining} scans remaining</p>
            </div>
            <Crown className="h-14 w-14 text-white/20" />
          </div>
        </div>

        {/* Plans */}
        {loading ? (
          <div className="flex justify-center py-12"><Loader className="h-8 w-8 animate-spin text-primary-600" /></div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {plans.map((plan) => {
              const cfg = planConfig[plan.name] || planConfig.free
              const Icon = cfg.icon
              const isCurrent = user?.subscription_plan === plan.name
              return (
                <div key={plan.id} className={`card p-0 overflow-hidden ${isCurrent ? 'ring-2 ring-primary-500' : ''}`}>
                  {isCurrent && (
                    <div className="bg-primary-500 text-white text-center py-1 text-xs font-bold">Current Plan</div>
                  )}
                  <div className={`bg-gradient-to-r ${cfg.gradient} p-4 text-white`}>
                    <Icon className="h-8 w-8 mb-2" />
                    <h3 className="text-lg font-display font-bold">{plan.display_name}</h3>
                  </div>
                  <div className="p-5">
                    <div className="mb-4">
                      <span className="text-3xl font-display font-bold text-surface-900 dark:text-white">${plan.price}</span>
                      <span className="text-sm text-surface-500">/mo</span>
                    </div>
                    <p className="text-sm text-surface-500 dark:text-surface-400 mb-4">
                      {plan.scans_per_month === -1 ? 'Unlimited' : plan.scans_per_month} scans/month
                    </p>
                    <ul className="space-y-2 mb-5">
                      {Object.entries(plan.features).filter(([, v]) => v).map(([key]) => (
                        <li key={key} className="flex items-center gap-2 text-sm text-surface-600 dark:text-surface-300">
                          <Check className="h-4 w-4 text-emerald-500 flex-shrink-0" />
                          {key.replace(/_/g, ' ')}
                        </li>
                      ))}
                    </ul>
                    <button
                      onClick={() => handleUpgrade(plan)}
                      disabled={isCurrent}
                      className={`btn btn-md w-full ${isCurrent ? 'btn-secondary opacity-50 cursor-not-allowed' : 'btn-primary'}`}
                    >
                      {isCurrent ? 'Current' : plan.price === 0 ? 'Switch' : 'Upgrade'}
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {/* Payment History */}
        {paymentHistory.length > 0 && (
          <div className="card p-6">
            <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <Clock className="h-5 w-5" /> Payment History
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-surface-200 dark:border-surface-700">
                    {['Date', 'Transaction', 'Plan', 'Amount', 'Card', 'Status'].map(h => (
                      <th key={h} className="text-left py-3 px-4 text-surface-500 dark:text-surface-400 font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {paymentHistory.map((p) => (
                    <tr key={p.id} className="border-b border-surface-100 dark:border-surface-700/50 hover:bg-surface-50 dark:hover:bg-surface-700/30">
                      <td className="py-3 px-4">{new Date(p.created_at).toLocaleDateString()}</td>
                      <td className="py-3 px-4 font-mono text-xs">{p.transaction_id}</td>
                      <td className="py-3 px-4 capitalize">{p.plan_name}</td>
                      <td className="py-3 px-4">${p.amount.toFixed(2)}</td>
                      <td className="py-3 px-4">****{p.card_last_four}</td>
                      <td className="py-3 px-4">
                        <span className={`badge ${p.status === 'completed' ? 'badge-success' : 'badge-warning'}`}>{p.status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Payment Modal */}
      {showPaymentModal && selectedPlan && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-surface-800 rounded-2xl max-w-md w-full p-6 shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white">
                Upgrade to {selectedPlan.display_name}
              </h3>
              <button onClick={() => setShowPaymentModal(false)} className="btn btn-ghost btn-sm"><X className="h-5 w-5" /></button>
            </div>
            <div className="bg-primary-50 dark:bg-primary-900/20 rounded-xl p-4 mb-6 text-center">
              <p className="text-3xl font-display font-bold text-primary-600">${selectedPlan.price}<span className="text-sm font-normal text-surface-500">/mo</span></p>
            </div>
            <form onSubmit={handlePayment} className="space-y-4">
              <div>
                <label className="input-label">Cardholder Name</label>
                <input type="text" className="input-field" required value={paymentForm.cardholder_name}
                  onChange={(e) => setPaymentForm({ ...paymentForm, cardholder_name: e.target.value })} placeholder="John Doe" />
              </div>
              <div>
                <label className="input-label">Card Number</label>
                <div className="relative">
                  <CreditCard className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
                  <input type="text" className="input-field pl-12" required maxLength={19}
                    value={paymentForm.card_number}
                    onChange={(e) => setPaymentForm({ ...paymentForm, card_number: formatCardNumber(e.target.value) })}
                    placeholder="4242 4242 4242 4242" />
                </div>
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="input-label">Month</label>
                  <input type="text" className="input-field" required maxLength={2}
                    value={paymentForm.expiry_month}
                    onChange={(e) => setPaymentForm({ ...paymentForm, expiry_month: e.target.value.replace(/\D/g, '') })}
                    placeholder="MM" />
                </div>
                <div>
                  <label className="input-label">Year</label>
                  <input type="text" className="input-field" required maxLength={4}
                    value={paymentForm.expiry_year}
                    onChange={(e) => setPaymentForm({ ...paymentForm, expiry_year: e.target.value.replace(/\D/g, '') })}
                    placeholder="YYYY" />
                </div>
                <div>
                  <label className="input-label">CVV</label>
                  <input type="password" className="input-field" required maxLength={4}
                    value={paymentForm.cvv}
                    onChange={(e) => setPaymentForm({ ...paymentForm, cvv: e.target.value.replace(/\D/g, '') })}
                    placeholder="***" />
                </div>
              </div>
              <button type="submit" disabled={processing} className="btn btn-lg btn-primary w-full">
                {processing ? <><Loader className="h-5 w-5 animate-spin" /> Processing...</> : `Pay $${selectedPlan.price}`}
              </button>
            </form>
          </div>
        </div>
      )}
    </TenantLayout>
  )
}

export default TenantSubscription
