import { useState, useEffect } from 'react'
import TenantLayout from '../../components/layouts/TenantLayout'
import { renterAPI } from '../../services/api'
import { CreditCard, Download, CheckCircle, Clock, DollarSign, ArrowUpRight, ArrowDownLeft, Calendar, Filter } from 'lucide-react'
import toast from 'react-hot-toast'

const Payments = () => {
  const [payments, setPayments] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    loadPayments()
  }, [])

  const loadPayments = async () => {
    try {
      const res = await renterAPI.getPaymentHistory()
      const raw = res.data || []
      // Map backend fields to UI fields
      setPayments(raw.map(p => ({
        id: p.id,
        type: 'subscription',
        description: p.plan_name || 'Subscription Payment',
        amount: p.amount,
        status: p.status,
        date: p.created_at ? p.created_at.split('T')[0] : '',
        method: p.card_last_four ? `Card ••••${p.card_last_four}` : 'Card',
        transaction_id: p.transaction_id,
      })))
    } catch {
      setPayments([])
    } finally {
      setLoading(false)
    }
  }

  const totalSpent = payments.filter(p => p.amount > 0).reduce((a, b) => a + b.amount, 0)
  const currentMonth = new Date().toISOString().slice(0, 7) // e.g. "2026-02"
  const thisMonth = payments.filter(p => p.date?.startsWith(currentMonth) && p.amount > 0).reduce((a, b) => a + b.amount, 0)

  const filtered = payments.filter(p => {
    if (filter === 'all') return true
    return p.type === filter
  })

  return (
    <TenantLayout title="Payments" subtitle="View your billing history and manage payment methods">
      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        <div className="card p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-surface-500 dark:text-surface-400">This Month</p>
              <p className="text-2xl font-bold text-surface-900 dark:text-white">${thisMonth.toFixed(2)}</p>
            </div>
            <div className="p-3 bg-primary-100 dark:bg-primary-950/50 rounded-xl">
              <Calendar className="h-5 w-5 text-primary-600 dark:text-primary-400" />
            </div>
          </div>
        </div>
        <div className="card p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-surface-500 dark:text-surface-400">Total Spent</p>
              <p className="text-2xl font-bold text-surface-900 dark:text-white">${totalSpent.toFixed(2)}</p>
            </div>
            <div className="p-3 bg-tenant-100 dark:bg-tenant-950/50 rounded-xl">
              <DollarSign className="h-5 w-5 text-tenant-600 dark:text-tenant-400" />
            </div>
          </div>
        </div>
        <div className="card p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-surface-500 dark:text-surface-400">Payment Method</p>
              <p className="text-lg font-bold text-surface-900 dark:text-white">{payments.length > 0 ? payments[0].method : 'No card on file'}</p>
            </div>
            <div className="p-3 bg-surface-100 dark:bg-surface-800 rounded-xl">
              <CreditCard className="h-5 w-5 text-surface-600 dark:text-surface-400" />
            </div>
          </div>
        </div>
      </div>

      {/* Filter tabs */}
      <div className="flex items-center gap-2 mb-6">
        {['all', 'subscription', 'refund'].map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
              filter === f
                ? 'bg-primary-600 text-white'
                : 'bg-surface-100 dark:bg-surface-800 text-surface-600 dark:text-surface-400 hover:bg-surface-200 dark:hover:bg-surface-700'
            }`}
          >
            {f === 'all' ? 'All' : f.charAt(0).toUpperCase() + f.slice(1) + 's'}
          </button>
        ))}
      </div>

      {/* Payments Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-surface-200 dark:border-surface-700">
                <th className="text-left text-xs font-semibold text-surface-500 dark:text-surface-400 uppercase tracking-wider px-5 py-3">Transaction</th>
                <th className="text-left text-xs font-semibold text-surface-500 dark:text-surface-400 uppercase tracking-wider px-5 py-3">Date</th>
                <th className="text-left text-xs font-semibold text-surface-500 dark:text-surface-400 uppercase tracking-wider px-5 py-3">Method</th>
                <th className="text-left text-xs font-semibold text-surface-500 dark:text-surface-400 uppercase tracking-wider px-5 py-3">Status</th>
                <th className="text-right text-xs font-semibold text-surface-500 dark:text-surface-400 uppercase tracking-wider px-5 py-3">Amount</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-100 dark:divide-surface-800">
              {filtered.map(payment => (
                <tr key={payment.id} className="hover:bg-surface-50 dark:hover:bg-surface-800/50 transition-colors">
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${payment.amount > 0 ? 'bg-primary-100 dark:bg-primary-950/50' : 'bg-green-100 dark:bg-green-950/50'}`}>
                        {payment.amount > 0 ? (
                          <ArrowUpRight className="h-4 w-4 text-primary-600 dark:text-primary-400" />
                        ) : (
                          <ArrowDownLeft className="h-4 w-4 text-green-600 dark:text-green-400" />
                        )}
                      </div>
                      <span className="text-sm font-medium text-surface-900 dark:text-white">{payment.description}</span>
                    </div>
                  </td>
                  <td className="px-5 py-4 text-sm text-surface-500 dark:text-surface-400">{payment.date}</td>
                  <td className="px-5 py-4 text-sm text-surface-500 dark:text-surface-400">{payment.method}</td>
                  <td className="px-5 py-4">
                    <span className="badge-success">
                      <CheckCircle className="h-3 w-3" />
                      {payment.status}
                    </span>
                  </td>
                  <td className="px-5 py-4 text-right">
                    <span className={`text-sm font-semibold ${payment.amount > 0 ? 'text-surface-900 dark:text-white' : 'text-green-600 dark:text-green-400'}`}>
                      {payment.amount > 0 ? '' : '+'}${Math.abs(payment.amount).toFixed(2)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {filtered.length === 0 && (
          <div className="p-12 text-center">
            <CreditCard className="h-10 w-10 text-surface-300 dark:text-surface-600 mx-auto mb-3" />
            <p className="text-sm text-surface-500">No payments found.</p>
          </div>
        )}
      </div>
    </TenantLayout>
  )
}

export default Payments
