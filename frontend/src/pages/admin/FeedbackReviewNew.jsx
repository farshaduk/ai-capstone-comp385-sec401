import { useState, useEffect } from 'react'
import AdminLayout from '../../components/layouts/AdminLayout'
import { adminAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  MessageSquare, CheckCircle, XCircle, Clock, Filter,
  RefreshCw, Loader, ChevronDown, ChevronUp, Shield,
  AlertTriangle, User, FileText
} from 'lucide-react'

const statusColors = {
  pending:  { bg: 'bg-amber-100 dark:bg-amber-900/30', text: 'text-amber-700 dark:text-amber-300', icon: Clock },
  approved: { bg: 'bg-emerald-100 dark:bg-emerald-900/30', text: 'text-emerald-700 dark:text-emerald-300', icon: CheckCircle },
  rejected: { bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-300', icon: XCircle },
}

const typeColors = {
  fraud: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300',
  safe: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300',
  unsure: 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300',
}

const FeedbackReviewNew = () => {
  const [feedback, setFeedback] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [statusFilter, setStatusFilter] = useState('pending')
  const [typeFilter, setTypeFilter] = useState('')
  const [expandedId, setExpandedId] = useState(null)
  const [reviewingId, setReviewingId] = useState(null)

  useEffect(() => {
    fetchAll()
  }, [statusFilter, typeFilter])

  const fetchAll = async () => {
    setLoading(true)
    try {
      const params = {}
      if (statusFilter) params.status = statusFilter
      if (typeFilter) params.feedback_type = typeFilter

      const [fb, st] = await Promise.all([
        adminAPI.getFeedback(params),
        adminAPI.getFeedbackStats()
      ])
      setFeedback(fb.data)
      setStats(st.data)
    } catch (e) {
      toast.error('Failed to load feedback')
    } finally {
      setLoading(false)
    }
  }

  const handleReview = async (id, status) => {
    setReviewingId(id)
    try {
      await adminAPI.reviewFeedback(id, { status })
      toast.success(`Feedback #${id} ${status}`)
      await fetchAll()
    } catch (e) {
      toast.error(`Review failed: ${e.response?.data?.detail || e.message}`)
    } finally {
      setReviewingId(null)
    }
  }

  return (
    <AdminLayout title="Feedback Review" subtitle="Review and approve user feedback before auto-learning">
      <div className="space-y-6 animate-fade-in">

        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: 'Pending', value: stats.pending_review ?? 0, icon: Clock, color: 'text-amber-600', bg: 'bg-amber-50 dark:bg-amber-900/20' },
              { label: 'Approved', value: stats.approved ?? 0, icon: CheckCircle, color: 'text-emerald-600', bg: 'bg-emerald-50 dark:bg-emerald-900/20' },
              { label: 'Rejected', value: stats.rejected ?? 0, icon: XCircle, color: 'text-red-600', bg: 'bg-red-50 dark:bg-red-900/20' },
              { label: 'Total', value: stats.total_feedback ?? 0, icon: MessageSquare, color: 'text-blue-600', bg: 'bg-blue-50 dark:bg-blue-900/20' },
            ].map((s, i) => (
              <div key={i} className={`card p-4 ${s.bg}`}>
                <s.icon className={`h-5 w-5 ${s.color} mb-1`} />
                <p className={`text-2xl font-display font-bold ${s.color}`}>{s.value}</p>
                <p className="text-xs text-surface-500 dark:text-surface-400">{s.label}</p>
              </div>
            ))}
          </div>
        )}

        {/* Info Banner */}
        <div className="p-4 rounded-xl bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800">
          <div className="flex items-start gap-3">
            <Shield className="h-5 w-5 text-primary-600 dark:text-primary-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-semibold text-primary-800 dark:text-primary-200">Approval Required</p>
              <p className="text-xs text-primary-600 dark:text-primary-400 mt-1">
                Only <strong>approved</strong> feedback is used by the AI auto-learning engine. 
                Pending and rejected feedback is excluded from model retraining to prevent data poisoning.
              </p>
            </div>
          </div>
        </div>

        {/* Filters + Refresh */}
        <div className="card p-4">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-surface-400" />
              <span className="text-sm font-medium text-surface-600 dark:text-surface-300">Filters:</span>
            </div>

            {/* Status filter */}
            <select
              value={statusFilter}
              onChange={e => setStatusFilter(e.target.value)}
              className="input-field w-auto text-sm py-1.5 px-3"
            >
              <option value="">All Statuses</option>
              <option value="pending">Pending</option>
              <option value="approved">Approved</option>
              <option value="rejected">Rejected</option>
            </select>

            {/* Type filter */}
            <select
              value={typeFilter}
              onChange={e => setTypeFilter(e.target.value)}
              className="input-field w-auto text-sm py-1.5 px-3"
            >
              <option value="">All Types</option>
              <option value="fraud">Fraud</option>
              <option value="safe">Safe</option>
              <option value="unsure">Unsure</option>
            </select>

            <div className="flex-1" />

            <button onClick={fetchAll} className="btn btn-sm btn-secondary">
              <RefreshCw className="h-3.5 w-3.5" /> Refresh
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="card overflow-hidden">
          {loading ? (
            <div className="flex justify-center py-16">
              <Loader className="h-8 w-8 animate-spin text-primary-600" />
            </div>
          ) : feedback.length === 0 ? (
            <div className="text-center py-16">
              <MessageSquare className="h-12 w-12 text-surface-300 mx-auto mb-3" />
              <p className="text-surface-500 dark:text-surface-400 text-sm">No feedback found for this filter</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-surface-50 dark:bg-surface-800/50 border-b border-surface-200 dark:border-surface-700">
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">ID</th>
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">Analysis</th>
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">User</th>
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">Type</th>
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">Status</th>
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">Submitted</th>
                    <th className="text-left text-xs font-semibold text-surface-500 uppercase tracking-wider px-4 py-3">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-surface-100 dark:divide-surface-800">
                  {feedback.map(f => {
                    const sc = statusColors[f.status] || statusColors.pending
                    const StatusIcon = sc.icon
                    const isExpanded = expandedId === f.id
                    const isReviewing = reviewingId === f.id

                    return (
                      <>
                        <tr
                          key={f.id}
                          className="hover:bg-surface-50/80 dark:hover:bg-surface-800/50 transition-colors cursor-pointer"
                          onClick={() => setExpandedId(isExpanded ? null : f.id)}
                        >
                          <td className="px-4 py-3 text-sm font-mono text-surface-600 dark:text-surface-300">#{f.id}</td>
                          <td className="px-4 py-3 text-sm text-surface-600 dark:text-surface-300">#{f.analysis_id}</td>
                          <td className="px-4 py-3">
                            <span className="inline-flex items-center gap-1.5 text-sm text-surface-600 dark:text-surface-300">
                              <User className="h-3.5 w-3.5" /> {f.user_id}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${typeColors[f.feedback_type] || 'bg-surface-100 text-surface-600'}`}>
                              {f.feedback_type}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium ${sc.bg} ${sc.text}`}>
                              <StatusIcon className="h-3 w-3" /> {f.status}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-xs text-surface-500">
                            {new Date(f.created_at).toLocaleString()}
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2" onClick={e => e.stopPropagation()}>
                              {f.status === 'pending' && (
                                <>
                                  <button
                                    onClick={() => handleReview(f.id, 'approved')}
                                    disabled={isReviewing}
                                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium
                                      bg-emerald-100 text-emerald-700 hover:bg-emerald-200
                                      dark:bg-emerald-900/30 dark:text-emerald-300 dark:hover:bg-emerald-900/50
                                      disabled:opacity-50 transition-colors"
                                  >
                                    <CheckCircle className="h-3 w-3" /> Approve
                                  </button>
                                  <button
                                    onClick={() => handleReview(f.id, 'rejected')}
                                    disabled={isReviewing}
                                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium
                                      bg-red-100 text-red-700 hover:bg-red-200
                                      dark:bg-red-900/30 dark:text-red-300 dark:hover:bg-red-900/50
                                      disabled:opacity-50 transition-colors"
                                  >
                                    <XCircle className="h-3 w-3" /> Reject
                                  </button>
                                </>
                              )}
                              {f.status !== 'pending' && (
                                <span className="text-xs text-surface-400 italic">Reviewed</span>
                              )}
                              {isExpanded
                                ? <ChevronUp className="h-4 w-4 text-surface-400" />
                                : <ChevronDown className="h-4 w-4 text-surface-400" />
                              }
                            </div>
                          </td>
                        </tr>

                        {/* Expanded detail row */}
                        {isExpanded && (
                          <tr key={`${f.id}-detail`} className="bg-surface-50/50 dark:bg-surface-800/30">
                            <td colSpan={7} className="px-6 py-4">
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                                <div>
                                  <p className="text-xs font-semibold text-surface-500 uppercase mb-1">Comments</p>
                                  <p className="text-surface-700 dark:text-surface-300">
                                    {f.comments || <span className="italic text-surface-400">No comments provided</span>}
                                  </p>
                                </div>
                                <div className="space-y-2">
                                  {f.reviewed_by && (
                                    <div>
                                      <p className="text-xs font-semibold text-surface-500 uppercase mb-1">Reviewed By</p>
                                      <p className="text-surface-700 dark:text-surface-300 flex items-center gap-1.5">
                                        <Shield className="h-3.5 w-3.5 text-primary-500" /> Admin #{f.reviewed_by}
                                      </p>
                                    </div>
                                  )}
                                  {f.reviewed_at && (
                                    <div>
                                      <p className="text-xs font-semibold text-surface-500 uppercase mb-1">Reviewed At</p>
                                      <p className="text-surface-700 dark:text-surface-300">
                                        {new Date(f.reviewed_at).toLocaleString()}
                                      </p>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Footer info */}
        <div className="text-xs text-surface-400 text-center">
          Showing {feedback.length} feedback entries • Auto-learning only uses approved feedback
        </div>
      </div>
    </AdminLayout>
  )
}

export default FeedbackReviewNew
