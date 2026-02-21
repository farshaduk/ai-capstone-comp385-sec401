import { useState, useEffect, Fragment } from 'react'
import AdminLayout from '../../components/layouts/AdminLayout'
import { adminAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  ScrollText, Loader, Search, Clock,
  ChevronLeft, ChevronRight
} from 'lucide-react'

const PAGE_SIZE = 25

const AdminAuditLogs = () => {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [actionFilter, setActionFilter] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(0)
  const [totalItems, setTotalItems] = useState(0)
  const [expandedRow, setExpandedRow] = useState(null)

  useEffect(() => { fetchLogs() }, [currentPage, actionFilter])

  const fetchLogs = async () => {
    setLoading(true)
    try {
      const res = await adminAPI.getAuditLogs({
        skip: (currentPage - 1) * PAGE_SIZE,
        limit: PAGE_SIZE,
        action: actionFilter || undefined
      })
      setLogs(res.data.items || [])
      setTotalPages(res.data.pages || 0)
      setTotalItems(res.data.total || 0)
    } catch { toast.error('Failed to fetch audit logs') }
    finally { setLoading(false) }
  }

  // Client-side search within current page
  const filtered = logs.filter(l =>
    (l.action || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
    (l.user_email || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
    (l.entity_type || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
    (l.ip_address || '').toLowerCase().includes(searchQuery.toLowerCase())
  )

  const getActionBadge = (action) => {
    if (!action) return 'badge-info'
    const a = action.toLowerCase()
    if (a.includes('login') || a.includes('register')) return 'badge-success'
    if (a.includes('delete') || a.includes('deactivate') || a.includes('error') || a.includes('fail')) return 'badge-danger'
    if (a.includes('update') || a.includes('edit') || a.includes('change')) return 'badge-warning'
    if (a.includes('create') || a.includes('add') || a.includes('upload')) return 'badge-info'
    if (a.includes('analyze') || a.includes('scan') || a.includes('risk')) return 'badge-primary'
    return 'badge-info'
  }

  const goToPage = (page) => {
    if (page < 1 || page > totalPages) return
    setCurrentPage(page)
    setExpandedRow(null)
  }

  const getPageNumbers = () => {
    const pages = []
    const maxVisible = 5
    let start = Math.max(1, currentPage - Math.floor(maxVisible / 2))
    let end = Math.min(totalPages, start + maxVisible - 1)
    if (end - start + 1 < maxVisible) {
      start = Math.max(1, end - maxVisible + 1)
    }
    for (let i = start; i <= end; i++) pages.push(i)
    return pages
  }

  return (
    <AdminLayout title="Audit Logs" subtitle="Security and activity tracking">
      <div className="space-y-6 animate-fade-in">
        {/* Toolbar */}
        <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center justify-between">
          <div className="flex flex-col sm:flex-row gap-3 flex-1">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4 text-surface-400" />
              <input type="text" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
                className="input-field pl-11" placeholder="Search logs on this page..." />
            </div>
            <select value={actionFilter} onChange={(e) => { setActionFilter(e.target.value); setCurrentPage(1) }}
              className="input-field w-auto">
              <option value="">All Actions</option>
              <option value="login">Login</option>
              <option value="register">Register</option>
              <option value="analyze">Analyze</option>
              <option value="create">Create</option>
              <option value="update">Update</option>
              <option value="delete">Delete</option>
              <option value="error">Error</option>
            </select>
          </div>
          <div className="text-xs text-surface-500 dark:text-surface-400 flex items-center gap-1">
            <ScrollText className="h-3.5 w-3.5" />
            {totalItems.toLocaleString()} total logs
          </div>
        </div>

        {/* Table */}
        {loading ? (
          <div className="flex justify-center py-16"><Loader className="h-8 w-8 animate-spin text-primary-600" /></div>
        ) : filtered.length > 0 ? (
          <>
            <div className="card p-0 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-surface-200 dark:border-surface-700 bg-surface-50 dark:bg-surface-800">
                      {['Action', 'User', 'Entity', 'IP Address', 'Timestamp', ''].map(h => (
                        <th key={h} className="text-left py-3 px-4 text-xs font-medium text-surface-500 dark:text-surface-400 uppercase tracking-wider">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((log) => (
                      <Fragment key={log.id}>
                        <tr
                          onClick={() => setExpandedRow(expandedRow === log.id ? null : log.id)}
                          className="border-b border-surface-100 dark:border-surface-700/50 hover:bg-surface-50 dark:hover:bg-surface-700/30 transition-colors cursor-pointer">
                          <td className="py-3 px-4">
                            <span className={`badge ${getActionBadge(log.action)}`}>
                              {log.action}
                            </span>
                          </td>
                          <td className="py-3 px-4">
                            {log.user_email ? (
                              <div className="flex items-center gap-2">
                                <div className="w-6 h-6 bg-primary-100 dark:bg-primary-900/30 rounded-md flex items-center justify-center text-primary-600 font-bold text-xs">
                                  {log.user_email[0].toUpperCase()}
                                </div>
                                <span className="text-sm text-surface-700 dark:text-surface-300">{log.user_email}</span>
                              </div>
                            ) : (
                              <span className="text-xs text-surface-400">System</span>
                            )}
                          </td>
                          <td className="py-3 px-4">
                            {log.entity_type ? (
                              <span className="text-sm text-surface-700 dark:text-surface-300">
                                {log.entity_type}{log.entity_id ? ` #${log.entity_id}` : ''}
                              </span>
                            ) : (
                              <span className="text-xs text-surface-400">—</span>
                            )}
                          </td>
                          <td className="py-3 px-4">
                            <span className="text-xs text-surface-500 dark:text-surface-400 font-mono">
                              {log.ip_address || '—'}
                            </span>
                          </td>
                          <td className="py-3 px-4">
                            <span className="text-xs text-surface-500 dark:text-surface-400 flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {log.created_at ? new Date(log.created_at).toLocaleString() : '—'}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-right">
                            <span className="text-xs text-surface-400">{expandedRow === log.id ? '▲' : '▼'}</span>
                          </td>
                        </tr>
                        {expandedRow === log.id && (
                          <tr className="bg-surface-50 dark:bg-surface-800/50">
                            <td colSpan={6} className="py-3 px-6">
                              <div className="text-xs space-y-1">
                                <p className="text-surface-500 dark:text-surface-400 font-medium mb-2">Details</p>
                                {log.details && Object.keys(log.details).length > 0 ? (
                                  <pre className="bg-surface-100 dark:bg-surface-900 rounded-lg p-3 text-surface-600 dark:text-surface-300 overflow-x-auto whitespace-pre-wrap break-all font-mono text-xs">
                                    {JSON.stringify(log.details, null, 2)}
                                  </pre>
                                ) : (
                                  <p className="text-surface-400 italic">No additional details</p>
                                )}
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between">
                <p className="text-xs text-surface-500 dark:text-surface-400">
                  Showing {((currentPage - 1) * PAGE_SIZE) + 1}–{Math.min(currentPage * PAGE_SIZE, totalItems)} of {totalItems.toLocaleString()}
                </p>
                <div className="flex items-center gap-1">
                  <button onClick={() => goToPage(currentPage - 1)} disabled={currentPage === 1}
                    className="btn btn-sm btn-ghost disabled:opacity-30">
                    <ChevronLeft className="h-4 w-4" />
                  </button>
                  {getPageNumbers()[0] > 1 && (
                    <>
                      <button onClick={() => goToPage(1)} className="btn btn-sm btn-ghost text-xs">1</button>
                      {getPageNumbers()[0] > 2 && <span className="text-surface-400 text-xs px-1">…</span>}
                    </>
                  )}
                  {getPageNumbers().map(p => (
                    <button key={p} onClick={() => goToPage(p)}
                      className={`btn btn-sm text-xs ${p === currentPage ? 'btn-primary' : 'btn-ghost'}`}>
                      {p}
                    </button>
                  ))}
                  {getPageNumbers()[getPageNumbers().length - 1] < totalPages && (
                    <>
                      {getPageNumbers()[getPageNumbers().length - 1] < totalPages - 1 && <span className="text-surface-400 text-xs px-1">…</span>}
                      <button onClick={() => goToPage(totalPages)} className="btn btn-sm btn-ghost text-xs">{totalPages}</button>
                    </>
                  )}
                  <button onClick={() => goToPage(currentPage + 1)} disabled={currentPage === totalPages}
                    className="btn btn-sm btn-ghost disabled:opacity-30">
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="card text-center py-16">
            <ScrollText className="h-16 w-16 text-surface-300 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-surface-600 mb-2">{searchQuery || actionFilter ? 'No matching logs' : 'No audit logs'}</h3>
            <p className="text-sm text-surface-400">
              {searchQuery || actionFilter ? 'Try adjusting your search or filter' : 'Activity logs will appear here as users interact with the system'}
            </p>
          </div>
        )}
      </div>
    </AdminLayout>
  )
}

export default AdminAuditLogs
