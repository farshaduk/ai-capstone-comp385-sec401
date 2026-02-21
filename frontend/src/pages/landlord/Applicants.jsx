import { useState, useEffect } from 'react'
import LandlordLayout from '../../components/layouts/LandlordLayout'
import { landlordAPI } from '../../services/api'
import { Users, Clock, CheckCircle, XCircle, Eye, Loader, Filter, MessageSquare, Send } from 'lucide-react'
import toast from 'react-hot-toast'

const statusConfig = {
  pending: { label: 'Pending', color: 'badge-warning', icon: Clock },
  approved: { label: 'Approved', color: 'badge-success', icon: CheckCircle },
  rejected: { label: 'Rejected', color: 'badge-danger', icon: XCircle },
  viewing_scheduled: { label: 'Viewing', color: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 rounded-full px-3 py-1 text-xs font-medium', icon: Eye },
}

const Applicants = () => {
  const [applicants, setApplicants] = useState([])
  const [counts, setCounts] = useState({})
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')
  const [activeChat, setActiveChat] = useState(null)
  const [messages, setMessages] = useState([])
  const [newMsg, setNewMsg] = useState('')
  const [loadingMsgs, setLoadingMsgs] = useState(false)

  const fetchApplicants = async () => {
    setLoading(true)
    try {
      const { data } = await landlordAPI.getApplicants(filter)
      setApplicants(data.applicants || [])
      setCounts(data.counts || {})
    } catch (err) {
      console.error(err)
      toast.error('Failed to load applicants')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchApplicants() }, [filter])

  const updateStatus = async (id, newStatus) => {
    try {
      await landlordAPI.updateApplicationStatus(id, newStatus)
      if (newStatus === 'approved') {
        toast.success('Application approved — listing deactivated, lease draft created, other applicants rejected')
      } else {
        toast.success(`Application ${newStatus}`)
      }
      fetchApplicants()
    } catch {
      toast.error('Failed to update status')
    }
  }

  const openChat = async (appId) => {
    setActiveChat(appId)
    setLoadingMsgs(true)
    try {
      const { data } = await landlordAPI.getAppMessages(appId)
      setMessages(data.messages || [])
    } catch {
      toast.error('Failed to load messages')
    } finally {
      setLoadingMsgs(false)
    }
  }

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!newMsg.trim() || !activeChat) return
    try {
      await landlordAPI.sendAppMessage(activeChat, newMsg.trim())
      setNewMsg('')
      const { data } = await landlordAPI.getAppMessages(activeChat)
      setMessages(data.messages || [])
      toast.success('Message sent')
    } catch {
      toast.error('Failed to send message')
    }
  }

  const activeApp = applicants.find(a => a.id === activeChat)

  return (
    <LandlordLayout title="Applicants" subtitle="Review and manage rental applications">
      {/* Filter tabs */}
      <div className="flex items-center gap-2 mb-6 flex-wrap">
        {[
          { key: '', label: 'All' },
          { key: 'pending', label: `Pending (${counts.pending || 0})` },
          { key: 'approved', label: `Approved (${counts.approved || 0})` },
          { key: 'viewing_scheduled', label: `Viewing (${counts.viewing_scheduled || 0})` },
          { key: 'rejected', label: `Rejected (${counts.rejected || 0})` },
        ].map(tab => (
          <button key={tab.key} onClick={() => setFilter(tab.key)}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${filter === tab.key
              ? 'bg-landlord-500 text-white'
              : 'bg-surface-100 dark:bg-surface-800 text-surface-600 dark:text-surface-400 hover:bg-surface-200 dark:hover:bg-surface-700'}`}>
            {tab.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="card p-16 text-center"><Loader className="h-8 w-8 animate-spin text-landlord-500 mx-auto mb-3" /><p className="text-surface-500">Loading...</p></div>
      ) : applicants.length === 0 ? (
        <div className="card p-16 text-center">
          <Users className="h-12 w-12 text-surface-300 dark:text-surface-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-surface-900 dark:text-white mb-2">No applicants</h3>
          <p className="text-sm text-surface-500">
            {filter ? 'No applicants match the selected filter.' : 'No one has applied to your listings yet.'}
          </p>
        </div>
      ) : (
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Applicants list */}
          <div className="space-y-4">
            {applicants.map(app => {
              const cfg = statusConfig[app.status] || statusConfig.pending
              const Icon = cfg.icon
              return (
                <div key={app.id} className={`card p-5 cursor-pointer transition-all ${activeChat === app.id ? 'ring-2 ring-landlord-500' : 'hover:shadow-md'}`}
                  onClick={() => openChat(app.id)}>
                  <div className="flex flex-col md:flex-row md:items-center gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-semibold text-surface-900 dark:text-white">{app.applicant_name}</h4>
                        <span className={cfg.color + ' flex items-center gap-1'}>
                          <Icon className="h-3 w-3" />{cfg.label}
                        </span>
                      </div>
                      <p className="text-sm text-surface-500">{app.applicant_email}</p>
                      <p className="text-xs text-surface-400 mt-1">
                        Applied for <span className="font-medium text-surface-700 dark:text-surface-300">{app.listing}</span>
                        {app.applied_at && ` • ${new Date(app.applied_at).toLocaleDateString()}`}
                      </p>
                      {app.message && (
                        <p className="text-xs text-surface-500 mt-2 bg-surface-50 dark:bg-surface-800 p-2 rounded-lg italic">
                          "{app.message}"
                        </p>
                      )}
                    </div>
                    {(app.status === 'pending' || app.status === 'viewing_scheduled') && (
                      <div className="flex items-center gap-2 flex-shrink-0" onClick={e => e.stopPropagation()}>
                        {app.status === 'pending' && (
                          <button onClick={() => updateStatus(app.id, 'viewing_scheduled')}
                            className="btn btn-secondary btn-sm text-xs"><Eye className="h-3.5 w-3.5" /> Viewing</button>
                        )}
                        <button onClick={() => updateStatus(app.id, 'approved')}
                          className="btn btn-primary btn-sm text-xs"><CheckCircle className="h-3.5 w-3.5" /> Approve</button>
                        <button onClick={() => updateStatus(app.id, 'rejected')}
                          className="btn btn-secondary btn-sm text-xs text-red-500"><XCircle className="h-3.5 w-3.5" /> Reject</button>
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Chat panel */}
          <div className="card flex flex-col h-[500px]">
            {!activeChat ? (
              <div className="flex-1 flex items-center justify-center text-surface-400 text-sm">
                <div className="text-center">
                  <MessageSquare className="h-10 w-10 mx-auto mb-3 text-surface-300" />
                  <p>Select an applicant to view messages</p>
                </div>
              </div>
            ) : (
              <>
                <div className="p-4 border-b border-surface-200 dark:border-surface-700">
                  <h4 className="font-semibold text-surface-900 dark:text-white text-sm">
                    {activeApp?.applicant_name}
                  </h4>
                  <p className="text-xs text-surface-500">
                    {activeApp?.listing} • {activeApp?.applicant_email}
                  </p>
                </div>
                <div className="flex-1 overflow-y-auto p-4 space-y-3">
                  {loadingMsgs ? (
                    <div className="text-center text-surface-400"><Loader className="h-5 w-5 animate-spin mx-auto" /></div>
                  ) : messages.length === 0 ? (
                    <p className="text-center text-surface-400 text-sm">No messages yet. Start a conversation!</p>
                  ) : (
                    messages.map(m => (
                      <div key={m.id} className={`flex ${m.from_me ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[75%] rounded-2xl px-4 py-2 text-sm ${m.from_me ? 'bg-landlord-500 text-white' : 'bg-surface-100 dark:bg-surface-700 text-surface-900 dark:text-white'}`}>
                          <p>{m.text}</p>
                          <p className={`text-[10px] mt-1 ${m.from_me ? 'text-landlord-200' : 'text-surface-400'}`}>
                            {m.at ? new Date(m.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
                          </p>
                        </div>
                      </div>
                    ))
                  )}
                </div>
                <form onSubmit={sendMessage} className="p-4 border-t border-surface-200 dark:border-surface-700 flex gap-2">
                  <input type="text" value={newMsg} onChange={e => setNewMsg(e.target.value)} placeholder="Type a message..."
                    className="input-field flex-1" />
                  <button type="submit" className="btn btn-primary btn-md"><Send className="h-4 w-4" /></button>
                </form>
              </>
            )}
          </div>
        </div>
      )}
    </LandlordLayout>
  )
}

export default Applicants
