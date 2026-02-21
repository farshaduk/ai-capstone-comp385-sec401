import { useState, useEffect } from 'react'
import TenantLayout from '../../components/layouts/TenantLayout'
import { renterAPI } from '../../services/api'
import { FileText, Clock, CheckCircle, XCircle, Eye, MessageSquare, Loader, Send } from 'lucide-react'
import toast from 'react-hot-toast'

const statusConfig = {
  pending: { label: 'Pending', color: 'badge-warning', icon: Clock },
  approved: { label: 'Approved', color: 'badge-success', icon: CheckCircle },
  rejected: { label: 'Rejected', color: 'badge-danger', icon: XCircle },
  viewing_scheduled: { label: 'Viewing Scheduled', color: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 rounded-full px-3 py-1 text-xs font-medium', icon: Eye },
}

const Applications = () => {
  const [applications, setApplications] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeChat, setActiveChat] = useState(null)
  const [messages, setMessages] = useState([])
  const [newMsg, setNewMsg] = useState('')
  const [loadingMsgs, setLoadingMsgs] = useState(false)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const { data } = await renterAPI.getApplications()
        setApplications(data.applications || [])
      } catch (err) {
        console.error(err)
        toast.error('Failed to load applications')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const openChat = async (appId) => {
    setActiveChat(appId)
    setLoadingMsgs(true)
    try {
      const { data } = await renterAPI.getAppMessages(appId)
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
      await renterAPI.sendAppMessage(activeChat, newMsg.trim())
      setNewMsg('')
      // Refresh messages
      const { data } = await renterAPI.getAppMessages(activeChat)
      setMessages(data.messages || [])
      toast.success('Message sent')
    } catch {
      toast.error('Failed to send message')
    }
  }

  return (
    <TenantLayout title="My Applications" subtitle="Track the status of your rental applications">
      {loading ? (
        <div className="card p-16 text-center">
          <Loader className="h-8 w-8 animate-spin text-tenant-500 mx-auto mb-3" />
          <p className="text-surface-500">Loading applications...</p>
        </div>
      ) : applications.length === 0 ? (
        <div className="card p-16 text-center">
          <FileText className="h-12 w-12 text-surface-300 dark:text-surface-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-surface-900 dark:text-white mb-2">No applications yet</h3>
          <p className="text-sm text-surface-500 dark:text-surface-400">
            Browse listings and apply to properties you are interested in.
          </p>
        </div>
      ) : (
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Applications List */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-surface-500 uppercase tracking-wider">
              {applications.length} Application{applications.length !== 1 && 's'}
            </h3>
            {applications.map(app => {
              const cfg = statusConfig[app.status] || statusConfig.pending
              const Icon = cfg.icon
              return (
                <div key={app.id} className={`card p-5 cursor-pointer transition-all ${activeChat === app.id ? 'ring-2 ring-tenant-500' : 'hover:shadow-md'}`}
                  onClick={() => openChat(app.id)}>
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1 min-w-0">
                      <h4 className="font-semibold text-surface-900 dark:text-white text-sm truncate">{app.listing}</h4>
                      <p className="text-xs text-surface-500 mt-0.5">{app.address}</p>
                    </div>
                    <span className={cfg.color + ' flex items-center gap-1'}>
                      <Icon className="h-3 w-3" />{cfg.label}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-surface-500">
                    <span>${app.price?.toLocaleString()}/mo</span>
                    <span>Applied {app.applied_at ? new Date(app.applied_at).toLocaleDateString() : '—'}</span>
                  </div>
                  {app.last_message && (
                    <div className="mt-3 pt-3 border-t border-surface-100 dark:border-surface-700 flex items-center gap-2 text-xs text-surface-400">
                      <MessageSquare className="h-3 w-3" />
                      <span className="truncate">{app.last_message.from}: {app.last_message.text}</span>
                    </div>
                  )}
                  {app.unread > 0 && (
                    <div className="mt-2 flex justify-end">
                      <span className="bg-tenant-500 text-white text-xs rounded-full px-2 py-0.5">{app.unread} new</span>
                    </div>
                  )}
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
                  <p>Select an application to view messages</p>
                </div>
              </div>
            ) : (
              <>
                <div className="p-4 border-b border-surface-200 dark:border-surface-700">
                  <h4 className="font-semibold text-surface-900 dark:text-white text-sm">
                    {applications.find(a => a.id === activeChat)?.listing}
                  </h4>
                  <p className="text-xs text-surface-500">
                    Landlord: {applications.find(a => a.id === activeChat)?.landlord}
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
                        <div className={`max-w-[75%] rounded-2xl px-4 py-2 text-sm ${m.from_me ? 'bg-tenant-500 text-white' : 'bg-surface-100 dark:bg-surface-700 text-surface-900 dark:text-white'}`}>
                          <p>{m.text}</p>
                          <p className={`text-[10px] mt-1 ${m.from_me ? 'text-tenant-200' : 'text-surface-400'}`}>
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
    </TenantLayout>
  )
}

export default Applications
