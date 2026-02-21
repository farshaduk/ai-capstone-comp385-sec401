import { useState } from 'react'
import TenantLayout from '../../components/layouts/TenantLayout'
import { renterAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  Search, Link as LinkIcon, Upload, AlertTriangle, CheckCircle,
  Shield, Loader, FileText, MessageSquare, Brain,
  Lightbulb, Send, Plus, Trash2
} from 'lucide-react'

const TenantAnalyze = () => {
  const [activeTab, setActiveTab] = useState('text')
  // Text analysis
  const [listingText, setListingText] = useState('')
  const [url, setUrl] = useState('')
  const [price, setPrice] = useState('')
  const [location, setLocation] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  // Message analysis
  const [messageText, setMessageText] = useState('')
  const [messageLoading, setMessageLoading] = useState(false)
  const [messageResult, setMessageResult] = useState(null)

  // Conversation analysis
  const [conversationMessages, setConversationMessages] = useState([
    { sender: 'landlord', text: '' },
  ])
  const [conversationLoading, setConversationLoading] = useState(false)
  const [conversationResult, setConversationResult] = useState(null)

  // XAI explanation
  const [xaiText, setXaiText] = useState('')
  const [xaiLoading, setXaiLoading] = useState(false)
  const [xaiResult, setXaiResult] = useState(null)

  // --- Handlers ---

  const handleAnalyze = async (e) => {
    e.preventDefault()
    setLoading(true)
    setResult(null)
    try {
      let response
      if (activeTab === 'url') {
        response = await renterAPI.analyzeUrl({
          url,
          listing_price: price ? parseFloat(price) : null,
          location: location || null,
        })
      } else {
        response = await renterAPI.analyzeListing({
          listing_text: listingText,
          listing_price: price ? parseFloat(price) : null,
          location: location || null,
        })
      }
      setResult(response.data)
      toast.success('Analysis complete!')
    } catch (error) {
      const detail = error.response?.data?.detail
      const msg = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(d => d.msg).join(', ') : 'Analysis failed'
      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleMessageAnalysis = async (e) => {
    e.preventDefault()
    if (!messageText.trim()) return toast.error('Enter a message to analyze')
    setMessageLoading(true)
    setMessageResult(null)
    try {
      const formData = new FormData()
      formData.append('message', messageText)
      const response = await renterAPI.analyzeMessage(formData)
      setMessageResult(response.data)
      toast.success('Message analysis complete!')
    } catch (error) {
      const detail = error.response?.data?.detail
      const msg = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(d => d.msg).join(', ') : 'Message analysis failed'
      toast.error(msg)
    } finally {
      setMessageLoading(false)
    }
  }

  const handleConversationAnalysis = async (e) => {
    e.preventDefault()
    const validMessages = conversationMessages.filter(m => m.text.trim())
    if (validMessages.length < 2) return toast.error('Add at least 2 messages to analyze a conversation')
    setConversationLoading(true)
    setConversationResult(null)
    try {
      const payload = validMessages.map(m => ({ content: m.text, sender: m.sender }))
      const response = await renterAPI.analyzeConversation({ messages: payload })
      setConversationResult(response.data)
      toast.success('Conversation analysis complete!')
    } catch (error) {
      const detail = error.response?.data?.detail
      const msg = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(d => d.msg).join(', ') : 'Conversation analysis failed'
      toast.error(msg)
    } finally {
      setConversationLoading(false)
    }
  }

  const handleXaiExplain = async (e) => {
    e.preventDefault()
    if (!xaiText.trim()) return toast.error('Enter text to explain')
    setXaiLoading(true)
    setXaiResult(null)
    try {
      const formData = new FormData()
      formData.append('text', xaiText)
      const response = await renterAPI.explainText(formData)
      setXaiResult(response.data)
      toast.success('Explanation generated!')
    } catch (error) {
      const detail = error.response?.data?.detail
      const msg = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(d => d.msg).join(', ') : 'Explanation failed'
      toast.error(msg)
    } finally {
      setXaiLoading(false)
    }
  }

  const addConversationMessage = () => {
    const lastSender = conversationMessages[conversationMessages.length - 1]?.sender
    setConversationMessages([
      ...conversationMessages,
      { sender: lastSender === 'landlord' ? 'tenant' : 'landlord', text: '' },
    ])
  }

  const removeConversationMessage = (index) => {
    if (conversationMessages.length <= 1) return
    setConversationMessages(conversationMessages.filter((_, i) => i !== index))
  }

  const updateConversationMessage = (index, field, value) => {
    const updated = [...conversationMessages]
    updated[index] = { ...updated[index], [field]: value }
    setConversationMessages(updated)
  }

  const getRiskBadge = (level) => {
    const config = {
      safe: { cls: 'badge-success', label: 'Safe' },
      very_low: { cls: 'badge-success', label: 'Very Low Risk' },
      low: { cls: 'badge-success', label: 'Low Risk' },
      medium: { cls: 'badge-warning', label: 'Medium Risk' },
      high: { cls: 'badge-danger', label: 'High Risk' },
      very_high: { cls: 'badge-danger', label: 'Very High Risk' },
      critical: { cls: 'badge-danger', label: 'Critical Risk' },
    }
    return config[level] || { cls: 'badge-info', label: level || 'Unknown' }
  }

  const tabs = [
    { id: 'text', icon: FileText, label: 'Text' },
    { id: 'url', icon: LinkIcon, label: 'URL' },
    { id: 'message', icon: MessageSquare, label: 'Message' },
    { id: 'conversation', icon: Send, label: 'Conversation' },
    { id: 'xai', icon: Lightbulb, label: 'XAI' },
  ]

  return (
    <TenantLayout title="Analyze Listing" subtitle="AI-powered fraud detection for rental listings">
      <div className="max-w-4xl mx-auto">
        {/* Tab selector */}
        <div className="flex items-center gap-2 mb-6 flex-wrap">
          {tabs.map(t => (
            <button key={t.id} onClick={() => setActiveTab(t.id)}
              className={`btn btn-md ${activeTab === t.id ? 'btn-primary' : 'btn-secondary'}`}>
              <t.icon className="h-4 w-4" /> {t.label}
            </button>
          ))}
        </div>

        {/* ====== TEXT / URL ANALYSIS ====== */}
        {(activeTab === 'text' || activeTab === 'url') && (
          <>
            <form onSubmit={handleAnalyze} className="card p-6 mb-6">
              {activeTab === 'text' ? (
                <div className="mb-4">
                  <label className="input-label">Listing Text</label>
                  <textarea value={listingText} onChange={e => setListingText(e.target.value)}
                    className="input-field min-h-[200px] resize-y"
                    placeholder="Paste the rental listing text here for analysis..." required minLength={10} />
                </div>
              ) : (
                <div className="mb-4">
                  <label className="input-label">Listing URL</label>
                  <div className="relative">
                    <LinkIcon className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
                    <input type="url" value={url} onChange={e => setUrl(e.target.value)}
                      className="input-field pl-12" placeholder="https://www.example.com/listing/12345" required />
                  </div>
                </div>
              )}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div>
                  <label className="input-label">Listing Price (optional)</label>
                  <input type="number" value={price} onChange={e => setPrice(e.target.value)} className="input-field" placeholder="e.g., 1500" />
                </div>
                <div>
                  <label className="input-label">Location (optional)</label>
                  <input type="text" value={location} onChange={e => setLocation(e.target.value)} className="input-field" placeholder="e.g., Toronto, Downtown" />
                </div>
              </div>
              <button type="submit" disabled={loading} className="btn btn-lg btn-primary w-full">
                {loading ? <><Loader className="h-5 w-5 animate-spin" /> Analyzing...</> : <><Shield className="h-5 w-5" /> Analyze for Fraud</>}
              </button>
            </form>
            {result && <ListingResultCard result={result} getRiskBadge={getRiskBadge} />}
          </>
        )}

        {/* ====== MESSAGE ANALYSIS (message_analysis_engine) ====== */}
        {activeTab === 'message' && (
          <>
            <div className="card p-6 mb-6">
              <div className="flex items-center gap-2 mb-4">
                <MessageSquare className="h-5 w-5 text-tenant-500" />
                <h3 className="text-base font-display font-bold text-surface-900 dark:text-white">Analyze a Landlord Message</h3>
              </div>
              <p className="text-sm text-surface-500 dark:text-surface-400 mb-4">
                Paste a message from a landlord or listing contact. Our AI detects social engineering tactics, urgency pressure, and scam patterns.
              </p>
              <form onSubmit={handleMessageAnalysis} className="space-y-4">
                <textarea value={messageText} onChange={e => setMessageText(e.target.value)}
                  className="input-field min-h-[160px] resize-y"
                  placeholder='e.g., "Hi! The apartment is available immediately. I am currently abroad but can send keys via courier. Please send a $500 deposit via e-transfer to hold it..."'
                  required minLength={10} />
                <button type="submit" disabled={messageLoading} className="btn btn-lg btn-primary w-full">
                  {messageLoading ? <><Loader className="h-5 w-5 animate-spin" /> Analyzing Message...</> : <><MessageSquare className="h-5 w-5" /> Analyze Message</>}
                </button>
              </form>
            </div>
            {messageResult && (
              <div className="card animate-fade-in-up">
                <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white">Message Analysis Result</h3>
                    <span className={getRiskBadge(messageResult.risk?.level).cls}>{getRiskBadge(messageResult.risk?.level).label}</span>
                  </div>
                  {messageResult.risk?.score != null && (
                    <div className="mt-3">
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-surface-500">Risk Score</span>
                        <span className="font-bold">{(messageResult.risk.score * 100).toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-surface-100 dark:bg-surface-800 rounded-full h-2.5">
                        <div className={`h-2.5 rounded-full ${messageResult.risk.score > 0.6 ? 'bg-red-500' : messageResult.risk.score > 0.3 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                          style={{ width: `${messageResult.risk.score * 100}%` }} />
                      </div>
                    </div>
                  )}
                  {messageResult.risk?.confidence != null && (
                    <p className="text-xs text-surface-500 mt-2">Confidence: {(messageResult.risk.confidence * 100).toFixed(0)}%</p>
                  )}
                </div>
                {messageResult.tactics?.detected && messageResult.tactics.detected.length > 0 && (
                  <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Scam Tactics Detected ({messageResult.tactics.detected.length})</h4>
                    <div className="space-y-2">
                      {messageResult.tactics.detected.map((tactic, i) => (
                        <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-red-50 dark:bg-red-900/10">
                          <AlertTriangle className="h-4 w-4 mt-0.5 text-red-500 flex-shrink-0" />
                          <div>
                            <p className="text-sm font-medium text-surface-900 dark:text-white capitalize">{tactic.replace(/_/g, ' ')}</p>
                            {messageResult.tactics.evidence?.[tactic] && messageResult.tactics.evidence[tactic].length > 0 && (
                              <p className="text-xs text-surface-500 mt-0.5">Evidence: {messageResult.tactics.evidence[tactic].slice(0, 3).join(', ')}</p>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {messageResult.nlp_scores && (
                  <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">NLP Analysis</h4>
                    <div className="grid grid-cols-3 gap-4">
                      {[
                        { label: 'Sentiment', value: messageResult.nlp_scores.sentiment, range: [-1, 1], format: v => v.toFixed(2) },
                        { label: 'Urgency', value: messageResult.nlp_scores.urgency, range: [0, 1], format: v => (v * 100).toFixed(0) + '%' },
                        { label: 'Manipulation', value: messageResult.nlp_scores.manipulation, range: [0, 1], format: v => (v * 100).toFixed(0) + '%' },
                      ].map(({ label, value, format }) => (
                        <div key={label} className="text-center p-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                          <p className="text-xs text-surface-500 mb-1">{label}</p>
                          <p className="text-lg font-bold text-surface-900 dark:text-white">{format(value)}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {messageResult.patterns && (messageResult.patterns.suspicious_phrases?.length > 0 || messageResult.patterns.payment_mentions?.length > 0 || messageResult.patterns.contact_redirects?.length > 0) && (
                  <div className="p-6">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Suspicious Patterns</h4>
                    <div className="space-y-2">
                      {messageResult.patterns.suspicious_phrases?.map((phrase, i) => (
                        <div key={`sp-${i}`} className="flex items-start gap-2 text-sm text-surface-700 dark:text-surface-300">
                          <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5 flex-shrink-0" />
                          <span>{phrase}</span>
                        </div>
                      ))}
                      {messageResult.patterns.payment_mentions?.map((m, i) => (
                        <div key={`pm-${i}`} className="flex items-start gap-2 text-sm text-surface-700 dark:text-surface-300">
                          <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                          <span>Payment mention: {m}</span>
                        </div>
                      ))}
                      {messageResult.patterns.contact_redirects?.map((r, i) => (
                        <div key={`cr-${i}`} className="flex items-start gap-2 text-sm text-surface-700 dark:text-surface-300">
                          <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                          <span>Contact redirect: {r}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}

        {/* ====== CONVERSATION ANALYSIS (message_analysis_engine) ====== */}
        {activeTab === 'conversation' && (
          <>
            <div className="card p-6 mb-6">
              <div className="flex items-center gap-2 mb-4">
                <Send className="h-5 w-5 text-tenant-500" />
                <h3 className="text-base font-display font-bold text-surface-900 dark:text-white">Analyze a Conversation</h3>
              </div>
              <p className="text-sm text-surface-500 dark:text-surface-400 mb-4">
                Add the back-and-forth messages from your rental conversation. Our AI analyzes the full flow for escalating pressure, inconsistencies, and manipulation patterns.
              </p>
              <form onSubmit={handleConversationAnalysis} className="space-y-3">
                {conversationMessages.map((msg, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <select value={msg.sender} onChange={e => updateConversationMessage(idx, 'sender', e.target.value)}
                      className="input-field w-28 flex-shrink-0 text-sm">
                      <option value="landlord">Landlord</option>
                      <option value="tenant">You</option>
                    </select>
                    <textarea value={msg.text} onChange={e => updateConversationMessage(idx, 'text', e.target.value)}
                      className="input-field min-h-[60px] resize-y flex-1" placeholder={`Message ${idx + 1}...`} required />
                    <button type="button" onClick={() => removeConversationMessage(idx)}
                      className="btn btn-ghost btn-sm text-red-500 mt-1" disabled={conversationMessages.length <= 1}>
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                ))}
                <button type="button" onClick={addConversationMessage} className="btn btn-sm btn-secondary w-full">
                  <Plus className="h-4 w-4" /> Add Message
                </button>
                <button type="submit" disabled={conversationLoading} className="btn btn-lg btn-primary w-full">
                  {conversationLoading ? <><Loader className="h-5 w-5 animate-spin" /> Analyzing...</> : <><Send className="h-5 w-5" /> Analyze Conversation</>}
                </button>
              </form>
            </div>
            {conversationResult && (
              <div className="card animate-fade-in-up">
                <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white">Conversation Analysis</h3>
                    <span className={getRiskBadge(conversationResult.summary?.overall_risk).cls}>{getRiskBadge(conversationResult.summary?.overall_risk).label}</span>
                  </div>
                  {conversationResult.summary?.risk_score != null && (
                    <div className="mt-3">
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-surface-500">Overall Risk</span>
                        <span className="font-bold">{(conversationResult.summary.risk_score * 100).toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-surface-100 dark:bg-surface-800 rounded-full h-2.5">
                        <div className={`h-2.5 rounded-full ${conversationResult.summary.risk_score > 0.6 ? 'bg-red-500' : conversationResult.summary.risk_score > 0.3 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                          style={{ width: `${conversationResult.summary.risk_score * 100}%` }} />
                      </div>
                    </div>
                  )}
                  {conversationResult.summary && (
                    <div className="flex gap-4 mt-3 text-xs text-surface-500">
                      <span>Messages: {conversationResult.summary.total_messages}</span>
                      <span>Analyzed: {conversationResult.summary.analyzed}</span>
                      {conversationResult.summary.confidence != null && <span>Confidence: {(conversationResult.summary.confidence * 100).toFixed(0)}%</span>}
                    </div>
                  )}
                </div>
                {conversationResult.escalation?.detected && (
                  <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                    <h4 className="text-sm font-semibold text-red-600 dark:text-red-400 mb-2 flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4" /> Escalation Detected
                    </h4>
                    <p className="text-sm text-surface-600 dark:text-surface-400">
                      Risk escalated at message{conversationResult.escalation.escalation_points?.length > 1 ? 's' : ''}: {conversationResult.escalation.escalation_points?.map(p => `#${p + 1}`).join(', ') || 'N/A'}
                    </p>
                  </div>
                )}
                {conversationResult.red_flags && conversationResult.red_flags.length > 0 && (
                  <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Red Flags ({conversationResult.red_flags.length})</h4>
                    <div className="space-y-2">
                      {conversationResult.red_flags.map((flag, i) => (
                        <div key={i} className="flex items-start gap-2 p-2 rounded-lg bg-red-50 dark:bg-red-900/10 text-sm text-red-700 dark:text-red-400">
                          <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                          {typeof flag === 'string' ? flag : flag.description || flag.text || flag.flag || JSON.stringify(flag)}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {conversationResult.action_items && conversationResult.action_items.length > 0 && (
                  <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Action Items</h4>
                    <ul className="space-y-2">
                      {conversationResult.action_items.map((item, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-surface-700 dark:text-surface-300">
                          <CheckCircle className="h-4 w-4 text-tenant-500 mt-0.5 flex-shrink-0" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {conversationResult.recommendation && (
                  <div className="p-6">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-2">Recommendation</h4>
                    <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed whitespace-pre-line">
                      {conversationResult.recommendation}
                    </p>
                  </div>
                )}
              </div>
            )}
          </>
        )}

        {/* ====== XAI EXPLANATION (real_xai_engine + explainability_engine) ====== */}
        {activeTab === 'xai' && (
          <>
            <div className="card p-6 mb-6">
              <div className="flex items-center gap-2 mb-4">
                <Lightbulb className="h-5 w-5 text-tenant-500" />
                <h3 className="text-base font-display font-bold text-surface-900 dark:text-white">Explainable AI (XAI)</h3>
              </div>
              <p className="text-sm text-surface-500 dark:text-surface-400 mb-4">
                Understand <em>why</em> our AI classifies text as fraudulent. See which words and patterns trigger the fraud detection, with attention-weight visualization and counterfactual analysis.
              </p>
              <form onSubmit={handleXaiExplain} className="space-y-4">
                <textarea value={xaiText} onChange={e => setXaiText(e.target.value)}
                  className="input-field min-h-[160px] resize-y"
                  placeholder="Paste listing or message text to explain the AI's reasoning..." required minLength={10} />
                <button type="submit" disabled={xaiLoading} className="btn btn-lg btn-primary w-full">
                  {xaiLoading ? <><Loader className="h-5 w-5 animate-spin" /> Generating Explanation...</> : <><Brain className="h-5 w-5" /> Explain AI Decision</>}
                </button>
              </form>
            </div>
            {xaiResult && (() => {
              const allTokens = xaiResult.token_level_explanation?.all_tokens || []
              const fraudTokens = xaiResult.token_level_explanation?.top_fraud_indicators || []
              const safeTokens = xaiResult.token_level_explanation?.top_safe_indicators || []
              const reasoning = xaiResult.reasoning_chain || []
              const whatIf = xaiResult.what_if_analysis || []
              const contributors = xaiResult.top_contributors || []
              const isFraud = xaiResult.is_fraud ?? (xaiResult.risk_score > 0.5)
              const riskScore = xaiResult.risk_score ?? 0
              const methods = xaiResult.methods_used || []
              const riskPct = (riskScore * 100).toFixed(0)
              return (
              <div className="card animate-fade-in-up">
                {/* Header */}
                <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white">AI Explanation</h3>
                      {methods.length > 0 && (
                        <div className="flex gap-1.5 mt-1.5">
                          {methods.map((m, i) => (
                            <span key={i} className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300">{m}</span>
                          ))}
                        </div>
                      )}
                    </div>
                    <span className={`badge ${isFraud ? 'badge-danger' : riskScore > 0.3 ? 'badge-warning' : 'badge-success'}`}>
                      {isFraud ? `Fraud Detected (${riskPct}%)` : riskScore > 0.3 ? `Suspicious (${riskPct}%)` : `Likely Safe (${riskPct}%)`}
                    </span>
                  </div>
                  <div className="mt-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-surface-500">Risk Score:</span>
                      <div className="flex-1 bg-surface-100 dark:bg-surface-800 rounded-full h-2.5 max-w-xs">
                        <div className={`h-2.5 rounded-full ${riskScore > 0.6 ? 'bg-red-500' : riskScore > 0.3 ? 'bg-amber-500' : 'bg-emerald-500'}`} style={{ width: `${riskPct}%` }} />
                      </div>
                      <span className="text-xs font-semibold text-surface-700 dark:text-surface-300">{riskPct}%</span>
                    </div>
                  </div>
                </div>

                {/* Word-Level Attribution (all tokens colored) */}
                {allTokens.length > 0 && (
                  <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Word-Level Attribution</h4>
                    <div className="flex flex-wrap gap-1.5">
                      {allTokens.map((item, i) => {
                        const score = item.attribution || 0
                        const intensity = Math.min(Math.abs(score) * 3, 1)
                        const bg = score > 0 ? `rgba(239,68,68,${intensity})` : score < 0 ? `rgba(16,185,129,${intensity})` : 'transparent'
                        return (
                          <span key={i} className="px-1.5 py-0.5 rounded text-sm font-mono" style={{ backgroundColor: bg }}
                            title={`Attribution: ${score.toFixed(4)} | ${item.direction} | ${item.contribution_percent}`}>
                            {item.token}
                          </span>
                        )
                      })}
                    </div>
                    <p className="text-xs text-surface-400 mt-3">
                      <span className="inline-block w-3 h-3 rounded bg-red-400 mr-1 align-middle" /> fraud-indicating
                      <span className="inline-block w-3 h-3 rounded bg-emerald-400 mx-1 ml-3 align-middle" /> safety-indicating
                    </p>
                  </div>
                )}

                {/* Top Fraud Indicators */}
                {fraudTokens.length > 0 && (
                  <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Top Fraud Indicators</h4>
                    <div className="space-y-1.5">
                      {fraudTokens.map((item, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs">
                          <span className="w-24 font-mono text-red-600 dark:text-red-400 font-semibold truncate">{item.token}</span>
                          <div className="flex-1 bg-surface-100 dark:bg-surface-800 rounded-full h-2">
                            <div className="h-2 rounded-full bg-red-500" style={{ width: `${Math.min(Math.abs(item.attribution) * 100, 100)}%` }} />
                          </div>
                          <span className="text-surface-500 w-16 text-right">{item.contribution_percent}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Top Safe Indicators */}
                {safeTokens.length > 0 && (
                  <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Safety Indicators</h4>
                    <div className="space-y-1.5">
                      {safeTokens.map((item, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs">
                          <span className="w-24 font-mono text-emerald-600 dark:text-emerald-400 font-semibold truncate">{item.token}</span>
                          <div className="flex-1 bg-surface-100 dark:bg-surface-800 rounded-full h-2">
                            <div className="h-2 rounded-full bg-emerald-500" style={{ width: `${Math.min(Math.abs(item.attribution) * 100, 100)}%` }} />
                          </div>
                          <span className="text-surface-500 w-16 text-right">{item.contribution_percent}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Feature Contributors (from explainability engine) */}
                {contributors.length > 0 && (
                  <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Feature Contributions</h4>
                    <div className="space-y-2">
                      {contributors.map((c, i) => (
                        <div key={i} className="p-3 rounded-lg bg-surface-50 dark:bg-surface-800/50">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm font-medium text-surface-800 dark:text-surface-200">
                              {c.feature || c.factor || c.name || 'Unknown Feature'}
                            </span>
                            <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${c.direction === 'increases_risk' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'}`}>
                              {c.direction === 'increases_risk' ? '↑ Risk' : '↓ Safe'} {c.contribution_percent || ''}
                            </span>
                          </div>
                          <p className="text-xs text-surface-500 dark:text-surface-400">{c.explanation || ''}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* AI Reasoning Chain */}
                {reasoning.length > 0 && (
                  <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">AI Reasoning Chain</h4>
                    <div className="space-y-3">
                      {reasoning.map((step, i) => (
                        <div key={i} className="flex gap-3">
                          <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
                            <span className="text-xs font-bold text-primary-700 dark:text-primary-300">{step.step}</span>
                          </div>
                          <div className="flex-1">
                            <p className="text-sm font-medium text-surface-800 dark:text-surface-200">{step.description}</p>
                            {step.evidence?.length > 0 && (
                              <div className="mt-1 flex flex-wrap gap-1">
                                {step.evidence.map((ev, j) => (
                                  <span key={j} className="text-[10px] px-1.5 py-0.5 rounded bg-surface-100 dark:bg-surface-700 text-surface-500">{ev}</span>
                                ))}
                              </div>
                            )}
                            <span className="text-[10px] text-surface-400 mt-0.5 block">via {step.method} | conf: {(step.confidence * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* What-If Analysis */}
                {whatIf.length > 0 && (
                  <div className="p-6 border-b border-surface-200 dark:border-surface-700">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">What-If Analysis</h4>
                    <p className="text-xs text-surface-500 mb-3">What would change the AI's decision?</p>
                    <div className="space-y-3">
                      {whatIf.map((item, i) => (
                        <div key={i} className="p-4 rounded-xl bg-blue-50 dark:bg-blue-900/10 border border-blue-100 dark:border-blue-800/20">
                          <p className="text-sm text-surface-800 dark:text-surface-200 mb-2">
                            {typeof item === 'string' ? item : item.explanation || item.description || ''}
                          </p>
                          {typeof item === 'object' && item.feature && (
                            <div className="flex flex-wrap gap-3 text-xs">
                              <span className="px-2 py-1 rounded bg-white dark:bg-surface-800 text-surface-600 dark:text-surface-400">
                                Feature: <strong>{item.feature}</strong>
                              </span>
                              {item.original_risk && (
                                <span className="px-2 py-1 rounded bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400">
                                  Current Risk: <strong>{item.original_risk}</strong>
                                </span>
                              )}
                              {item.new_risk && (
                                <span className="px-2 py-1 rounded bg-emerald-50 dark:bg-emerald-900/20 text-emerald-600 dark:text-emerald-400">
                                  New Risk: <strong>{item.new_risk}</strong>
                                </span>
                              )}
                              {item.risk_reduction && (
                                <span className="px-2 py-1 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
                                  ↓ {item.risk_reduction}
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Summary */}
                {xaiResult.summary && (
                  <div className="p-6">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-2">Summary</h4>
                    <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed whitespace-pre-line">{xaiResult.summary}</p>
                  </div>
                )}
              </div>
              )
            })()}
          </>
        )}
      </div>
    </TenantLayout>
  )
}

/* Extracted listing result sub-component */
const ListingResultCard = ({ result, getRiskBadge }) => (
  <div className="card animate-fade-in-up">
    <div className="p-6 border-b border-surface-200 dark:border-surface-700">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white">Analysis Result</h3>
          <p className="text-sm text-surface-500 dark:text-surface-400 mt-1">AI confidence: {(result.confidence * 100).toFixed(1)}%</p>
        </div>
        <div className="text-right">
          <div className="text-3xl font-display font-bold text-surface-900 dark:text-white">{result.risk_score}</div>
          <div className={getRiskBadge(result.risk_level).cls}>{getRiskBadge(result.risk_level).label}</div>
        </div>
      </div>
    </div>
    {result.risk_story && (
      <div className="p-6 border-b border-surface-200 dark:border-surface-700">
        <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-2">Risk Assessment</h4>
        <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed">{result.risk_story}</p>
      </div>
    )}
    {result.indicators && result.indicators.length > 0 && (
      <div className="p-6">
        <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-4">Risk Indicators ({result.indicators.length})</h4>
        <div className="space-y-3">
          {result.indicators.map((indicator, i) => (
            <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-surface-50 dark:bg-surface-800">
              <AlertTriangle className={`h-4 w-4 mt-0.5 flex-shrink-0 ${indicator.severity >= 3 ? 'text-accent-red' : indicator.severity >= 2 ? 'text-accent-amber' : 'text-accent-blue'}`} />
              <div>
                <p className="text-sm font-medium text-surface-900 dark:text-white">{indicator.description || indicator.type}</p>
                {indicator.evidence && indicator.evidence.length > 0 && (
                  <p className="text-xs text-surface-500 dark:text-surface-400 mt-1">Evidence: {indicator.evidence.join(', ')}</p>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    )}
    {/* Price Analysis Detail */}
    {result.price_analysis && result.price_analysis.risk_level !== 'normal' && (
      <div className="p-6 border-t border-surface-200 dark:border-surface-700">
        <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Price Market Analysis</h4>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
          <div className="p-3 rounded-xl bg-surface-50 dark:bg-surface-800 text-center">
            <p className="text-xs text-surface-500 dark:text-surface-400">Listed Price</p>
            <p className="text-lg font-bold text-surface-900 dark:text-white">${result.price_analysis.listing_price?.toLocaleString()}</p>
          </div>
          <div className="p-3 rounded-xl bg-surface-50 dark:bg-surface-800 text-center">
            <p className="text-xs text-surface-500 dark:text-surface-400">Market Average</p>
            <p className="text-lg font-bold text-surface-900 dark:text-white">${result.price_analysis.market_average?.toLocaleString()}</p>
          </div>
          <div className="p-3 rounded-xl bg-surface-50 dark:bg-surface-800 text-center">
            <p className="text-xs text-surface-500 dark:text-surface-400">Deviation</p>
            <p className={`text-lg font-bold ${result.price_analysis.price_deviation_percent < -30 ? 'text-red-500' : result.price_analysis.price_deviation_percent < -15 ? 'text-amber-500' : 'text-emerald-500'}`}>
              {result.price_analysis.price_deviation_percent?.toFixed(1)}%
            </p>
          </div>
          <div className="p-3 rounded-xl bg-surface-50 dark:bg-surface-800 text-center">
            <p className="text-xs text-surface-500 dark:text-surface-400">Risk Level</p>
            <p className={`text-sm font-bold ${result.price_analysis.risk_level === 'extremely_low' || result.price_analysis.risk_level === 'suspiciously_low' ? 'text-red-500' : 'text-amber-500'}`}>
              {result.price_analysis.risk_level?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
            </p>
          </div>
        </div>
        {result.price_analysis.explanation && (
          <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed">{result.price_analysis.explanation}</p>
        )}
      </div>
    )}
    {result.ai_components && (
      <div className="p-6 border-t border-surface-200 dark:border-surface-700">
        <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">AI Components Used</h4>
        <div className="flex flex-wrap gap-2">
          {Object.entries(result.ai_components).map(([key, enabled]) => (
            <span key={key} className={`badge ${enabled ? 'badge-success' : 'badge-info'}`}>
              {enabled ? <CheckCircle className="h-3 w-3 mr-1" /> : null}
              {key.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      </div>
    )}
  </div>
)

export default TenantAnalyze
