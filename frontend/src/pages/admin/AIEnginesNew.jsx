import { useState, useEffect } from 'react'
import AdminLayout from '../../components/layouts/AdminLayout'
import { adminAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  Cpu, Loader, RefreshCw, CheckCircle, XCircle, Brain, MessageSquare,
  FileText, Shield, Zap, AlertTriangle, Layers, BarChart3, Settings
} from 'lucide-react'

const AdminAIEngines = () => {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('status')

  // BERT test
  const [testText, setTestText] = useState('')
  const [testResult, setTestResult] = useState(null)
  const [testing, setTesting] = useState(false)

  // Message analysis test
  const [msgTestText, setMsgTestText] = useState('')
  const [msgTestResult, setMsgTestResult] = useState(null)
  const [msgTesting, setMsgTesting] = useState(false)

  // Cross-document test
  const [crossDocData, setCrossDocData] = useState({ doc1_text: '', doc2_text: '' })
  const [crossDocResult, setCrossDocResult] = useState(null)
  const [crossDocTesting, setCrossDocTesting] = useState(false)

  // Training data stats
  const [trainingStats, setTrainingStats] = useState(null)
  const [trainingStatsLoading, setTrainingStatsLoading] = useState(false)

  // Preprocessing status
  const [preprocessingStatus, setPreprocessingStatus] = useState(null)
  const [preprocessingLoading, setPreprocessingLoading] = useState(false)

  useEffect(() => { fetchStatus() }, [])

  const fetchStatus = async () => {
    try {
      const res = await adminAPI.getAIEnginesStatus()
      setStatus(res.data)
    } catch { toast.error('Failed to fetch AI engine status') }
    finally { setLoading(false) }
  }

  const handleTestBert = async () => {
    if (!testText.trim()) return toast.error('Enter text to test')
    setTesting(true)
    setTestResult(null)
    try {
      const formData = new FormData()
      formData.append('text', testText)
      const res = await adminAPI.testBert(formData)
      setTestResult(res.data)
    } catch (e) { toast.error(e.response?.data?.detail || 'Test failed') }
    finally { setTesting(false) }
  }

  const handleTestMessage = async () => {
    if (!msgTestText.trim()) return toast.error('Enter message text to test')
    setMsgTesting(true)
    setMsgTestResult(null)
    try {
      const formData = new FormData()
      formData.append('message', msgTestText)
      const res = await adminAPI.testMessageAnalysis(formData)
      setMsgTestResult(res.data)
    } catch (e) { toast.error(e.response?.data?.detail || 'Message analysis test failed') }
    finally { setMsgTesting(false) }
  }

  const handleTestCrossDoc = async () => {
    if (!crossDocData.doc1_text.trim() || !crossDocData.doc2_text.trim()) return toast.error('Enter text for both documents')
    setCrossDocTesting(true)
    setCrossDocResult(null)
    try {
      const payload = {
        documents: [
          { name: 'document_1', type: 'general', text: crossDocData.doc1_text },
          { name: 'document_2', type: 'general', text: crossDocData.doc2_text }
        ]
      }
      const res = await adminAPI.testCrossDocument(payload)
      setCrossDocResult(res.data)
    } catch (e) { toast.error(e.response?.data?.detail || 'Cross-document test failed') }
    finally { setCrossDocTesting(false) }
  }

  const loadTrainingStats = async () => {
    setTrainingStatsLoading(true)
    try {
      const res = await adminAPI.getTrainingDataStats()
      setTrainingStats(res.data)
    } catch (e) { toast.error('Failed to load training data stats') }
    finally { setTrainingStatsLoading(false) }
  }

  const loadPreprocessingStatus = async () => {
    setPreprocessingLoading(true)
    try {
      const res = await adminAPI.getPreprocessingStatus()
      setPreprocessingStatus(res.data)
    } catch (e) { toast.error('Failed to load preprocessing status') }
    finally { setPreprocessingLoading(false) }
  }

  const tabs = [
    { id: 'status', icon: Cpu, label: 'Engine Status' },
    { id: 'bert', icon: Brain, label: 'Test BERT' },
    { id: 'message', icon: MessageSquare, label: 'Test Message AI' },
    { id: 'crossdoc', icon: Layers, label: 'Test Cross-Doc' },
    { id: 'training', icon: BarChart3, label: 'Training Data' },
    { id: 'preprocessing', icon: Settings, label: 'Preprocessing' },
  ]

  return (
    <AdminLayout title="AI Engines" subtitle="Monitor, test, and manage all AI/ML components">
      <div className="space-y-6">
        {/* Tabs */}
        <div className="flex items-center gap-2 flex-wrap">
          {tabs.map(t => (
            <button key={t.id} onClick={() => {
              setActiveTab(t.id)
              if (t.id === 'training' && !trainingStats) loadTrainingStats()
              if (t.id === 'preprocessing' && !preprocessingStatus) loadPreprocessingStatus()
            }} className={`btn btn-sm ${activeTab === t.id ? 'btn-primary' : 'btn-secondary'}`}>
              <t.icon className="h-3.5 w-3.5" /> {t.label}
            </button>
          ))}
          <button onClick={fetchStatus} className="btn btn-sm btn-secondary ml-auto">
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </button>
        </div>

        {/* ====== ENGINE STATUS ====== */}
        {activeTab === 'status' && (
          loading ? (
            <div className="flex justify-center py-16"><Loader className="h-8 w-8 animate-spin text-primary-600" /></div>
          ) : status ? (
            <div className="space-y-6">
              {/* Summary Cards */}
              {status.summary && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700 text-center">
                    <p className="text-2xl font-display font-bold text-surface-900 dark:text-white">{status.summary.total_engines}</p>
                    <p className="text-[10px] text-surface-500 uppercase font-semibold">Total Engines</p>
                  </div>
                  <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700 text-center">
                    <p className="text-2xl font-display font-bold text-emerald-600 dark:text-emerald-400">{status.summary.available}</p>
                    <p className="text-[10px] text-surface-500 uppercase font-semibold">Available</p>
                  </div>
                  <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700 text-center">
                    <p className="text-2xl font-display font-bold text-primary-600 dark:text-primary-400">{status.summary.real_ai_components}</p>
                    <p className="text-[10px] text-surface-500 uppercase font-semibold">Real AI Components</p>
                  </div>
                  <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700 text-center">
                    <p className="text-2xl font-display font-bold text-emerald-600 dark:text-emerald-400">{status.summary.health_percentage}</p>
                    <p className="text-[10px] text-surface-500 uppercase font-semibold">Health</p>
                  </div>
                </div>
              )}

              {/* Engine Cards */}
              {status.engines && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(status.engines).map(([key, engine]) => {
                    if (typeof engine !== 'object' || !engine) return null
                    const isActive = engine.status === 'available' || engine.status === 'ready' || engine.status === 'active'
                    const isError = engine.status === 'error'
                    return (
                      <div key={key} className="card p-5">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <Cpu className={`h-5 w-5 ${isActive ? 'text-emerald-600' : isError ? 'text-red-500' : 'text-surface-400'}`} />
                            <span className="text-sm font-semibold text-surface-900 dark:text-white">{engine.name || key.replace(/_/g, ' ')}</span>
                          </div>
                          {isActive ? <CheckCircle className="h-5 w-5 text-emerald-500" /> : <XCircle className="h-5 w-5 text-surface-400" />}
                        </div>
                        {engine.description && (
                          <p className="text-xs text-surface-500 mb-2">{engine.description}</p>
                        )}
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className={`badge ${isActive ? 'badge-success' : isError ? 'badge-danger' : 'badge-warning'}`}>
                            {engine.status}
                          </span>
                          {engine.is_real_ai && <span className="badge badge-info text-[10px]">Real AI</span>}
                          {engine.is_real_ai === false && <span className="badge badge-secondary text-[10px]">Rule-based</span>}
                        </div>
                        {engine.error && (
                          <p className="text-xs text-red-600 dark:text-red-400 mt-2">{engine.error}</p>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          ) : (
            <div className="card text-center py-16">
              <Cpu className="h-16 w-16 text-surface-300 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-surface-600">Failed to load engine status</h3>
            </div>
          )
        )}

        {/* ====== BERT TEST ====== */}
        {activeTab === 'bert' && (
          <div className="card p-6">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary-600" /> Test BERT Fraud Classifier
            </h3>
            <p className="text-sm text-surface-500 mb-4">Test the fine-tuned DistilBERT model for fraud text classification</p>
            <div className="space-y-4">
              <textarea value={testText} onChange={e => setTestText(e.target.value)}
                className="input-field min-h-[100px] resize-y" placeholder="Enter listing text to test fraud detection..." />
              <button onClick={handleTestBert} disabled={testing} className="btn btn-primary">
                {testing ? <><Loader className="h-4 w-4 animate-spin" /> Testing...</> : <><Zap className="h-4 w-4" /> Run BERT Test</>}
              </button>
            </div>
            {testResult && (
              <div className="mt-4 space-y-4 animate-fade-in-up">
                {/* Prediction Result */}
                <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-semibold text-surface-900 dark:text-white">Prediction</span>
                    <span className={`badge ${testResult.prediction?.is_fraud ? 'badge-danger' : 'badge-success'}`}>
                      {testResult.prediction?.is_fraud ? 'Fraud Detected' : 'Safe'}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                      <p className="text-[10px] text-surface-500 uppercase font-semibold">Confidence</p>
                      <p className="text-sm font-bold text-surface-900 dark:text-white">
                        {((testResult.prediction?.confidence || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                      <p className="text-[10px] text-surface-500 uppercase font-semibold">Fraud Probability</p>
                      <p className="text-sm font-bold text-red-600 dark:text-red-400">
                        {((testResult.prediction?.fraud_probability || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                      <p className="text-[10px] text-surface-500 uppercase font-semibold">Legitimate Probability</p>
                      <p className="text-sm font-bold text-emerald-600 dark:text-emerald-400">
                        {((testResult.prediction?.legitimate_probability || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                      <p className="text-[10px] text-surface-500 uppercase font-semibold">Label</p>
                      <p className="text-sm font-bold text-surface-900 dark:text-white capitalize">
                        {testResult.prediction?.prediction_label || '—'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* XAI Explanation */}
                {testResult.explanation && (
                  <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700 space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-semibold text-surface-900 dark:text-white flex items-center gap-2">
                        <Brain className="h-4 w-4 text-purple-600" /> XAI Explanation
                      </span>
                      {testResult.explanation.methods_used && (
                        <div className="flex gap-1">
                          {testResult.explanation.methods_used.map((m, i) => (
                            <span key={i} className="badge badge-info text-[10px]">{m}</span>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Token Attributions */}
                    {testResult.explanation.token_level_explanation && (
                      <div>
                        <p className="text-xs font-semibold text-surface-700 dark:text-surface-300 mb-2">Token-Level Attributions</p>
                        <div className="flex flex-wrap gap-1.5">
                          {(testResult.explanation.token_level_explanation.all_tokens || []).map((t, i) => (
                            <span key={i} className={`text-xs font-mono px-2 py-1 rounded-lg border ${
                              t.attribution > 0.1
                                ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-700 dark:text-red-400'
                                : t.attribution < -0.1
                                ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400'
                                : 'bg-surface-100 dark:bg-surface-600 border-surface-200 dark:border-surface-500 text-surface-600 dark:text-surface-300'
                            }`} title={`${t.direction}: ${t.contribution_percent}`}>
                              {t.token} <span className="text-[9px] opacity-75">{t.attribution > 0 ? '+' : ''}{t.attribution.toFixed(2)}</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Top Fraud Indicators */}
                    {testResult.explanation.token_level_explanation?.top_fraud_indicators?.length > 0 && (
                      <div>
                        <p className="text-xs font-semibold text-red-600 dark:text-red-400 mb-1.5">Top Fraud Indicators</p>
                        <div className="space-y-1">
                          {testResult.explanation.token_level_explanation.top_fraud_indicators.map((t, i) => (
                            <div key={i} className="flex items-center gap-2">
                              <span className="text-xs font-mono font-bold text-red-700 dark:text-red-400 w-16">{t.token}</span>
                              <div className="flex-1 bg-surface-200 dark:bg-surface-600 rounded-full h-3 overflow-hidden">
                                <div className="bg-red-500 h-full rounded-full" style={{ width: `${Math.min(Math.abs(t.attribution) * 200, 100)}%` }} />
                              </div>
                              <span className="text-[10px] font-mono text-surface-500 w-12 text-right">{t.contribution_percent}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Top Safe Indicators */}
                    {testResult.explanation.token_level_explanation?.top_safe_indicators?.length > 0 && (
                      <div>
                        <p className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 mb-1.5">Top Safe Indicators</p>
                        <div className="space-y-1">
                          {testResult.explanation.token_level_explanation.top_safe_indicators.map((t, i) => (
                            <div key={i} className="flex items-center gap-2">
                              <span className="text-xs font-mono font-bold text-emerald-700 dark:text-emerald-400 w-16">{t.token}</span>
                              <div className="flex-1 bg-surface-200 dark:bg-surface-600 rounded-full h-3 overflow-hidden">
                                <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${Math.min(Math.abs(t.attribution) * 200, 100)}%` }} />
                              </div>
                              <span className="text-[10px] font-mono text-surface-500 w-12 text-right">{t.contribution_percent}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Reasoning Chain */}
                    {testResult.explanation.reasoning_chain && testResult.explanation.reasoning_chain.length > 0 && (
                      <div>
                        <p className="text-xs font-semibold text-surface-700 dark:text-surface-300 mb-2">Reasoning Chain</p>
                        <div className="space-y-2">
                          {testResult.explanation.reasoning_chain.map((step, i) => (
                            <div key={i} className="flex items-start gap-3 p-2.5 rounded-lg bg-white dark:bg-surface-600">
                              <div className="w-6 h-6 bg-primary-100 dark:bg-primary-900/30 rounded-full flex items-center justify-center flex-shrink-0">
                                <span className="text-[10px] font-bold text-primary-700 dark:text-primary-400">{step.step}</span>
                              </div>
                              <div className="flex-1 min-w-0">
                                <p className="text-xs font-semibold text-surface-900 dark:text-white">{step.description}</p>
                                <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                                  <span className="text-[10px] text-surface-500">{step.method}</span>
                                  <span className="text-[10px] text-surface-400">•</span>
                                  <span className="text-[10px] text-surface-500">{(step.confidence * 100).toFixed(0)}% confidence</span>
                                </div>
                                {step.evidence && step.evidence.length > 0 && (
                                  <div className="flex flex-wrap gap-1 mt-1">
                                    {step.evidence.map((e, j) => (
                                      <span key={j} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-surface-100 dark:bg-surface-500 text-surface-600 dark:text-surface-300">{e}</span>
                                    ))}
                                  </div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ====== MESSAGE ANALYSIS TEST (message_analysis_engine) ====== */}
        {activeTab === 'message' && (
          <div className="card p-6">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <MessageSquare className="h-5 w-5 text-primary-600" /> Test Message Analysis Engine
            </h3>
            <p className="text-sm text-surface-500 mb-4">Test the NLP engine that detects social engineering, urgency tactics, and scam patterns in messages</p>
            <div className="space-y-4">
              <textarea value={msgTestText} onChange={e => setMsgTestText(e.target.value)}
                className="input-field min-h-[100px] resize-y"
                placeholder='e.g., "Send deposit via wire transfer immediately, keys will be shipped..."' />
              <button onClick={handleTestMessage} disabled={msgTesting} className="btn btn-primary">
                {msgTesting ? <><Loader className="h-4 w-4 animate-spin" /> Testing...</> : <><MessageSquare className="h-4 w-4" /> Test Message Analysis</>}
              </button>
            </div>
            {msgTestResult && (
              <div className="mt-4 space-y-4 animate-fade-in-up">
                {/* Risk Assessment */}
                <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-semibold text-surface-900 dark:text-white">Risk Assessment</span>
                    <span className={`badge ${
                      (msgTestResult.risk?.level === 'critical' || msgTestResult.risk?.level === 'high') ? 'badge-danger'
                      : msgTestResult.risk?.level === 'medium' ? 'badge-warning'
                      : msgTestResult.risk?.level === 'low' ? 'badge-info'
                      : 'badge-success'
                    }`}>
                      {(msgTestResult.risk?.level || 'unknown').toUpperCase()} RISK
                    </span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                      <p className="text-[10px] text-surface-500 uppercase font-semibold">Risk Score</p>
                      <p className={`text-sm font-bold ${
                        (msgTestResult.risk?.score || 0) >= 0.6 ? 'text-red-600 dark:text-red-400'
                        : (msgTestResult.risk?.score || 0) >= 0.4 ? 'text-amber-600 dark:text-amber-400'
                        : 'text-emerald-600 dark:text-emerald-400'
                      }`}>
                        {((msgTestResult.risk?.score || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                      <p className="text-[10px] text-surface-500 uppercase font-semibold">Confidence</p>
                      <p className="text-sm font-bold text-surface-900 dark:text-white">
                        {((msgTestResult.risk?.confidence || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                      <p className="text-[10px] text-surface-500 uppercase font-semibold">Sender</p>
                      <p className="text-sm font-bold text-surface-900 dark:text-white capitalize">
                        {msgTestResult.sender || 'unknown'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* NLP Scores */}
                {msgTestResult.nlp_scores && (
                  <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                    <p className="text-xs font-semibold text-surface-700 dark:text-surface-300 mb-3">NLP Analysis Scores</p>
                    <div className="grid grid-cols-3 gap-3">
                      <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                        <p className="text-[10px] text-surface-500 uppercase font-semibold">Sentiment</p>
                        <p className={`text-sm font-bold ${msgTestResult.nlp_scores.sentiment < -0.3 ? 'text-red-600 dark:text-red-400' : msgTestResult.nlp_scores.sentiment > 0.3 ? 'text-emerald-600 dark:text-emerald-400' : 'text-surface-900 dark:text-white'}`}>
                          {msgTestResult.nlp_scores.sentiment?.toFixed(3)}
                        </p>
                      </div>
                      <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                        <p className="text-[10px] text-surface-500 uppercase font-semibold">Urgency</p>
                        <p className={`text-sm font-bold ${msgTestResult.nlp_scores.urgency > 0.5 ? 'text-red-600 dark:text-red-400' : 'text-surface-900 dark:text-white'}`}>
                          {((msgTestResult.nlp_scores.urgency || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                        <p className="text-[10px] text-surface-500 uppercase font-semibold">Manipulation</p>
                        <p className={`text-sm font-bold ${msgTestResult.nlp_scores.manipulation > 0.5 ? 'text-red-600 dark:text-red-400' : 'text-surface-900 dark:text-white'}`}>
                          {((msgTestResult.nlp_scores.manipulation || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Detected Tactics with Evidence */}
                {msgTestResult.tactics?.detected?.length > 0 && (
                  <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                    <p className="text-xs font-semibold text-red-600 dark:text-red-400 mb-3">
                      <AlertTriangle className="h-3.5 w-3.5 inline mr-1" />
                      Detected Tactics ({msgTestResult.tactics.detected.length})
                    </p>
                    <div className="space-y-2">
                      {msgTestResult.tactics.detected.map((tactic, i) => (
                        <div key={i} className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                          <span className="badge badge-danger text-xs capitalize mb-1.5">{tactic.replace(/_/g, ' ')}</span>
                          {msgTestResult.tactics.evidence?.[tactic]?.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-1">
                              {msgTestResult.tactics.evidence[tactic].map((ev, j) => (
                                <span key={j} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-800">
                                  "{ev}"
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Suspicious Patterns */}
                {(msgTestResult.patterns?.payment_mentions?.length > 0 || msgTestResult.patterns?.contact_redirects?.length > 0 || msgTestResult.patterns?.suspicious_phrases?.length > 0) && (
                  <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700 space-y-3">
                    <p className="text-xs font-semibold text-surface-700 dark:text-surface-300">Suspicious Patterns Detected</p>
                    {msgTestResult.patterns.payment_mentions?.length > 0 && (
                      <div>
                        <p className="text-[10px] text-red-600 dark:text-red-400 font-semibold uppercase mb-1">Payment Red Flags</p>
                        <div className="flex flex-wrap gap-1">
                          {msgTestResult.patterns.payment_mentions.map((p, i) => (
                            <span key={i} className="text-xs font-mono px-2 py-1 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400">{p}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    {msgTestResult.patterns.contact_redirects?.length > 0 && (
                      <div>
                        <p className="text-[10px] text-amber-600 dark:text-amber-400 font-semibold uppercase mb-1">Contact Redirect Attempts</p>
                        <div className="flex flex-wrap gap-1">
                          {msgTestResult.patterns.contact_redirects.map((c, i) => (
                            <span key={i} className="text-xs font-mono px-2 py-1 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 text-amber-700 dark:text-amber-400">{c}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    {msgTestResult.patterns.suspicious_phrases?.length > 0 && (
                      <div>
                        <p className="text-[10px] text-orange-600 dark:text-orange-400 font-semibold uppercase mb-1">Suspicious Phrases</p>
                        <div className="flex flex-wrap gap-1">
                          {msgTestResult.patterns.suspicious_phrases.map((s, i) => (
                            <span key={i} className="text-xs font-mono px-2 py-1 rounded-lg bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 text-orange-700 dark:text-orange-400">"{s}"</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ====== CROSS-DOCUMENT TEST (cross_document_engine) ====== */}
        {activeTab === 'crossdoc' && (
          <div className="card p-6">
            <h3 className="text-base font-display font-bold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
              <Layers className="h-5 w-5 text-primary-600" /> Test Cross-Document Engine
            </h3>
            <p className="text-sm text-surface-500 mb-4">Test the NER + fuzzy matching engine that checks consistency between two document texts</p>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="input-label">Document 1 Text</label>
                  <textarea value={crossDocData.doc1_text} onChange={e => setCrossDocData({ ...crossDocData, doc1_text: e.target.value })}
                    className="input-field min-h-[120px] resize-y" placeholder="Text extracted from first document..." />
                </div>
                <div>
                  <label className="input-label">Document 2 Text</label>
                  <textarea value={crossDocData.doc2_text} onChange={e => setCrossDocData({ ...crossDocData, doc2_text: e.target.value })}
                    className="input-field min-h-[120px] resize-y" placeholder="Text extracted from second document..." />
                </div>
              </div>
              <button onClick={handleTestCrossDoc} disabled={crossDocTesting} className="btn btn-primary">
                {crossDocTesting ? <><Loader className="h-4 w-4 animate-spin" /> Testing...</> : <><Layers className="h-4 w-4" /> Test Cross-Document</>}
              </button>
            </div>
            {crossDocResult && (
              <div className="mt-4 space-y-4 animate-fade-in-up">
                {/* Overall Assessment */}
                <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-semibold text-surface-900 dark:text-white">Consistency Assessment</span>
                    <span className={`badge ${
                      crossDocResult.overall?.consistency === 'consistent' ? 'badge-success'
                      : crossDocResult.overall?.consistency === 'minor_discrepancy' ? 'badge-warning'
                      : crossDocResult.overall?.consistency === 'critical_mismatch' ? 'badge-danger'
                      : crossDocResult.overall?.consistency === 'major_discrepancy' ? 'badge-danger'
                      : 'badge-info'
                    }`}>
                      {(crossDocResult.overall?.consistency || 'unknown').replace(/_/g, ' ').toUpperCase()}
                    </span>
                  </div>
                  {crossDocResult.summary && (
                    <p className="text-sm text-surface-700 dark:text-surface-300 mb-3">{crossDocResult.summary}</p>
                  )}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                      <p className="text-[10px] text-surface-500 uppercase font-semibold">Score</p>
                      <p className={`text-sm font-bold ${
                        (crossDocResult.overall?.score || 0) >= 0.8 ? 'text-emerald-600 dark:text-emerald-400'
                        : (crossDocResult.overall?.score || 0) >= 0.6 ? 'text-amber-600 dark:text-amber-400'
                        : 'text-red-600 dark:text-red-400'
                      }`}>
                        {((crossDocResult.overall?.score || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                      <p className="text-[10px] text-surface-500 uppercase font-semibold">Confidence</p>
                      <p className="text-sm font-bold text-surface-900 dark:text-white">
                        {((crossDocResult.overall?.confidence || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                      <p className="text-[10px] text-surface-500 uppercase font-semibold">Checks Passed</p>
                      <p className="text-sm font-bold text-emerald-600 dark:text-emerald-400">
                        {crossDocResult.checks?.passed || 0}/{crossDocResult.checks?.total || 0}
                      </p>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600">
                      <p className="text-[10px] text-surface-500 uppercase font-semibold">Checks Failed</p>
                      <p className="text-sm font-bold text-red-600 dark:text-red-400">
                        {crossDocResult.checks?.failed || 0}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Critical Issues */}
                {crossDocResult.critical_issues?.length > 0 && (
                  <div className="p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
                    <p className="text-xs font-semibold text-red-700 dark:text-red-400 mb-2">
                      <AlertTriangle className="h-3.5 w-3.5 inline mr-1" />
                      Critical Issues ({crossDocResult.critical_issues.length})
                    </p>
                    <div className="space-y-1.5">
                      {crossDocResult.critical_issues.map((issue, i) => (
                        <div key={i} className="flex items-start gap-2 text-xs text-red-700 dark:text-red-400">
                          <Shield className="h-3.5 w-3.5 mt-0.5 flex-shrink-0" />
                          <div>
                            <span className="font-semibold capitalize">{issue.type?.replace(/_/g, ' ')}:</span> {issue.description}
                            {issue.documents && <span className="text-red-500 ml-1">({issue.documents.join(' vs ')})</span>}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Warnings */}
                {crossDocResult.warnings?.length > 0 && (
                  <div className="p-4 rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
                    <p className="text-xs font-semibold text-amber-700 dark:text-amber-400 mb-2">
                      <AlertTriangle className="h-3.5 w-3.5 inline mr-1" />
                      Warnings ({crossDocResult.warnings.length})
                    </p>
                    <div className="space-y-1.5">
                      {crossDocResult.warnings.map((w, i) => (
                        <div key={i} className="flex items-start gap-2 text-xs text-amber-700 dark:text-amber-400">
                          <AlertTriangle className="h-3.5 w-3.5 mt-0.5 flex-shrink-0" />
                          <div>
                            <span className="font-semibold capitalize">{w.type?.replace(/_/g, ' ')}:</span> {w.description}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Consistency Checks Detail */}
                {crossDocResult.consistency_checks?.length > 0 && (
                  <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                    <p className="text-xs font-semibold text-surface-700 dark:text-surface-300 mb-3">Consistency Checks ({crossDocResult.consistency_checks.length})</p>
                    <div className="space-y-2">
                      {crossDocResult.consistency_checks.map((check, i) => (
                        <div key={i} className={`p-2.5 rounded-lg border ${check.is_consistent
                          ? 'bg-emerald-50 dark:bg-emerald-900/10 border-emerald-200 dark:border-emerald-800'
                          : 'bg-red-50 dark:bg-red-900/10 border-red-200 dark:border-red-800'
                        }`}>
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-semibold capitalize text-surface-900 dark:text-white">
                              {check.entity_type?.replace(/_/g, ' ')}
                            </span>
                            <div className="flex items-center gap-2">
                              <span className="text-[10px] text-surface-500">Similarity: {((check.similarity || 0) * 100).toFixed(0)}%</span>
                              {check.is_consistent
                                ? <CheckCircle className="h-3.5 w-3.5 text-emerald-500" />
                                : <XCircle className="h-3.5 w-3.5 text-red-500" />
                              }
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-2 text-[10px]">
                            <div>
                              <span className="text-surface-500">{check.documents?.doc1?.source}:</span>{' '}
                              <span className="font-mono text-surface-700 dark:text-surface-300">{check.documents?.doc1?.value}</span>
                            </div>
                            <div>
                              <span className="text-surface-500">{check.documents?.doc2?.source}:</span>{' '}
                              <span className="font-mono text-surface-700 dark:text-surface-300">{check.documents?.doc2?.value}</span>
                            </div>
                          </div>
                          <p className="text-[10px] text-surface-500 mt-1">{check.explanation}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Verified / Unverified Fields */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {crossDocResult.verified_fields?.length > 0 && (
                    <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                      <p className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 mb-2">Verified Fields</p>
                      <div className="flex flex-wrap gap-1">
                        {crossDocResult.verified_fields.map((f, i) => (
                          <span key={i} className="text-xs px-2 py-1 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400 capitalize">
                            {f.replace(/_/g, ' ')}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {crossDocResult.unverified_fields?.length > 0 && (
                    <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                      <p className="text-xs font-semibold text-amber-600 dark:text-amber-400 mb-2">Unverified Fields</p>
                      <div className="flex flex-wrap gap-1">
                        {crossDocResult.unverified_fields.map((f, i) => (
                          <span key={i} className="text-xs px-2 py-1 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 text-amber-700 dark:text-amber-400 capitalize">
                            {f.replace(/_/g, ' ')}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Recommendation */}
                {crossDocResult.recommendation && (
                  <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                    <p className="text-xs font-semibold text-surface-700 dark:text-surface-300 mb-1">Recommendation</p>
                    <p className="text-sm text-surface-600 dark:text-surface-400">{crossDocResult.recommendation}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ====== TRAINING DATA STATS ====== */}
        {activeTab === 'training' && (
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-display font-bold text-surface-900 dark:text-white flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-primary-600" /> Training Data Statistics
              </h3>
              <button onClick={loadTrainingStats} className="btn btn-sm btn-secondary">
                <RefreshCw className="h-3.5 w-3.5" /> Refresh
              </button>
            </div>
            {trainingStatsLoading ? (
              <div className="flex justify-center py-8"><Loader className="h-6 w-6 animate-spin text-primary-600" /></div>
            ) : trainingStats ? (
              <div className="space-y-6">
                {/* Fraud Dataset */}
                {trainingStats.fraud_dataset && (
                  <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                    <div className="flex items-center justify-between mb-3">
                      <p className="text-sm font-semibold text-surface-900 dark:text-white flex items-center gap-2">
                        <FileText className="h-4 w-4 text-primary-600" /> Fraud Dataset (CSV)
                      </p>
                      <span className={`badge ${trainingStats.fraud_dataset.exists ? 'badge-success' : 'badge-danger'}`}>
                        {trainingStats.fraud_dataset.exists ? 'Available' : 'Not Found'}
                      </span>
                    </div>
                    {trainingStats.fraud_dataset.exists && (
                      <>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-3">
                          <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600 text-center">
                            <p className="text-xl font-display font-bold text-surface-900 dark:text-white">
                              {(trainingStats.fraud_dataset.total_rows || 0).toLocaleString()}
                            </p>
                            <p className="text-[10px] text-surface-500 uppercase font-semibold">Total Rows</p>
                          </div>
                          <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600 text-center">
                            <p className="text-xl font-display font-bold text-red-600 dark:text-red-400">
                              {(trainingStats.fraud_dataset.fraud_examples || 0).toLocaleString()}
                            </p>
                            <p className="text-[10px] text-surface-500 uppercase font-semibold">Fraud Examples</p>
                          </div>
                          <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600 text-center">
                            <p className="text-xl font-display font-bold text-emerald-600 dark:text-emerald-400">
                              {(trainingStats.fraud_dataset.safe_examples || 0).toLocaleString()}
                            </p>
                            <p className="text-[10px] text-surface-500 uppercase font-semibold">Safe Examples</p>
                          </div>
                        </div>
                        {/* Class balance bar */}
                        {trainingStats.fraud_dataset.total_rows > 0 && (
                          <div>
                            <p className="text-[10px] text-surface-500 uppercase font-semibold mb-1">Class Balance</p>
                            <div className="flex h-4 rounded-full overflow-hidden bg-surface-200 dark:bg-surface-600">
                              <div className="bg-red-500 h-full" style={{ width: `${(trainingStats.fraud_dataset.fraud_examples / trainingStats.fraud_dataset.total_rows * 100)}%` }} />
                              <div className="bg-emerald-500 h-full flex-1" />
                            </div>
                            <div className="flex justify-between mt-1">
                              <span className="text-[10px] text-red-600 dark:text-red-400 font-semibold">
                                Fraud {(trainingStats.fraud_dataset.fraud_examples / trainingStats.fraud_dataset.total_rows * 100).toFixed(1)}%
                              </span>
                              <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-semibold">
                                Safe {(trainingStats.fraud_dataset.safe_examples / trainingStats.fraud_dataset.total_rows * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        )}
                        {/* Fraud types breakdown */}
                        {trainingStats.fraud_dataset.fraud_types && Object.keys(trainingStats.fraud_dataset.fraud_types).length > 0 && (
                          <div className="mt-3">
                            <p className="text-[10px] text-surface-500 uppercase font-semibold mb-2">Fraud Types</p>
                            <div className="flex flex-wrap gap-1.5">
                              {Object.entries(trainingStats.fraud_dataset.fraud_types).map(([type, count]) => (
                                <span key={type} className="text-xs px-2 py-1 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 capitalize">
                                  {type.replace(/_/g, ' ')} ({count})
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        {trainingStats.fraud_dataset.path && (
                          <p className="text-[10px] text-surface-400 font-mono mt-2 truncate">{trainingStats.fraud_dataset.path}</p>
                        )}
                      </>
                    )}
                    {trainingStats.fraud_dataset.error && (
                      <p className="text-xs text-red-600 dark:text-red-400 mt-2">Error: {trainingStats.fraud_dataset.error}</p>
                    )}
                  </div>
                )}

                {/* Feedback Data */}
                {trainingStats.feedback_data && (
                  <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                    <p className="text-sm font-semibold text-surface-900 dark:text-white mb-3 flex items-center gap-2">
                      <Zap className="h-4 w-4 text-amber-600" /> User Feedback Data (Auto-Learning)
                    </p>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600 text-center">
                        <p className="text-xl font-display font-bold text-surface-900 dark:text-white">
                          {(trainingStats.feedback_data.total_samples || 0).toLocaleString()}
                        </p>
                        <p className="text-[10px] text-surface-500 uppercase font-semibold">Total Samples</p>
                      </div>
                      <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600 text-center">
                        <p className="text-xl font-display font-bold text-red-600 dark:text-red-400">
                          {(trainingStats.feedback_data.fraud_samples || 0).toLocaleString()}
                        </p>
                        <p className="text-[10px] text-surface-500 uppercase font-semibold">Fraud Samples</p>
                      </div>
                      <div className="p-2.5 rounded-lg bg-white dark:bg-surface-600 text-center">
                        <p className="text-xl font-display font-bold text-emerald-600 dark:text-emerald-400">
                          {(trainingStats.feedback_data.safe_samples || 0).toLocaleString()}
                        </p>
                        <p className="text-[10px] text-surface-500 uppercase font-semibold">Safe Samples</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-sm text-surface-500 text-center py-8">Click refresh to load training data statistics</p>
            )}
          </div>
        )}

        {/* ====== PREPROCESSING STATUS (data_preprocessing_pipeline) ====== */}
        {activeTab === 'preprocessing' && (
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-display font-bold text-surface-900 dark:text-white flex items-center gap-2">
                <Settings className="h-5 w-5 text-primary-600" /> Data Preprocessing Pipeline
              </h3>
              <button onClick={loadPreprocessingStatus} className="btn btn-sm btn-secondary">
                <RefreshCw className="h-3.5 w-3.5" /> Refresh
              </button>
            </div>
            {preprocessingLoading ? (
              <div className="flex justify-center py-8"><Loader className="h-6 w-6 animate-spin text-primary-600" /></div>
            ) : preprocessingStatus ? (
              <div className="space-y-6">
                {/* Pipeline Status */}
                <div className="flex items-center gap-3">
                  <span className={`badge ${preprocessingStatus.pipeline_status === 'available' ? 'badge-success' : 'badge-danger'}`}>
                    Pipeline {preprocessingStatus.pipeline_status || 'unknown'}
                  </span>
                </div>

                {/* Dependencies */}
                {preprocessingStatus.dependencies && (
                  <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                    <p className="text-xs font-semibold text-surface-700 dark:text-surface-300 mb-3">Library Dependencies</p>
                    <div className="flex flex-wrap gap-3">
                      {Object.entries(preprocessingStatus.dependencies).map(([lib, available]) => (
                        <div key={lib} className="flex items-center gap-1.5">
                          {available
                            ? <CheckCircle className="h-4 w-4 text-emerald-500" />
                            : <XCircle className="h-4 w-4 text-red-500" />
                          }
                          <span className="text-sm font-medium text-surface-900 dark:text-white capitalize">{lib}</span>
                          <span className={`text-[10px] ${available ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'}`}>
                            {available ? 'installed' : 'missing'}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Pipeline Components */}
                {preprocessingStatus.components && (
                  <div>
                    <p className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Pipeline Components ({Object.keys(preprocessingStatus.components).length})</p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {Object.entries(preprocessingStatus.components).map(([key, comp]) => (
                        <div key={key} className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-semibold text-surface-900 dark:text-white capitalize">
                              {key.replace(/_/g, ' ')}
                            </span>
                            <span className={`badge ${comp.status === 'available' ? 'badge-success' : comp.status === 'limited' ? 'badge-warning' : 'badge-danger'}`}>
                              {comp.status}
                            </span>
                          </div>
                          {comp.description && (
                            <p className="text-xs text-surface-500 mb-2">{comp.description}</p>
                          )}
                          {/* Show supported methods/strategies */}
                          {(comp.strategies || comp.methods || comp.vectorization_methods || comp.scaling_methods || comp.encoding_methods || comp.supported_types) && (
                            <div className="flex flex-wrap gap-1">
                              {(comp.strategies || comp.methods || comp.vectorization_methods || comp.scaling_methods || comp.encoding_methods || comp.supported_types || []).map((m, i) => (
                                <span key={i} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 text-primary-700 dark:text-primary-400 capitalize">
                                  {m.replace(/_/g, ' ')}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-sm text-surface-500 text-center py-8">Click refresh to load preprocessing pipeline status</p>
            )}
          </div>
        )}
      </div>
    </AdminLayout>
  )
}

export default AdminAIEngines
