import { useState, useEffect } from 'react'
import TenantLayout from '../../components/layouts/TenantLayout'
import { renterAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  History as HistoryIcon, Calendar, TrendingUp, Eye, Download,
  ThumbsUp, ThumbsDown, HelpCircle, CheckCircle, Loader, FileText, X, AlertTriangle,
  Lightbulb, Brain
} from 'lucide-react'

const TenantHistory = () => {
  const [analyses, setAnalyses] = useState([])
  const [selectedAnalysis, setSelectedAnalysis] = useState(null)
  const [showDetail, setShowDetail] = useState(false)
  const [loading, setLoading] = useState(true)
  const [feedbackSubmitted, setFeedbackSubmitted] = useState({})
  const [exportingReport, setExportingReport] = useState(false)
  const [xaiResult, setXaiResult] = useState(null)
  const [xaiLoading, setXaiLoading] = useState(false)
  const [showXai, setShowXai] = useState(false)

  useEffect(() => { fetchHistory() }, [])

  const fetchHistory = async () => {
    try {
      const response = await renterAPI.getHistory()
      setAnalyses(response.data)
    } catch (error) {
      toast.error('Failed to fetch history')
    } finally {
      setLoading(false)
    }
  }

  const handleViewDetail = async (analysis) => {
    try {
      const response = await renterAPI.getAnalysisDetail(analysis.id)
      setSelectedAnalysis(response.data)
      setShowDetail(true)
    } catch (error) {
      toast.error('Failed to load details')
    }
  }

  const handleFeedback = async (analysisId, feedbackType) => {
    if (feedbackSubmitted[analysisId]) return
    try {
      await renterAPI.submitFeedback({ analysis_id: analysisId, feedback_type: feedbackType, comments: null })
      setFeedbackSubmitted({ ...feedbackSubmitted, [analysisId]: true })
      toast.success('Thank you for your feedback!')
    } catch (error) {
      if (error.response?.data?.detail?.includes('already submitted')) {
        setFeedbackSubmitted({ ...feedbackSubmitted, [analysisId]: true })
        toast.error('Feedback already submitted for this analysis')
      } else {
        toast.error('Failed to submit feedback')
      }
    }
  }

  const handleExportReport = async (analysisId, format) => {
    setExportingReport(true)
    try {
      const response = await renterAPI.exportReport({ analysis_id: analysisId, format })
      let blob
      if (format === 'pdf') {
        const binaryString = atob(response.data.content)
        const bytes = new Uint8Array(binaryString.length)
        for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i)
        blob = new Blob([bytes], { type: 'application/pdf' })
      } else {
        blob = new Blob([response.data.content], { type: 'text/html' })
      }
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = response.data.filename
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      toast.success('Report downloaded!')
    } catch (error) {
      toast.error('Failed to export report')
    } finally {
      setExportingReport(false)
    }
  }

  const handleExplain = async (analysisId) => {
    setXaiLoading(true)
    setXaiResult(null)
    setShowXai(true)
    try {
      const response = await renterAPI.getExplanation(analysisId)
      setXaiResult(response.data)
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to generate explanation')
      setShowXai(false)
    } finally {
      setXaiLoading(false)
    }
  }

  const getRiskBadge = (level) => {
    const map = {
      very_low: 'badge-success', low: 'badge-success', medium: 'badge-warning',
      high: 'badge-danger', very_high: 'badge-danger',
    }
    return map[level] || 'badge-info'
  }

  const getRiskColor = (level) => {
    const map = {
      very_low: 'text-emerald-600', low: 'text-emerald-500', medium: 'text-amber-600',
      high: 'text-accent-red', very_high: 'text-red-700',
    }
    return map[level] || 'text-surface-500'
  }

  const stats = {
    total: analyses.length,
    high: analyses.filter(a => a.risk_level === 'high' || a.risk_level === 'very_high').length,
    low: analyses.filter(a => a.risk_level === 'low' || a.risk_level === 'very_low').length,
  }

  return (
    <TenantLayout title="Analysis History" subtitle="Review past fraud risk assessments">
      <div className="space-y-6">
        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { icon: HistoryIcon, label: 'Total Analyses', value: stats.total, color: 'text-primary-600', bg: 'bg-primary-50 dark:bg-primary-900/20' },
            { icon: AlertTriangle, label: 'High Risk Detected', value: stats.high, color: 'text-accent-red', bg: 'bg-red-50 dark:bg-red-900/20' },
            { icon: CheckCircle, label: 'Low Risk', value: stats.low, color: 'text-emerald-600', bg: 'bg-emerald-50 dark:bg-emerald-900/20' },
          ].map((stat, i) => (
            <div key={i} className={`card p-5 ${stat.bg}`}>
              <stat.icon className={`h-8 w-8 ${stat.color} mb-2`} />
              <p className={`text-3xl font-display font-bold ${stat.color}`}>{stat.value}</p>
              <p className="text-sm text-surface-600 dark:text-surface-400 font-medium">{stat.label}</p>
            </div>
          ))}
        </div>

        {/* History List */}
        {loading ? (
          <div className="flex justify-center py-16">
            <Loader className="h-8 w-8 animate-spin text-primary-600" />
          </div>
        ) : analyses.length > 0 ? (
          <div className="space-y-3">
            {analyses.map((analysis) => (
              <div key={analysis.id} className="card p-5 hover:shadow-elevated transition-all duration-200 group">
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-2">
                      <span className={getRiskBadge(analysis.risk_level)}>
                        {(analysis.risk_level || '').replace('_', ' ')}
                      </span>
                      <span className="flex items-center text-xs text-surface-500 dark:text-surface-400">
                        <Calendar className="h-3.5 w-3.5 mr-1" />
                        {new Date(analysis.created_at).toLocaleString()}
                      </span>
                    </div>
                    <p className="text-sm text-surface-700 dark:text-surface-300 line-clamp-2">{analysis.listing_text}</p>
                  </div>
                  <div className="ml-6 text-right flex-shrink-0">
                    <p className="text-xs text-surface-500 dark:text-surface-400 mb-1">Risk Score</p>
                    <p className={`text-3xl font-display font-bold ${getRiskColor(analysis.risk_level)}`}>
                      {(analysis.risk_score * 100).toFixed(0)}%
                    </p>
                    <button
                      onClick={() => handleViewDetail(analysis)}
                      className="mt-2 btn btn-sm btn-secondary opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <Eye className="h-3.5 w-3.5" /> Details
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="card text-center py-16">
            <HistoryIcon className="h-16 w-16 text-surface-300 dark:text-surface-600 mx-auto mb-4" />
            <h3 className="text-lg font-display font-semibold text-surface-600 dark:text-surface-300 mb-2">No analyses yet</h3>
            <p className="text-surface-500 dark:text-surface-400 mb-6">Start analyzing rental listings to build your history</p>
            <a href="/tenant/analyze" className="btn btn-primary">Analyze Your First Listing</a>
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {showDetail && selectedAnalysis && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-surface-800 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6 shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-display font-bold text-surface-900 dark:text-white">Analysis Details</h3>
              <button onClick={() => setShowDetail(false)} className="btn btn-ghost btn-sm"><X className="h-5 w-5" /></button>
            </div>

            {/* Risk Score Header */}
            <div className="p-6 rounded-2xl mb-6 bg-gradient-to-r from-primary-600 to-primary-800 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white/70 text-sm mb-1">Risk Level</p>
                  <p className="text-2xl font-display font-bold capitalize">{(selectedAnalysis.risk_level || '').replace('_', ' ')}</p>
                </div>
                <div className="text-right">
                  <p className="text-white/70 text-sm mb-1">Risk Score</p>
                  <p className="text-4xl font-display font-bold">{(selectedAnalysis.risk_score * 100).toFixed(0)}%</p>
                </div>
              </div>
            </div>

            {/* Listing Text */}
            <div className="mb-6">
              <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-2">Listing Description</h4>
              <div className="bg-surface-50 dark:bg-surface-700 rounded-xl p-4 text-sm text-surface-700 dark:text-surface-300">
                {selectedAnalysis.listing_text}
              </div>
            </div>

            {/* Risk Story */}
            {selectedAnalysis.risk_story && (
              <div className="mb-6">
                <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-2">Risk Analysis</h4>
                <div className="bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 rounded-xl p-4 text-sm text-surface-700 dark:text-surface-300 whitespace-pre-line">
                  {selectedAnalysis.risk_story}
                </div>
              </div>
            )}

            {/* Risk Indicators */}
            {selectedAnalysis.risk_indicators?.length > 0 && (
              <div className="mb-6">
                <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Risk Indicators</h4>
                <div className="space-y-2">
                  {selectedAnalysis.risk_indicators.map((ind, i) => (
                    <div key={i} className="p-3 rounded-xl bg-surface-50 dark:bg-surface-700 flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className={`badge ${ind.severity >= 4 ? 'badge-danger' : ind.severity >= 2 ? 'badge-warning' : 'badge-info'}`}>
                            {ind.severity >= 4 ? 'High' : ind.severity >= 2 ? 'Medium' : 'Low'}
                          </span>
                          <span className="badge badge-info capitalize">{(ind.category_display || ind.category || '').replace(/_/g, ' ')}</span>
                        </div>
                        <p className="text-sm text-surface-700 dark:text-surface-300">{ind.description_friendly || ind.description}</p>
                        {(ind.evidence || ind.examples)?.length > 0 && (
                          <ul className="mt-1 text-xs text-surface-500 dark:text-surface-400 list-disc list-inside">
                            {(ind.evidence || ind.examples).map((ex, j) => <li key={j}>{ex}</li>)}
                          </ul>
                        )}
                      </div>
                      <span className="text-primary-600 font-bold">+{((ind.impact_score || ind.impact || 0) * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Feedback */}
            <div className="border-t border-surface-200 dark:border-surface-700 pt-4">
              <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Was this analysis helpful?</h4>
              {feedbackSubmitted[selectedAnalysis.id] ? (
                <div className="bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 rounded-xl p-4 text-center">
                  <CheckCircle className="h-5 w-5 text-emerald-600 mx-auto mb-1" />
                  <p className="text-sm text-emerald-800 dark:text-emerald-300 font-medium">Thank you for your feedback!</p>
                </div>
              ) : (
                <div className="flex gap-2">
                  <button onClick={() => handleFeedback(selectedAnalysis.id, 'safe')} className="btn btn-sm flex-1 bg-emerald-100 text-emerald-700 hover:bg-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-400">
                    <ThumbsUp className="h-4 w-4" /> Safe
                  </button>
                  <button onClick={() => handleFeedback(selectedAnalysis.id, 'fraud')} className="btn btn-sm flex-1 bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400">
                    <ThumbsDown className="h-4 w-4" /> Fraud
                  </button>
                  <button onClick={() => handleFeedback(selectedAnalysis.id, 'unsure')} className="btn btn-sm flex-1 btn-secondary">
                    <HelpCircle className="h-4 w-4" /> Unsure
                  </button>
                </div>
              )}
            </div>

            {/* Export */}
            <div className="border-t border-surface-200 dark:border-surface-700 pt-4 mt-4">
              <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3 flex items-center gap-2">
                <FileText className="h-4 w-4" /> Export Report
              </h4>
              <div className="flex gap-2">
                <button onClick={() => handleExportReport(selectedAnalysis.id, 'html')} disabled={exportingReport} className="btn btn-sm btn-primary flex-1">
                  {exportingReport ? <Loader className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />} HTML
                </button>
                <button onClick={() => handleExportReport(selectedAnalysis.id, 'pdf')} disabled={exportingReport} className="btn btn-sm btn-secondary flex-1">
                  {exportingReport ? <Loader className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />} PDF
                </button>
              </div>
            </div>

            {/* XAI Explain Button */}
            <div className="border-t border-surface-200 dark:border-surface-700 pt-4 mt-4">
              <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3 flex items-center gap-2">
                <Brain className="h-4 w-4" /> Explainable AI
              </h4>
              <p className="text-xs text-surface-500 mb-3">Understand why AIs reasoning led to this risk score</p>
              <button onClick={() => { setShowDetail(false); handleExplain(selectedAnalysis.id) }}
                className="btn btn-sm w-full bg-purple-100 text-purple-700 hover:bg-purple-200 dark:bg-purple-900/30 dark:text-purple-300 dark:hover:bg-purple-900/50">
                <Lightbulb className="h-4 w-4" /> Generate XAI Explanation
              </button>
            </div>
          </div>
        </div>
      )}
      {/* XAI Explanation Modal */}
      {showXai && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-surface-800 rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto p-6 shadow-2xl">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-display font-bold text-surface-900 dark:text-white flex items-center gap-2">
                <Brain className="h-5 w-5 text-purple-600" /> AI Explanation
              </h3>
              <button onClick={() => setShowXai(false)} className="btn btn-ghost btn-sm"><X className="h-5 w-5" /></button>
            </div>

            {xaiLoading ? (
              <div className="flex flex-col items-center justify-center py-16">
                <Loader className="h-8 w-8 animate-spin text-purple-600 mb-3" />
                <p className="text-surface-500 text-sm">Generating explanation...</p>
              </div>
            ) : xaiResult ? (
              <div className="space-y-6">
                {/* Classification */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                  <div>
                    <p className="text-sm text-surface-500">Classification</p>
                    <p className="text-lg font-bold text-surface-900 dark:text-white">
                      {(xaiResult.is_fraud ?? (xaiResult.risk_score > 0.5)) ? 'Fraud Detected' : 'Likely Safe'}
                    </p>
                  </div>
                  {(xaiResult.confidence != null || xaiResult.risk_score != null) && (
                    <div className="text-right">
                      <p className="text-sm text-surface-500">Confidence</p>
                      <p className="text-lg font-bold text-surface-900 dark:text-white">
                        {((xaiResult.confidence ?? xaiResult.risk_score) * 100).toFixed(1)}%
                      </p>
                    </div>
                  )}
                </div>

                {/* Methodology badge */}
                {xaiResult.methodology && (
                  <div className="flex items-center gap-2 text-xs text-purple-600 dark:text-purple-400">
                    <Brain className="h-3.5 w-3.5" />
                    <span>{xaiResult.methodology}</span>
                  </div>
                )}

                {/* Top Risk Contributors (from top_contributors or all_contributions) */}
                {(() => {
                  const contributors = xaiResult.top_contributors || xaiResult.all_contributions || []
                  return contributors.length > 0 ? (
                    <div>
                      <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Feature Contributions</h4>
                      <div className="space-y-2">
                        {contributors.slice(0, 10).map((item, i) => {
                          const contribution = item.contribution ?? 0
                          const isNeg = item.direction === 'decreases_risk' || contribution < 0
                          return (
                            <div key={i} className={`p-3 rounded-xl ${isNeg ? 'bg-emerald-50 dark:bg-emerald-900/10' : 'bg-red-50 dark:bg-red-900/10'}`}>
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-sm font-medium text-surface-900 dark:text-white">{item.feature || `Feature ${i + 1}`}</span>
                                <span className={`text-xs font-bold ${isNeg ? 'text-emerald-600' : 'text-red-600'}`}>
                                  {isNeg ? '↓' : '↑'} {item.contribution_percent || `${Math.abs(contribution * 100).toFixed(1)}%`}
                                </span>
                              </div>
                              <p className="text-xs text-surface-600 dark:text-surface-400">{item.explanation}</p>
                              {item.value && item.value !== 'Detected' && (
                                <p className="text-xs text-surface-400 mt-1">Evidence: {item.value}</p>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  ) : null
                })()}

                {/* Word Importance — from real_xai token attributions */}
                {(() => {
                  const tokens = xaiResult.word_importance
                    || xaiResult.real_xai?.token_level_explanation?.all_tokens
                    || []
                  return tokens.length > 0 ? (
                    <div>
                      <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Word Importance</h4>
                      <div className="flex flex-wrap gap-1.5">
                        {tokens.slice(0, 40).map((item, i) => {
                          const score = typeof item === 'object' ? (item.attribution || item.score || item.importance || 0) : 0
                          const word = typeof item === 'object' ? (item.token || item.word) : item
                          const intensity = Math.min(Math.abs(score) * 3, 1)
                          const bg = score > 0 ? `rgba(239,68,68,${intensity})` : score < 0 ? `rgba(16,185,129,${intensity})` : 'transparent'
                          return (
                            <span key={i} className="px-1.5 py-0.5 rounded text-sm font-mono" style={{ backgroundColor: bg }}>{word}</span>
                          )
                        })}
                      </div>
                      <p className="text-xs text-surface-400 mt-2">
                        <span className="inline-block w-3 h-3 rounded bg-red-400 mr-1 align-middle" /> fraud
                        <span className="inline-block w-3 h-3 rounded bg-emerald-400 mx-1 ml-3 align-middle" /> safe
                      </p>
                    </div>
                  ) : null
                })()}

                {/* Attention Weights — from real_xai top_fraud_indicators */}
                {(() => {
                  const fraudTokens = xaiResult.attention_weights
                    || xaiResult.real_xai?.token_level_explanation?.top_fraud_indicators
                    || []
                  return fraudTokens.length > 0 ? (
                    <div>
                      <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Top Fraud Indicators (Token-Level)</h4>
                      <div className="space-y-1">
                        {fraudTokens.slice(0, 15).map((item, i) => {
                          const weight = typeof item === 'object' ? (item.attribution || item.weight || item.attention || 0) : item
                          const token = typeof item === 'object' ? (item.token || item.word || `Token ${i}`) : `Token ${i}`
                          return (
                            <div key={i} className="flex items-center gap-2 text-xs">
                              <span className="w-24 font-mono text-surface-600 dark:text-surface-400 truncate">{token}</span>
                              <div className="flex-1 bg-surface-100 dark:bg-surface-800 rounded-full h-2">
                                <div className="h-2 rounded-full bg-purple-500" style={{ width: `${Math.min(Math.abs(weight) * 100, 100)}%` }} />
                              </div>
                              <span className="text-surface-400 w-12 text-right">{(Math.abs(weight) * 100).toFixed(0)}%</span>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  ) : null
                })()}

                {/* Counterfactual / What-If Analysis */}
                {(() => {
                  const whatIf = xaiResult.counterfactuals || xaiResult.what_if_analysis || []
                  return whatIf.length > 0 ? (
                    <div>
                      <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">What-If Analysis</h4>
                      <div className="space-y-2">
                        {whatIf.map((cf, i) => (
                          <div key={i} className="p-3 rounded-xl bg-blue-50 dark:bg-blue-900/10 text-sm text-surface-700 dark:text-surface-300">
                            <p>{typeof cf === 'string' ? cf : cf.explanation || cf.description || JSON.stringify(cf)}</p>
                            {cf.original_risk && cf.new_risk && (
                              <div className="flex items-center gap-3 mt-2 text-xs">
                                <span className="text-red-600 font-medium">Risk: {cf.original_risk}</span>
                                <span>→</span>
                                <span className="text-emerald-600 font-medium">{cf.new_risk}</span>
                                <span className="text-blue-600 font-medium">(−{cf.risk_reduction})</span>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null
                })()}

                {/* Reasoning Chain (from real_xai) */}
                {xaiResult.real_xai?.reasoning_chain?.length > 0 && (
                  <div>
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">AI Reasoning Chain</h4>
                    <div className="space-y-2">
                      {xaiResult.real_xai.reasoning_chain.map((step, i) => (
                        <div key={i} className="p-3 rounded-xl bg-purple-50 dark:bg-purple-900/10">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="w-6 h-6 bg-purple-600 text-white text-xs rounded-full flex items-center justify-center font-bold">{step.step}</span>
                            <span className="text-sm font-medium text-surface-900 dark:text-white">{step.description}</span>
                          </div>
                          {step.evidence?.length > 0 && (
                            <ul className="ml-8 text-xs text-surface-500 list-disc list-inside mt-1">
                              {step.evidence.map((e, j) => <li key={j}>{e}</li>)}
                            </ul>
                          )}
                          <div className="ml-8 mt-1 text-xs text-purple-500">
                            {step.method} • Confidence: {((step.confidence || 0) * 100).toFixed(0)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Rule-Based Explanations (from top_contributors explanations) */}
                {(() => {
                  const explanations = xaiResult.explanations
                    || (xaiResult.top_contributors || []).filter(c => c.explanation).map(c => c.explanation)
                    || []
                  // Only show if not already shown in Feature Contributions section
                  return explanations.length > 0 && !xaiResult.top_contributors ? (
                    <div>
                      <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Rule-Based Explanations</h4>
                      <div className="space-y-2">
                        {explanations.map((exp, i) => (
                          <div key={i} className="flex items-start gap-2 text-sm text-surface-700 dark:text-surface-300">
                            <Lightbulb className="h-4 w-4 text-amber-500 mt-0.5 flex-shrink-0" />
                            {typeof exp === 'string' ? exp : exp.text || exp.explanation || JSON.stringify(exp)}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null
                })()}

                {xaiResult.summary && (
                  <div className="p-4 rounded-xl bg-purple-50 dark:bg-purple-900/10">
                    <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-2">Summary</h4>
                    <p className="text-sm text-surface-600 dark:text-surface-400 whitespace-pre-line">{xaiResult.summary}</p>
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </div>
      )}
    </TenantLayout>
  )
}

export default TenantHistory
