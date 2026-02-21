import { useState, useEffect } from 'react'
import AdminLayout from '../../components/layouts/AdminLayout'
import { adminAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  Brain, Loader, RefreshCw, CheckCircle, XCircle, Eye, X,
  BarChart3, Zap, Activity, FileText, HardDrive, Shield,
  TrendingDown, TrendingUp, AlertTriangle, Info, ChevronDown,
  ChevronRight, Layers, Target, Box, Clock, Database
} from 'lucide-react'

const AdminTrainedModels = () => {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)

  // Report modal
  const [report, setReport] = useState(null)
  const [reportLoading, setReportLoading] = useState(false)

  // Collapsible sections in report
  const [expandedSections, setExpandedSections] = useState({})

  useEffect(() => { fetchTrainedModels() }, [])

  const fetchTrainedModels = async () => {
    setLoading(true)
    try {
      const res = await adminAPI.getTrainedModels()
      setModels(res.data.models || [])
    } catch { toast.error('Failed to discover trained models') }
    finally { setLoading(false) }
  }

  const handleViewReport = async (modelId) => {
    setReportLoading(true)
    setReport(null)
    setExpandedSections({
      overview: true,
      architecture: true,
      training: true,
      performance: true,
      confusion: false,
      history: false,
      features: false,
      scores: false,
      business: false,
      readiness: true,
      files: false,
      recommendations: true,
    })
    try {
      const res = await adminAPI.getTrainedModelReport(modelId)
      setReport(res.data)
    } catch (e) { toast.error(e.response?.data?.detail || 'Failed to load report') }
    finally { setReportLoading(false) }
  }

  const toggleSection = (key) => {
    setExpandedSections(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const getModelTypeColor = (type) => {
    if (type === 'DistilBERT') return 'bg-violet-100 dark:bg-violet-900/30 text-violet-600 dark:text-violet-400'
    if (type === 'IsolationForest') return 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400'
    return 'bg-surface-100 dark:bg-surface-700 text-surface-600'
  }

  const getStatusBadge = (status) => {
    if (status === 'active') return 'badge badge-success'
    if (status === 'archived') return 'badge badge-secondary'
    return 'badge'
  }

  const formatPercent = (v) => typeof v === 'number' ? `${(v * 100).toFixed(2)}%` : v
  const formatNumber = (v) => typeof v === 'number' ? v.toLocaleString() : v

  // ── Report Section Component ──
  const ReportSection = ({ id, icon: Icon, title, children, defaultOpen }) => (
    <div className="border border-surface-200 dark:border-surface-700 rounded-xl overflow-hidden">
      <button
        onClick={() => toggleSection(id)}
        className="w-full flex items-center justify-between px-4 py-3 bg-surface-50 dark:bg-surface-800 hover:bg-surface-100 dark:hover:bg-surface-750 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-primary-600" />
          <span className="text-sm font-semibold text-surface-900 dark:text-white">{title}</span>
        </div>
        {expandedSections[id] ? <ChevronDown className="h-4 w-4 text-surface-400" /> : <ChevronRight className="h-4 w-4 text-surface-400" />}
      </button>
      {expandedSections[id] && (
        <div className="px-4 py-4 space-y-3">
          {children}
        </div>
      )}
    </div>
  )

  // ── Metric Card ──
  const MetricCard = ({ label, value, sub, highlight }) => (
    <div className={`p-3 rounded-xl ${highlight ? 'bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800' : 'bg-surface-50 dark:bg-surface-700'}`}>
      <p className="text-xs text-surface-500 dark:text-surface-400 mb-1">{label}</p>
      <p className={`text-sm font-bold ${highlight ? 'text-primary-700 dark:text-primary-300' : 'text-surface-900 dark:text-white'}`}>{value}</p>
      {sub && <p className="text-[10px] text-surface-400 mt-0.5">{sub}</p>}
    </div>
  )

  return (
    <AdminLayout title="Trained Models" subtitle="Discover on-disk trained models with comprehensive analysis reports">
      <div className="space-y-6">
        {/* Header actions */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-primary-600 rounded-xl flex items-center justify-center">
              <HardDrive className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="text-base font-display font-bold text-surface-900 dark:text-white">On-Disk Models</h3>
              <p className="text-xs text-surface-500">{models.length} model{models.length !== 1 ? 's' : ''} discovered</p>
            </div>
          </div>
          <button onClick={fetchTrainedModels} className="btn btn-sm btn-secondary">
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </button>
        </div>

        {/* Models Grid */}
        {loading ? (
          <div className="flex justify-center py-16"><Loader className="h-8 w-8 animate-spin text-primary-600" /></div>
        ) : models.length > 0 ? (
          <div className="grid gap-4">
            {models.map((model) => {
              const metrics = model.metrics || {}
              const isBert = model.model_type === 'DistilBERT'

              return (
                <div key={model.id} className="card p-5">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-start gap-4 flex-1 min-w-0">
                      {/* Type badge icon */}
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${getModelTypeColor(model.model_type)}`}>
                        {isBert ? <Brain className="h-6 w-6" /> : <Layers className="h-6 w-6" />}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <p className="text-sm font-bold text-surface-900 dark:text-white truncate">{model.name}</p>
                          <span className={getStatusBadge(model.status)}>{model.status}</span>
                          {model.is_latest && <span className="badge badge-primary">Latest</span>}
                        </div>
                        <p className="text-xs text-surface-500 dark:text-surface-400 mt-0.5">
                          {model.model_type} • {model.algorithm}
                        </p>
                        {/* Quick metrics row */}
                        <div className="flex items-center gap-4 mt-2 flex-wrap">
                          {isBert && metrics.accuracy != null && (
                            <span className="text-xs font-mono text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                              <Target className="h-3 w-3" /> Acc: {formatPercent(metrics.accuracy)}
                            </span>
                          )}
                          {isBert && metrics.f1_score != null && (
                            <span className="text-xs font-mono text-blue-600 dark:text-blue-400 flex items-center gap-1">
                              <BarChart3 className="h-3 w-3" /> F1: {formatPercent(metrics.f1_score)}
                            </span>
                          )}
                          {isBert && metrics.roc_auc != null && (
                            <span className="text-xs font-mono text-purple-600 dark:text-purple-400 flex items-center gap-1">
                              <TrendingUp className="h-3 w-3" /> AUC: {formatPercent(metrics.roc_auc)}
                            </span>
                          )}
                          {!isBert && metrics.n_samples != null && (
                            <span className="text-xs font-mono text-surface-600 dark:text-surface-300 flex items-center gap-1">
                              <Database className="h-3 w-3" /> {formatNumber(metrics.n_samples)} samples
                            </span>
                          )}
                          {!isBert && metrics.n_anomalies != null && (
                            <span className="text-xs font-mono text-amber-600 dark:text-amber-400 flex items-center gap-1">
                              <AlertTriangle className="h-3 w-3" /> {formatNumber(metrics.n_anomalies)} anomalies ({(metrics.anomaly_rate * 100).toFixed(1)}%)
                            </span>
                          )}
                          {!isBert && metrics.n_features != null && (
                            <span className="text-xs font-mono text-surface-600 dark:text-surface-300 flex items-center gap-1">
                              <Layers className="h-3 w-3" /> {metrics.n_features} features
                            </span>
                          )}
                          {isBert && metrics.total_training_time != null && (
                            <span className="text-xs font-mono text-surface-500 flex items-center gap-1">
                              <Clock className="h-3 w-3" /> {(metrics.total_training_time / 60).toFixed(1)} min
                            </span>
                          )}
                          {!isBert && metrics.training_time_seconds != null && (
                            <span className="text-xs font-mono text-surface-500 flex items-center gap-1">
                              <Clock className="h-3 w-3" /> {metrics.training_time_seconds.toFixed(1)}s
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    {/* Action */}
                    <button onClick={() => handleViewReport(model.id)} className="btn btn-sm btn-primary flex-shrink-0">
                      <Eye className="h-3.5 w-3.5" /> View Report
                    </button>
                  </div>

                  {/* Files summary */}
                  {model.files && (
                    <div className="mt-3 pt-3 border-t border-surface-100 dark:border-surface-700">
                      <div className="flex items-center gap-3 flex-wrap">
                        {Object.entries(model.files).map(([fname, finfo]) => (
                          <span key={fname} className="text-[10px] font-mono px-2 py-1 rounded-lg bg-surface-100 dark:bg-surface-700 text-surface-500">
                            {fname} ({finfo.size_mb} MB)
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        ) : (
          <div className="card text-center py-16">
            <HardDrive className="h-16 w-16 text-surface-300 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-surface-600 mb-2">No trained models found</h3>
            <p className="text-surface-500">Train models using the ML Models page to see them here</p>
          </div>
        )}
      </div>

      {/* ══════════════════════════════════════════════════════════════════ */}
      {/*  COMPREHENSIVE ANALYSIS REPORT MODAL                            */}
      {/* ══════════════════════════════════════════════════════════════════ */}
      {(report || reportLoading) && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-surface-800 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6 shadow-2xl">
            {/* Header */}
            <div className="flex justify-between items-start mb-6">
              <div>
                <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-primary-600" /> Comprehensive Model Analysis Report
                </h3>
                {report && (
                  <p className="text-xs text-surface-500 mt-1">
                    {report.model_type} • {report.model_name} • Generated {new Date(report.generated_at).toLocaleString()}
                  </p>
                )}
              </div>
              <button onClick={() => setReport(null)} className="btn btn-ghost btn-sm"><X className="h-5 w-5" /></button>
            </div>

            {reportLoading ? (
              <div className="flex flex-col items-center justify-center py-16">
                <Loader className="h-8 w-8 animate-spin text-primary-600 mb-3" />
                <p className="text-sm text-surface-500">Generating comprehensive report...</p>
              </div>
            ) : report ? (
              <div className="space-y-4">

                {/* ── Overview ── */}
                {report.overview && (
                  <ReportSection id="overview" icon={Info} title="Model Overview">
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {Object.entries(report.overview).map(([key, value]) => (
                        <MetricCard
                          key={key}
                          label={key.replace(/_/g, ' ')}
                          value={typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value)}
                        />
                      ))}
                    </div>
                  </ReportSection>
                )}

                {/* ── Architecture ── */}
                {report.architecture && (
                  <ReportSection id="architecture" icon={Layers} title="Architecture Details">
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {Object.entries(report.architecture).filter(([k, v]) => !Array.isArray(v) || v.length <= 5).map(([key, value]) => (
                        <MetricCard
                          key={key}
                          label={key.replace(/_/g, ' ')}
                          value={Array.isArray(value) ? value.length + ' items' : typeof value === 'number' ? formatNumber(value) : String(value)}
                        />
                      ))}
                    </div>
                    {/* Feature names list for IsolationForest */}
                    {report.architecture.feature_names && report.architecture.feature_names.length > 0 && (
                      <div className="mt-3">
                        <p className="text-xs font-semibold text-surface-600 dark:text-surface-300 mb-2">Feature Names ({report.architecture.feature_names.length})</p>
                        <div className="flex flex-wrap gap-1.5">
                          {report.architecture.feature_names.map((f, i) => (
                            <span key={i} className="text-[10px] font-mono px-2 py-1 rounded-lg bg-surface-100 dark:bg-surface-700 text-surface-600 dark:text-surface-300">
                              {f}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </ReportSection>
                )}

                {/* ── Training Configuration ── */}
                {report.training_config && (
                  <ReportSection id="training" icon={Zap} title="Training Configuration">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {Object.entries(report.training_config).map(([key, value]) => (
                        <MetricCard
                          key={key}
                          label={key.replace(/_/g, ' ')}
                          value={typeof value === 'number' ? (key.includes('rate') || key.includes('split') || key.includes('ratio') || key.includes('decay') ? value.toExponential ? (value < 0.01 ? value.toExponential(1) : value) : value : formatNumber(value)) : String(value)}
                        />
                      ))}
                    </div>
                  </ReportSection>
                )}

                {/* ── Performance Metrics ── */}
                {report.performance_metrics && (
                  <ReportSection id="performance" icon={Target} title="Performance Metrics">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {Object.entries(report.performance_metrics).map(([key, value]) => {
                        const isPercent = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'specificity', 'false_positive_rate', 'false_negative_rate', 'anomaly_rate'].includes(key)
                        const isHighlight = ['accuracy', 'f1_score', 'roc_auc', 'anomaly_rate'].includes(key)
                        return (
                          <MetricCard
                            key={key}
                            label={key.replace(/_/g, ' ')}
                            value={isPercent && typeof value === 'number' ? formatPercent(value) : typeof value === 'number' ? (Math.abs(value) < 0.001 && value !== 0 ? value.toExponential(3) : value.toLocaleString(undefined, { maximumFractionDigits: 4 })) : String(value)}
                            highlight={isHighlight}
                          />
                        )
                      })}
                    </div>
                  </ReportSection>
                )}

                {/* ── Confusion Matrix (BERT only) ── */}
                {report.confusion_matrix && (
                  <ReportSection id="confusion" icon={Box} title="Confusion Matrix Analysis">
                    <div className="grid grid-cols-2 gap-4 max-w-md">
                      <div className="p-4 rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 text-center">
                        <p className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 uppercase">True Negatives</p>
                        <p className="text-2xl font-bold text-emerald-700 dark:text-emerald-300">{formatNumber(report.confusion_matrix.true_negatives)}</p>
                        <p className="text-[10px] text-emerald-500">Correctly identified legit</p>
                      </div>
                      <div className="p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-center">
                        <p className="text-[10px] font-bold text-red-600 dark:text-red-400 uppercase">False Positives</p>
                        <p className="text-2xl font-bold text-red-700 dark:text-red-300">{formatNumber(report.confusion_matrix.false_positives)}</p>
                        <p className="text-[10px] text-red-500">False alarms</p>
                      </div>
                      <div className="p-4 rounded-xl bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 text-center">
                        <p className="text-[10px] font-bold text-orange-600 dark:text-orange-400 uppercase">False Negatives</p>
                        <p className="text-2xl font-bold text-orange-700 dark:text-orange-300">{formatNumber(report.confusion_matrix.false_negatives)}</p>
                        <p className="text-[10px] text-orange-500">Missed fraud</p>
                      </div>
                      <div className="p-4 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 text-center">
                        <p className="text-[10px] font-bold text-blue-600 dark:text-blue-400 uppercase">True Positives</p>
                        <p className="text-2xl font-bold text-blue-700 dark:text-blue-300">{formatNumber(report.confusion_matrix.true_positives)}</p>
                        <p className="text-[10px] text-blue-500">Correctly caught fraud</p>
                      </div>
                    </div>
                  </ReportSection>
                )}

                {/* ── Training History (BERT only) ── */}
                {report.training_history && (
                  <ReportSection id="history" icon={Activity} title="Training History">
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
                      <MetricCard label="Best Epoch" value={report.training_history.best_epoch} highlight />
                      <MetricCard label="Training Time" value={`${report.training_history.total_training_time_minutes} min`} />
                      <MetricCard
                        label="Loss Reduction"
                        value={`${report.training_history.convergence_analysis?.loss_reduction_pct || 0}%`}
                        sub="From first to last epoch"
                      />
                    </div>
                    {/* Loss per epoch table */}
                    {report.training_history.training_loss_per_epoch && report.training_history.training_loss_per_epoch.length > 0 && (
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-surface-200 dark:border-surface-700">
                              <th className="text-left py-2 px-3 text-surface-500 font-semibold">Epoch</th>
                              <th className="text-right py-2 px-3 text-surface-500 font-semibold">Training Loss</th>
                              <th className="text-right py-2 px-3 text-surface-500 font-semibold">Validation Loss</th>
                              <th className="text-right py-2 px-3 text-surface-500 font-semibold">Δ Change</th>
                            </tr>
                          </thead>
                          <tbody>
                            {report.training_history.training_loss_per_epoch.map((tl, i) => {
                              const vl = report.training_history.validation_loss_per_epoch?.[i]
                              const prevTl = i > 0 ? report.training_history.training_loss_per_epoch[i - 1] : null
                              const change = prevTl ? ((tl - prevTl) / prevTl * 100) : null
                              return (
                                <tr key={i} className="border-b border-surface-100 dark:border-surface-800">
                                  <td className="py-2 px-3 font-medium text-surface-900 dark:text-white">
                                    {i + 1} {i + 1 === report.training_history.best_epoch && <span className="text-primary-500 ml-1">★</span>}
                                  </td>
                                  <td className="py-2 px-3 text-right font-mono text-surface-700 dark:text-surface-300">{tl.toExponential(4)}</td>
                                  <td className="py-2 px-3 text-right font-mono text-surface-700 dark:text-surface-300">{vl != null ? vl.toExponential(4) : '—'}</td>
                                  <td className={`py-2 px-3 text-right font-mono ${change != null && change < 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                                    {change != null ? `${change > 0 ? '+' : ''}${change.toFixed(1)}%` : '—'}
                                  </td>
                                </tr>
                              )
                            })}
                          </tbody>
                        </table>
                      </div>
                    )}
                    {/* Convergence info */}
                    {report.training_history.convergence_analysis && (
                      <div className="mt-3 p-3 rounded-xl bg-surface-50 dark:bg-surface-700">
                        <p className="text-xs font-semibold text-surface-600 dark:text-surface-300 mb-2">Convergence Analysis</p>
                        <div className="flex items-center gap-2">
                          {report.training_history.convergence_analysis.overfitting_detected ? (
                            <span className="text-xs text-amber-600 dark:text-amber-400 flex items-center gap-1">
                              <AlertTriangle className="h-3 w-3" /> Possible overfitting detected (validation loss increased in last epoch)
                            </span>
                          ) : (
                            <span className="text-xs text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                              <CheckCircle className="h-3 w-3" /> Model converged normally — no overfitting detected
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </ReportSection>
                )}

                {/* ── Feature Analysis (IsolationForest only) ── */}
                {report.feature_analysis && (
                  <ReportSection id="features" icon={Layers} title="Feature Analysis">
                    <MetricCard label="Total Features" value={report.feature_analysis.total_features} highlight />
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-3">
                      {Object.entries(report.feature_analysis.category_counts || {}).map(([cat, count]) => (
                        <MetricCard key={cat} label={cat.replace(/_/g, ' ')} value={count} />
                      ))}
                    </div>
                    {/* Feature lists per category */}
                    {report.feature_analysis.feature_categories && Object.entries(report.feature_analysis.feature_categories).map(([cat, features]) => (
                      features.length > 0 && (
                        <div key={cat} className="mt-3">
                          <p className="text-xs font-semibold text-surface-600 dark:text-surface-300 mb-1 capitalize">{cat.replace(/_/g, ' ')} ({features.length})</p>
                          <div className="flex flex-wrap gap-1">
                            {features.map((f, i) => (
                              <span key={i} className="text-[10px] font-mono px-2 py-0.5 rounded bg-surface-100 dark:bg-surface-700 text-surface-500">{f}</span>
                            ))}
                          </div>
                        </div>
                      )
                    ))}
                  </ReportSection>
                )}

                {/* ── Score Distribution (IsolationForest only) ── */}
                {report.score_distribution && (
                  <ReportSection id="scores" icon={TrendingDown} title="Anomaly Score Distribution">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <MetricCard label="Mean" value={report.score_distribution.mean?.toFixed(4)} />
                      <MetricCard label="Std Dev" value={report.score_distribution.std?.toFixed(4)} />
                      <MetricCard label="Min" value={report.score_distribution.min?.toFixed(4)} />
                      <MetricCard label="Max" value={report.score_distribution.max?.toFixed(4)} />
                    </div>
                    <div className="mt-3 p-3 rounded-xl bg-surface-50 dark:bg-surface-700">
                      <p className="text-xs text-surface-600 dark:text-surface-300">{report.score_distribution.interpretation}</p>
                    </div>
                  </ReportSection>
                )}

                {/* ── Business Impact ── */}
                {report.business_impact && (
                  <ReportSection id="business" icon={TrendingUp} title="Business Impact Analysis">
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {Object.entries(report.business_impact).map(([key, value]) => (
                        <MetricCard
                          key={key}
                          label={key.replace(/_/g, ' ')}
                          value={String(value)}
                          highlight={key === 'risk_assessment'}
                        />
                      ))}
                    </div>
                  </ReportSection>
                )}

                {/* ── Deployment Readiness ── */}
                {report.deployment_readiness && (
                  <ReportSection id="readiness" icon={Shield} title="Deployment Readiness">
                    {/* Score header */}
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className={`px-3 py-1.5 rounded-lg text-sm font-bold ${
                          report.deployment_readiness.verdict === 'READY' ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400' :
                          report.deployment_readiness.verdict === 'READY_WITH_WARNINGS' ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400' :
                          'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                        }`}>
                          {report.deployment_readiness.verdict}
                        </div>
                        <span className="text-sm text-surface-600 dark:text-surface-300">
                          {report.deployment_readiness.passed}/{report.deployment_readiness.total} checks passed ({report.deployment_readiness.score})
                        </span>
                      </div>
                    </div>
                    {/* Checks list */}
                    <div className="space-y-2">
                      {report.deployment_readiness.checks?.map((check, i) => (
                        <div key={i} className={`flex items-center gap-3 p-2.5 rounded-lg ${check.passed ? 'bg-emerald-50 dark:bg-emerald-900/10' : 'bg-red-50 dark:bg-red-900/10'}`}>
                          {check.passed ? (
                            <CheckCircle className="h-4 w-4 text-emerald-500 flex-shrink-0" />
                          ) : (
                            <XCircle className="h-4 w-4 text-red-500 flex-shrink-0" />
                          )}
                          <span className="text-xs text-surface-700 dark:text-surface-300 flex-1">{check.check}</span>
                          <span className={`text-[10px] font-semibold uppercase px-2 py-0.5 rounded ${
                            check.severity === 'critical' ? 'bg-red-100 dark:bg-red-900/30 text-red-600' :
                            check.severity === 'high' ? 'bg-orange-100 dark:bg-orange-900/30 text-orange-600' :
                            check.severity === 'medium' ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-600' :
                            'bg-surface-100 dark:bg-surface-700 text-surface-500'
                          }`}>
                            {check.severity}
                          </span>
                        </div>
                      ))}
                    </div>
                  </ReportSection>
                )}

                {/* ── File Inventory ── */}
                {report.files && (
                  <ReportSection id="files" icon={FileText} title="File Inventory">
                    <div className="flex items-center gap-2 mb-3">
                      <HardDrive className="h-4 w-4 text-surface-400" />
                      <span className="text-sm font-semibold text-surface-700 dark:text-surface-300">
                        Total Size: {report.files.total_size_mb} MB
                      </span>
                    </div>
                    <div className="space-y-1.5">
                      {Object.entries(report.files.inventory || {}).map(([fname, info]) => (
                        <div key={fname} className="flex items-center justify-between p-2.5 rounded-lg bg-surface-50 dark:bg-surface-700">
                          <span className="text-xs font-mono text-surface-700 dark:text-surface-300">{fname}</span>
                          <span className="text-xs text-surface-500">{info.size_mb} MB</span>
                        </div>
                      ))}
                    </div>
                  </ReportSection>
                )}

                {/* ── Recommendations ── */}
                {report.recommendations && report.recommendations.length > 0 && (
                  <ReportSection id="recommendations" icon={AlertTriangle} title="Recommendations">
                    <div className="space-y-2">
                      {report.recommendations.map((rec, i) => (
                        <div key={i} className={`flex items-start gap-3 p-3 rounded-lg ${
                          rec.type === 'success' ? 'bg-emerald-50 dark:bg-emerald-900/10 border border-emerald-200 dark:border-emerald-800' :
                          rec.type === 'warning' ? 'bg-amber-50 dark:bg-amber-900/10 border border-amber-200 dark:border-amber-800' :
                          'bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800'
                        }`}>
                          {rec.type === 'success' ? <CheckCircle className="h-4 w-4 text-emerald-500 flex-shrink-0 mt-0.5" /> :
                           rec.type === 'warning' ? <AlertTriangle className="h-4 w-4 text-amber-500 flex-shrink-0 mt-0.5" /> :
                           <Info className="h-4 w-4 text-blue-500 flex-shrink-0 mt-0.5" />}
                          <p className="text-xs text-surface-700 dark:text-surface-300">{rec.message}</p>
                        </div>
                      ))}
                    </div>
                  </ReportSection>
                )}

              </div>
            ) : null}
          </div>
        </div>
      )}
    </AdminLayout>
  )
}

export default AdminTrainedModels
