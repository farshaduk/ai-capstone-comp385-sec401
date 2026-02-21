import { useState, useEffect } from 'react'
import AdminLayout from '../../components/layouts/AdminLayout'
import { adminAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  Database, Loader, RefreshCw, Eye, X, Search,
  BarChart3, FileText, HardDrive, Shield, Info,
  ChevronDown, ChevronRight, Layers, Target,
  CheckCircle, AlertTriangle,
  Activity, Tag, GitBranch, Table, Hash
} from 'lucide-react'

const AdminDatasets = () => {
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterCategory, setFilterCategory] = useState('all')
  const [summary, setSummary] = useState({})

  // Report modal
  const [report, setReport] = useState(null)
  const [reportLoading, setReportLoading] = useState(false)

  // Collapsible sections in report
  const [expandedSections, setExpandedSections] = useState({})

  useEffect(() => { fetchDatasets() }, [])

  const fetchDatasets = async () => {
    setLoading(true)
    try {
      const res = await adminAPI.getDatasets()
      setDatasets(res.data.datasets || [])
      setSummary(res.data.summary || {})
    } catch { toast.error('Failed to discover system datasets') }
    finally { setLoading(false) }
  }

  const handleViewReport = async (datasetId) => {
    setReportLoading(true)
    setReport(null)
    setExpandedSections({
      overview: true,
      schema: true,
      quality: true,
      statistics: false,
      categorical: false,
      text: false,
      labels: true,
      preview: false,
      lineage: true,
      recommendations: true,
    })
    try {
      const res = await adminAPI.getDatasetReport(datasetId)
      setReport(res.data)
    } catch (e) { toast.error(e.response?.data?.detail || 'Failed to load report') }
    finally { setReportLoading(false) }
  }

  const toggleSection = (key) => {
    setExpandedSections(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const getCategoryColor = (category) => {
    if (category === 'source') return 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
    if (category === 'generated') return 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'
    if (category === 'processed') return 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400'
    if (category === 'reference') return 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400'
    return 'bg-surface-100 dark:bg-surface-700 text-surface-600'
  }

  const getStatusBadge = (status) => {
    if (status === 'active') return 'badge badge-success'
    if (status === 'available') return 'badge badge-secondary'
    return 'badge'
  }

  const getCategoryIcon = (category) => {
    if (category === 'source') return Database
    if (category === 'generated') return Activity
    if (category === 'processed') return Layers
    if (category === 'reference') return Shield
    return Database
  }

  const formatNumber = (v) => typeof v === 'number' ? v.toLocaleString() : v

  // Filtering
  const filtered = datasets.filter(d => {
    const matchesSearch = (d.name || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
                          (d.purpose || '').toLowerCase().includes(searchQuery.toLowerCase())
    const matchesCategory = filterCategory === 'all' || d.category === filterCategory
    return matchesSearch && matchesCategory
  })

  // ── Report Section Component ──
  const ReportSection = ({ id, icon: Icon, title, children }) => (
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
    <AdminLayout title="Datasets" subtitle="System datasets powering the FARUD ML pipeline">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
              <Database className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="text-base font-display font-bold text-surface-900 dark:text-white">System Datasets</h3>
              <p className="text-xs text-surface-500">{datasets.length} dataset{datasets.length !== 1 ? 's' : ''} discovered • {summary.active || 0} active in pipeline</p>
            </div>
          </div>
          <button onClick={fetchDatasets} className="btn btn-sm btn-secondary">
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </button>
        </div>

        {/* Summary Cards */}
        {!loading && datasets.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="card p-3 flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                <Database className="h-4 w-4 text-blue-600" />
              </div>
              <div>
                <p className="text-xs text-surface-500">Source</p>
                <p className="text-sm font-bold text-surface-900 dark:text-white">{summary.source || 0}</p>
              </div>
            </div>
            <div className="card p-3 flex items-center gap-3">
              <div className="w-8 h-8 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                <Activity className="h-4 w-4 text-purple-600" />
              </div>
              <div>
                <p className="text-xs text-surface-500">Generated</p>
                <p className="text-sm font-bold text-surface-900 dark:text-white">{summary.generated || 0}</p>
              </div>
            </div>
            <div className="card p-3 flex items-center gap-3">
              <div className="w-8 h-8 bg-amber-100 dark:bg-amber-900/30 rounded-lg flex items-center justify-center">
                <Layers className="h-4 w-4 text-amber-600" />
              </div>
              <div>
                <p className="text-xs text-surface-500">Processed</p>
                <p className="text-sm font-bold text-surface-900 dark:text-white">{summary.processed || 0}</p>
              </div>
            </div>
            <div className="card p-3 flex items-center gap-3">
              <div className="w-8 h-8 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg flex items-center justify-center">
                <Shield className="h-4 w-4 text-emerald-600" />
              </div>
              <div>
                <p className="text-xs text-surface-500">Reference</p>
                <p className="text-sm font-bold text-surface-900 dark:text-white">{summary.reference || 0}</p>
              </div>
            </div>
          </div>
        )}

        {/* Search & Filters */}
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4 text-surface-400" />
            <input type="text" value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
              className="input-field pl-11" placeholder="Search datasets..." />
          </div>
          <div className="flex gap-2">
            {['all', 'source', 'generated', 'processed', 'reference'].map(cat => (
              <button key={cat} onClick={() => setFilterCategory(cat)}
                className={`btn btn-sm ${filterCategory === cat ? 'btn-primary' : 'btn-secondary'}`}>
                {cat === 'all' ? 'All' : cat.charAt(0).toUpperCase() + cat.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Dataset List */}
        {loading ? (
          <div className="flex justify-center py-16"><Loader className="h-8 w-8 animate-spin text-primary-600" /></div>
        ) : filtered.length > 0 ? (
          <div className="grid gap-4">
            {filtered.map((ds) => {
              const CategoryIcon = getCategoryIcon(ds.category)

              return (
                <div key={ds.id} className="card p-5">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-start gap-4 flex-1 min-w-0">
                      {/* Category icon */}
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${getCategoryColor(ds.category)}`}>
                        <CategoryIcon className="h-6 w-6" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <p className="text-sm font-bold text-surface-900 dark:text-white truncate">{ds.name}</p>
                          <span className={getStatusBadge(ds.status)}>{ds.status}</span>
                          <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${getCategoryColor(ds.category)}`}>
                            {ds.category}
                          </span>
                        </div>
                        <p className="text-xs text-surface-500 dark:text-surface-400 mt-0.5 line-clamp-2">
                          {ds.purpose}
                        </p>
                        {/* Quick metrics row */}
                        <div className="flex items-center gap-4 mt-2 flex-wrap">
                          {ds.row_count != null && (
                            <span className="text-xs font-mono text-surface-600 dark:text-surface-300 flex items-center gap-1">
                              <Hash className="h-3 w-3" /> {formatNumber(ds.row_count)} rows
                            </span>
                          )}
                          {ds.col_count != null && (
                            <span className="text-xs font-mono text-surface-600 dark:text-surface-300 flex items-center gap-1">
                              <Layers className="h-3 w-3" /> {ds.col_count} columns
                            </span>
                          )}
                          <span className="text-xs font-mono text-surface-500 flex items-center gap-1">
                            <HardDrive className="h-3 w-3" /> {ds.file_size_mb} MB
                          </span>
                          {ds.pipeline_steps && ds.pipeline_steps.length > 0 && (
                            <span className="text-xs font-mono text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                              <GitBranch className="h-3 w-3" /> {ds.pipeline_steps.length} pipeline step{ds.pipeline_steps.length !== 1 ? 's' : ''}
                            </span>
                          )}
                        </div>
                        {/* Tags */}
                        {ds.tags && ds.tags.length > 0 && (
                          <div className="flex items-center gap-1.5 mt-2 flex-wrap">
                            {ds.tags.map((tag, i) => (
                              <span key={i} className="text-[10px] font-mono px-2 py-0.5 rounded-lg bg-surface-100 dark:bg-surface-700 text-surface-500">
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                    {/* Action */}
                    <button onClick={() => handleViewReport(ds.id)} className="btn btn-sm btn-primary flex-shrink-0">
                      <Eye className="h-3.5 w-3.5" /> View Report
                    </button>
                  </div>

                  {/* Pipeline steps */}
                  {ds.pipeline_steps && ds.pipeline_steps.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-surface-100 dark:border-surface-700">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-[10px] font-semibold text-surface-400 uppercase">Pipeline:</span>
                        {ds.pipeline_steps.map((step, i) => (
                          <span key={i} className="text-[10px] font-mono px-2 py-1 rounded-lg bg-primary-50 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400">
                            {step}
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
            <Database className="h-16 w-16 text-surface-300 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-surface-600 mb-2">{searchQuery || filterCategory !== 'all' ? 'No matching datasets' : 'No datasets found'}</h3>
            <p className="text-surface-500">Run the training pipeline to generate system datasets</p>
          </div>
        )}
      </div>

      {/* ══════════════════════════════════════════════════════════════════ */}
      {/*  COMPREHENSIVE DATASET ANALYSIS REPORT MODAL                    */}
      {/* ══════════════════════════════════════════════════════════════════ */}
      {(report || reportLoading) && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-surface-800 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6 shadow-2xl">
            {/* Header */}
            <div className="flex justify-between items-start mb-6">
              <div>
                <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-primary-600" /> Comprehensive Dataset Analysis Report
                </h3>
                {report && (
                  <p className="text-xs text-surface-500 mt-1">
                    {report.dataset_name} • {report.location} • Generated {new Date(report.generated_at).toLocaleString()}
                  </p>
                )}
              </div>
              <button onClick={() => setReport(null)} className="btn btn-ghost btn-sm"><X className="h-5 w-5" /></button>
            </div>

            {reportLoading ? (
              <div className="flex flex-col items-center justify-center py-16">
                <Loader className="h-8 w-8 animate-spin text-primary-600 mb-3" />
                <p className="text-sm text-surface-500">Analyzing dataset...</p>
              </div>
            ) : report ? (
              <div className="space-y-4">

                {/* ── Overview ── */}
                {report.overview && (
                  <ReportSection id="overview" icon={Info} title="Dataset Overview">
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {Object.entries(report.overview).map(([key, value]) => (
                        <MetricCard
                          key={key}
                          label={key.replace(/_/g, ' ')}
                          value={typeof value === 'number' ? formatNumber(value) : String(value)}
                          highlight={key === 'total_rows' || key === 'total_columns'}
                        />
                      ))}
                    </div>
                  </ReportSection>
                )}

                {/* ── Schema Analysis ── */}
                {report.schema_analysis && (
                  <ReportSection id="schema" icon={Layers} title="Schema Analysis">
                    {/* Dtype summary */}
                    {report.schema_analysis.dtype_summary && (
                      <div className="mb-4">
                        <p className="text-xs font-semibold text-surface-600 dark:text-surface-300 mb-2">Data Type Distribution</p>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                          {Object.entries(report.schema_analysis.dtype_summary).map(([dtype, count]) => (
                            <MetricCard key={dtype} label={dtype} value={count} />
                          ))}
                        </div>
                      </div>
                    )}
                    {/* Column details table */}
                    {report.schema_analysis.columns && report.schema_analysis.columns.length > 0 && (
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-surface-200 dark:border-surface-700">
                              <th className="text-left py-2 px-3 text-surface-500 font-semibold">Column</th>
                              <th className="text-left py-2 px-3 text-surface-500 font-semibold">Type</th>
                              <th className="text-right py-2 px-3 text-surface-500 font-semibold">Non-Null</th>
                              <th className="text-right py-2 px-3 text-surface-500 font-semibold">Null %</th>
                              <th className="text-right py-2 px-3 text-surface-500 font-semibold">Unique</th>
                              <th className="text-left py-2 px-3 text-surface-500 font-semibold">Sample</th>
                            </tr>
                          </thead>
                          <tbody>
                            {report.schema_analysis.columns.map((col, i) => (
                              <tr key={i} className="border-b border-surface-100 dark:border-surface-800">
                                <td className="py-2 px-3 font-mono font-medium text-surface-900 dark:text-white">{col.name}</td>
                                <td className="py-2 px-3 text-surface-600 dark:text-surface-400">{col.dtype}</td>
                                <td className="py-2 px-3 text-right text-surface-700 dark:text-surface-300">{formatNumber(col.non_null)}</td>
                                <td className={`py-2 px-3 text-right ${col.null_pct > 0 ? 'text-amber-600 dark:text-amber-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                                  {col.null_pct}%
                                </td>
                                <td className="py-2 px-3 text-right text-surface-700 dark:text-surface-300">{formatNumber(col.unique)}</td>
                                <td className="py-2 px-3 text-surface-500 max-w-[150px] truncate">{(col.sample_values || []).join(', ')}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                    {/* JSON schema for non-CSV */}
                    {report.schema_analysis.sample_keys && (
                      <div>
                        <p className="text-xs font-semibold text-surface-600 dark:text-surface-300 mb-2">Keys ({report.schema_analysis.top_level_keys || report.schema_analysis.total_records})</p>
                        <div className="flex flex-wrap gap-1.5">
                          {report.schema_analysis.sample_keys.map((k, i) => (
                            <span key={i} className="text-[10px] font-mono px-2 py-1 rounded-lg bg-surface-100 dark:bg-surface-700 text-surface-600 dark:text-surface-300">{k}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </ReportSection>
                )}

                {/* ── Data Quality ── */}
                {report.data_quality && (
                  <ReportSection id="quality" icon={Shield} title="Data Quality Assessment">
                    {/* Grade header */}
                    <div className="flex items-center gap-3 mb-4">
                      <div className={`px-3 py-1.5 rounded-lg text-sm font-bold ${
                        report.data_quality.grade === 'A' ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400' :
                        report.data_quality.grade === 'B' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400' :
                        report.data_quality.grade === 'C' ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400' :
                        'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                      }`}>
                        Grade: {report.data_quality.grade}
                      </div>
                      <span className="text-sm text-surface-600 dark:text-surface-300">
                        {report.data_quality.grade_label} — {report.data_quality.completeness_score}% complete
                      </span>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <MetricCard label="Total Cells" value={formatNumber(report.data_quality.total_cells)} />
                      <MetricCard label="Missing Cells" value={formatNumber(report.data_quality.missing_cells)} sub={`${report.data_quality.missing_pct}%`} />
                      <MetricCard label="Duplicate Rows" value={formatNumber(report.data_quality.duplicate_rows)} sub={`${report.data_quality.duplicate_pct}%`} />
                      <MetricCard label="Complete Rows" value={formatNumber(report.data_quality.complete_rows)} sub={`${report.data_quality.complete_rows_pct}%`} highlight />
                    </div>
                    {/* Columns with most missing */}
                    {report.data_quality.columns_with_missing && (
                      <div className="mt-3">
                        <p className="text-xs font-semibold text-surface-600 dark:text-surface-300 mb-2">Columns with Missing Values</p>
                        <div className="space-y-1.5">
                          {Object.entries(report.data_quality.columns_with_missing).map(([col, info]) => (
                            <div key={col} className="flex items-center justify-between p-2 rounded-lg bg-amber-50 dark:bg-amber-900/10">
                              <span className="text-xs font-mono text-surface-700 dark:text-surface-300">{col}</span>
                              <span className="text-xs text-amber-600 dark:text-amber-400">{formatNumber(info.count)} missing ({info.pct}%)</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </ReportSection>
                )}

                {/* ── Statistical Summary ── */}
                {report.statistical_summary && (
                  <ReportSection id="statistics" icon={BarChart3} title={`Statistical Summary (${report.statistical_summary.numeric_column_count} numeric columns)`}>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-surface-200 dark:border-surface-700">
                            <th className="text-left py-2 px-3 text-surface-500 font-semibold">Column</th>
                            <th className="text-right py-2 px-3 text-surface-500 font-semibold">Mean</th>
                            <th className="text-right py-2 px-3 text-surface-500 font-semibold">Std</th>
                            <th className="text-right py-2 px-3 text-surface-500 font-semibold">Min</th>
                            <th className="text-right py-2 px-3 text-surface-500 font-semibold">Median</th>
                            <th className="text-right py-2 px-3 text-surface-500 font-semibold">Max</th>
                            <th className="text-right py-2 px-3 text-surface-500 font-semibold">Skew</th>
                          </tr>
                        </thead>
                        <tbody>
                          {report.statistical_summary.columns.map((col, i) => (
                            <tr key={i} className="border-b border-surface-100 dark:border-surface-800">
                              <td className="py-2 px-3 font-mono font-medium text-surface-900 dark:text-white">{col.column}</td>
                              <td className="py-2 px-3 text-right font-mono text-surface-700 dark:text-surface-300">{col.mean}</td>
                              <td className="py-2 px-3 text-right font-mono text-surface-700 dark:text-surface-300">{col.std}</td>
                              <td className="py-2 px-3 text-right font-mono text-surface-700 dark:text-surface-300">{col.min}</td>
                              <td className="py-2 px-3 text-right font-mono text-surface-700 dark:text-surface-300">{col.median}</td>
                              <td className="py-2 px-3 text-right font-mono text-surface-700 dark:text-surface-300">{col.max}</td>
                              <td className={`py-2 px-3 text-right font-mono ${col.skew != null && Math.abs(col.skew) > 2 ? 'text-amber-600' : 'text-surface-500'}`}>
                                {col.skew != null ? col.skew : '—'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </ReportSection>
                )}

                {/* ── Categorical Analysis ── */}
                {report.categorical_analysis && (
                  <ReportSection id="categorical" icon={Tag} title={`Categorical Analysis (${report.categorical_analysis.categorical_column_count} columns)`}>
                    {report.categorical_analysis.columns.map((col, i) => (
                      <div key={i} className="mb-4 last:mb-0">
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-xs font-semibold text-surface-900 dark:text-white font-mono">{col.column}</p>
                          <span className="text-[10px] text-surface-500">{col.unique_values} unique • avg {col.avg_length} chars</span>
                        </div>
                        <div className="space-y-1">
                          {Object.entries(col.top_values).map(([val, count]) => (
                            <div key={val} className="flex items-center gap-2">
                              <div className="flex-1 bg-surface-100 dark:bg-surface-700 rounded-full h-4 overflow-hidden">
                                <div
                                  className="bg-primary-500 dark:bg-primary-600 h-full rounded-full"
                                  style={{ width: `${Math.min((count / Object.values(col.top_values)[0]) * 100, 100)}%` }}
                                />
                              </div>
                              <span className="text-[10px] font-mono text-surface-600 dark:text-surface-300 w-24 truncate">{val}</span>
                              <span className="text-[10px] font-mono text-surface-500 w-12 text-right">{formatNumber(count)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </ReportSection>
                )}

                {/* ── Text Analysis ── */}
                {report.text_analysis && (
                  <ReportSection id="text" icon={FileText} title="Text Column Analysis">
                    {Object.entries(report.text_analysis).map(([col, stats]) => (
                      <div key={col} className="mb-4 last:mb-0">
                        <p className="text-xs font-semibold text-surface-900 dark:text-white font-mono mb-2">{col}</p>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                          <MetricCard label="Total Texts" value={formatNumber(stats.total_texts)} />
                          <MetricCard label="Avg Char Length" value={stats.avg_char_length} />
                          <MetricCard label="Avg Word Count" value={stats.avg_word_count} />
                          <MetricCard label="Vocabulary Size" value={formatNumber(stats.vocabulary_size)} highlight />
                          <MetricCard label="Min Length" value={stats.min_char_length} />
                          <MetricCard label="Max Length" value={formatNumber(stats.max_char_length)} />
                          <MetricCard label="Empty/Short (<10)" value={stats.empty_or_short} />
                        </div>
                      </div>
                    ))}
                  </ReportSection>
                )}

                {/* ── Label Distribution ── */}
                {report.label_distribution && (
                  <ReportSection id="labels" icon={Target} title="Label Distribution">
                    <div className="flex items-center gap-3 mb-4">
                      <MetricCard label="Label Column" value={report.label_distribution.label_column} highlight />
                      <MetricCard label="Classes" value={report.label_distribution.classes} />
                      <MetricCard label="Class Balance" value={report.label_distribution.class_balance} />
                    </div>
                    <div className="space-y-2">
                      {Object.entries(report.label_distribution.distribution).map(([label, info]) => (
                        <div key={label} className="flex items-center gap-3 p-2.5 rounded-lg bg-surface-50 dark:bg-surface-700">
                          <span className="text-xs font-mono font-bold text-surface-900 dark:text-white w-20">{label}</span>
                          <div className="flex-1 bg-surface-200 dark:bg-surface-600 rounded-full h-5 overflow-hidden">
                            <div
                              className={`h-full rounded-full ${label === '1' || label.toLowerCase() === 'scam' || label.toLowerCase() === 'fraud' ? 'bg-red-500' : 'bg-emerald-500'}`}
                              style={{ width: `${info.percentage}%` }}
                            />
                          </div>
                          <span className="text-xs font-mono text-surface-600 dark:text-surface-300 w-24 text-right">
                            {formatNumber(info.count)} ({info.percentage}%)
                          </span>
                        </div>
                      ))}
                    </div>
                  </ReportSection>
                )}

                {/* ── Data Preview ── */}
                {report.data_preview && (
                  <ReportSection id="preview" icon={Table} title="Data Preview (First 5 Rows)">
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-surface-200 dark:border-surface-700">
                            {report.data_preview.columns.map((col, i) => (
                              <th key={i} className="px-3 py-2 text-left font-semibold text-surface-700 dark:text-surface-300 whitespace-nowrap">{col}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {report.data_preview.rows.map((row, ri) => (
                            <tr key={ri} className="border-b border-surface-100 dark:border-surface-700/50 hover:bg-surface-50 dark:hover:bg-surface-700/30">
                              {report.data_preview.columns.map((col, ci) => (
                                <td key={ci} className="px-3 py-1.5 text-surface-600 dark:text-surface-400 max-w-[200px] truncate">{String(row[col] ?? '')}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </ReportSection>
                )}

                {/* ── Data Lineage ── */}
                {report.data_lineage && (
                  <ReportSection id="lineage" icon={GitBranch} title="Data Lineage">
                    <div className="space-y-3">
                      {report.data_lineage.origin && (
                        <div className="p-3 rounded-xl bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800">
                          <p className="text-[10px] font-bold text-blue-600 dark:text-blue-400 uppercase mb-1">Origin</p>
                          <p className="text-xs text-surface-700 dark:text-surface-300">{report.data_lineage.origin}</p>
                        </div>
                      )}
                      {report.data_lineage.used_by && report.data_lineage.used_by.length > 0 && (
                        <div>
                          <p className="text-[10px] font-bold text-surface-500 uppercase mb-1.5">Used By</p>
                          <div className="flex flex-wrap gap-1.5">
                            {report.data_lineage.used_by.map((u, i) => (
                              <span key={i} className="text-[10px] font-mono px-2 py-1 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800">
                                {u}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      {report.data_lineage.generates && report.data_lineage.generates.length > 0 && (
                        <div>
                          <p className="text-[10px] font-bold text-surface-500 uppercase mb-1.5">Generates</p>
                          <div className="flex flex-wrap gap-1.5">
                            {report.data_lineage.generates.map((g, i) => (
                              <span key={i} className="text-[10px] font-mono px-2 py-1 rounded-lg bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 border border-purple-200 dark:border-purple-800">
                                {g}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      {report.data_lineage.models_trained && report.data_lineage.models_trained.length > 0 && (
                        <div>
                          <p className="text-[10px] font-bold text-surface-500 uppercase mb-1.5">Models Trained</p>
                          <div className="flex flex-wrap gap-1.5">
                            {report.data_lineage.models_trained.map((m, i) => (
                              <span key={i} className="text-[10px] font-mono px-2 py-1 rounded-lg bg-amber-50 dark:bg-amber-900/20 text-amber-600 dark:text-amber-400 border border-amber-200 dark:border-amber-800">
                                {m}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
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

export default AdminDatasets
