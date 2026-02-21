import { useState, useRef } from 'react'
import LandlordLayout from '../../components/layouts/LandlordLayout'
import { landlordAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  Users, Loader, Shield, AlertTriangle, CheckCircle, UserCheck,
  Upload, X, FileText, Eye, EyeOff, GitCompareArrows, Info
} from 'lucide-react'

const DOCUMENT_TYPES = [
  { value: 'paystub', label: 'Pay Stub' },
  { value: 'id_card', label: 'ID Card' },
  { value: 'bank_statement', label: 'Bank Statement' },
  { value: 'rental_application', label: 'Rental Application' },
  { value: 'employment_letter', label: 'Employment Letter' },
  { value: 'tax_document', label: 'Tax Document' },
  { value: 'utility_bill', label: 'Utility Bill' },
]

const TenantVerificationNew = () => {
  const [applicantName, setApplicantName] = useState('')
  const [documents, setDocuments] = useState([])   // { file, base64, document_type, preview }
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const fileInputRef = useRef(null)

  // ---- file helpers ----
  const fileToBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result)   // data:…;base64,…
      reader.onerror = reject
      reader.readAsDataURL(file)
    })

  const handleFilesSelected = async (e) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) return

    const newDocs = await Promise.all(
      files.map(async (file) => {
        const base64 = await fileToBase64(file)
        return {
          file,
          base64,
          document_type: 'id_card',          // default; user can change
          preview: file.type.startsWith('image/') ? base64 : null,
        }
      })
    )
    setDocuments((prev) => [...prev, ...newDocs])
    // reset input so the same file can be added again if needed
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const removeDocument = (idx) =>
    setDocuments((prev) => prev.filter((_, i) => i !== idx))

  const updateDocType = (idx, newType) =>
    setDocuments((prev) =>
      prev.map((d, i) => (i === idx ? { ...d, document_type: newType } : d))
    )

  // ---- submit ----
  const handleSubmit = async (e) => {
    e.preventDefault()
    if (documents.length === 0) {
      toast.error('Please upload at least one document')
      return
    }
    setLoading(true)
    setResult(null)
    try {
      const payload = {
        applicant_name: applicantName,
        documents: documents.map((d) => ({
          document_base64: d.base64,
          document_type: d.document_type,
          applicant_name: applicantName,
          filename: d.file.name,
        })),
      }
      const res = await landlordAPI.verifyTenant(payload)
      setResult(res.data)
      toast.success('Tenant verification complete!')
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Verification failed')
    } finally {
      setLoading(false)
    }
  }

  // ---- risk badge ----
  const getRiskBadge = (level) => {
    const m = { low: 'badge-success', medium: 'badge-warning', high: 'badge-danger' }
    return m[level] || 'badge-info'
  }

  const riskBarColor = (score) =>
    score < 0.3 ? 'bg-emerald-500' : score < 0.6 ? 'bg-amber-500' : 'bg-red-500'

  return (
    <LandlordLayout title="Tenant Screening" subtitle="Comprehensive applicant verification">
      <div className="max-w-3xl mx-auto space-y-6">

        {/* ── Applicant Name + Document Upload Form ── */}
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-purple-500 rounded-xl flex items-center justify-center">
              <Users className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white">Applicant Details</h3>
              <p className="text-sm text-surface-500 dark:text-surface-400">
                Enter the applicant&apos;s name and upload their supporting documents
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Name field */}
            <div>
              <label className="input-label">Full Name *</label>
              <input
                type="text" required className="input-field"
                value={applicantName}
                onChange={(e) => setApplicantName(e.target.value)}
                placeholder="John Smith"
              />
            </div>

            {/* Upload area */}
            <div>
              <label className="input-label">Documents *</label>
              <div
                className="border-2 border-dashed border-surface-300 dark:border-surface-600 rounded-xl p-6
                           text-center cursor-pointer hover:border-purple-400 dark:hover:border-purple-500 transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="h-8 w-8 mx-auto text-surface-400 mb-2" />
                <p className="text-sm text-surface-500 dark:text-surface-400">
                  Click or drag &amp; drop to upload documents
                </p>
                <p className="text-xs text-surface-400 dark:text-surface-500 mt-1">
                  Accepts images &amp; PDFs — pay stubs, ID cards, bank statements, etc.
                </p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/*,.pdf"
                className="hidden"
                onChange={handleFilesSelected}
              />
            </div>

            {/* Document list */}
            {documents.length > 0 && (
              <div className="space-y-3">
                {documents.map((doc, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-3 p-3 rounded-xl bg-surface-50 dark:bg-surface-700 border border-surface-200 dark:border-surface-600"
                  >
                    {/* Thumbnail / icon */}
                    {doc.preview ? (
                      <img src={doc.preview} alt="" className="h-12 w-12 object-cover rounded-lg flex-shrink-0" />
                    ) : (
                      <div className="h-12 w-12 flex items-center justify-center rounded-lg bg-surface-200 dark:bg-surface-600 flex-shrink-0">
                        <FileText className="h-5 w-5 text-surface-400" />
                      </div>
                    )}

                    {/* File name */}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-surface-800 dark:text-surface-200 truncate">
                        {doc.file.name}
                      </p>
                      <p className="text-xs text-surface-400">
                        {(doc.file.size / 1024).toFixed(0)} KB
                      </p>
                    </div>

                    {/* Document type selector */}
                    <select
                      className="input-field !w-auto text-sm py-1.5"
                      value={doc.document_type}
                      onChange={(e) => updateDocType(idx, e.target.value)}
                    >
                      {DOCUMENT_TYPES.map((t) => (
                        <option key={t.value} value={t.value}>{t.label}</option>
                      ))}
                    </select>

                    {/* Remove button */}
                    <button type="button" onClick={() => removeDocument(idx)}
                      className="p-1.5 rounded-lg hover:bg-red-100 dark:hover:bg-red-900/30 text-surface-400 hover:text-red-500 transition-colors">
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Submit */}
            <button type="submit" disabled={loading || documents.length === 0} className="btn btn-lg btn-primary w-full">
              {loading
                ? <><Loader className="h-5 w-5 animate-spin" /> Verifying Documents…</>
                : <><UserCheck className="h-5 w-5" /> Verify Tenant ({documents.length} doc{documents.length !== 1 ? 's' : ''})</>
              }
            </button>
          </form>
        </div>

        {/* ── Verification Result ── */}
        {result && (
          <div className="card p-6 animate-fade-in-up space-y-5">
            {/* Header */}
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white">Verification Result</h3>
              <span className={getRiskBadge(result.overall_risk_level)}>{result.overall_risk_level}</span>
            </div>

            {/* Overall risk score bar */}
            {result.overall_risk_score !== undefined && (
              <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700">
                <div className="flex justify-between items-baseline">
                  <span className="text-sm text-surface-500">Overall Risk Score</span>
                  <span className="text-2xl font-display font-bold text-surface-900 dark:text-white">
                    {(result.overall_risk_score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-surface-200 dark:bg-surface-600 rounded-full h-2 mt-2">
                  <div className={`h-2 rounded-full ${riskBarColor(result.overall_risk_score)}`}
                    style={{ width: `${result.overall_risk_score * 100}%` }} />
                </div>
              </div>
            )}

            {/* Quick stats */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3 rounded-xl bg-surface-50 dark:bg-surface-700 text-center">
                <p className="text-xs text-surface-500 mb-1">Documents</p>
                <p className="text-lg font-bold text-surface-900 dark:text-white">{result.document_count}</p>
              </div>
              <div className="p-3 rounded-xl bg-emerald-50 dark:bg-emerald-900/20 text-center">
                <p className="text-xs text-surface-500 mb-1">Verified</p>
                <p className="text-lg font-bold text-emerald-600">{result.verified_count}</p>
              </div>
              <div className="p-3 rounded-xl bg-red-50 dark:bg-red-900/20 text-center">
                <p className="text-xs text-surface-500 mb-1">Suspicious</p>
                <p className="text-lg font-bold text-red-600">{result.suspicious_count}</p>
              </div>
              <div className="p-3 rounded-xl bg-surface-50 dark:bg-surface-700 text-center">
                <p className="text-xs text-surface-500 mb-1">Name Match</p>
                <p className="text-lg font-bold text-surface-900 dark:text-white">
                  {result.name_consistent === true ? '✅ Yes' : result.name_consistent === false ? '❌ No' : '—'}
                </p>
              </div>
            </div>

            {/* Recommendation */}
            {result.recommendation && (
              <div className="p-4 rounded-xl bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 text-sm font-medium text-surface-700 dark:text-surface-300">
                {result.recommendation}
              </div>
            )}

            {/* Summary */}
            {result.summary && (
              <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700 text-sm text-surface-600 dark:text-surface-300">
                <p className="font-semibold text-surface-900 dark:text-white mb-1">Summary</p>
                {result.summary}
              </div>
            )}

            {/* Per-document breakdown */}
            {result.documents?.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Document Details</h4>
                <div className="space-y-3">
                  {result.documents.map((doc, i) => (
                    <div key={i} className="p-4 rounded-xl border border-surface-200 dark:border-surface-600">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <FileText className="h-4 w-4 text-surface-400" />
                          <span className="text-sm font-medium text-surface-800 dark:text-surface-200">
                            {doc.filename || `Document ${i + 1}`}
                          </span>
                          <span className="text-xs text-surface-400">
                            ({DOCUMENT_TYPES.find((t) => t.value === doc.document_type)?.label || doc.document_type})
                          </span>
                        </div>
                        <span className={getRiskBadge(doc.risk_level)}>{doc.risk_level}</span>
                      </div>

                      {/* Risk bar */}
                      {doc.risk_score !== undefined && (
                        <div className="flex items-center gap-3 mb-2">
                          <div className="flex-1 bg-surface-200 dark:bg-surface-600 rounded-full h-1.5">
                            <div className={`h-1.5 rounded-full ${riskBarColor(doc.risk_score)}`}
                              style={{ width: `${doc.risk_score * 100}%` }} />
                          </div>
                          <span className="text-xs font-medium text-surface-500">
                            {(doc.risk_score * 100).toFixed(0)}%
                          </span>
                        </div>
                      )}

                      {/* Extracted info */}
                      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-surface-500">
                        {doc.extracted_name && <span>Name: <strong className="text-surface-700 dark:text-surface-300">{doc.extracted_name}</strong></span>}
                        {doc.extracted_employer && <span>Employer: <strong className="text-surface-700 dark:text-surface-300">{doc.extracted_employer}</strong></span>}
                        {doc.extracted_amounts?.length > 0 && (
                          <span>Amounts: <strong className="text-surface-700 dark:text-surface-300">${doc.extracted_amounts.join(', $')}</strong></span>
                        )}
                        {doc.quality_score !== undefined && <span>Quality: {(doc.quality_score * 100).toFixed(0)}%</span>}
                      </div>

                      {/* Explanation */}
                      {doc.explanation && (
                        <p className="text-xs text-surface-400 mt-2">{doc.explanation}</p>
                      )}

                      {/* Fraud indicators */}
                      {doc.indicators?.length > 0 && (
                        <div className="mt-2 space-y-1">
                          {doc.indicators.map((ind, j) => (
                            <div key={j} className="flex items-start gap-1.5 text-xs">
                              <AlertTriangle className="h-3 w-3 text-amber-500 flex-shrink-0 mt-0.5" />
                              <span className="text-surface-600 dark:text-surface-400">
                                {ind.description || ind.indicator || JSON.stringify(ind)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── Cross-Document Consistency Analysis ── */}
            {result.cross_document_analysis && (
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <GitCompareArrows className="h-4 w-4 text-purple-500" />
                  <h4 className="text-sm font-semibold text-surface-900 dark:text-white">Cross-Document Consistency</h4>
                  <span className={getRiskBadge(
                    result.cross_document_analysis.overall?.consistency === 'consistent' ? 'low'
                    : result.cross_document_analysis.overall?.consistency === 'critical_mismatch' ? 'high'
                    : 'medium'
                  )}>
                    {result.cross_document_analysis.overall?.consistency?.replace(/_/g, ' ')}
                  </span>
                </div>

                {/* Consistency score */}
                {result.cross_document_analysis.overall?.score !== undefined && (
                  <div className="flex items-center gap-3 mb-3 p-3 rounded-xl bg-surface-50 dark:bg-surface-700">
                    <span className="text-xs text-surface-500">Consistency Score</span>
                    <div className="flex-1 bg-surface-200 dark:bg-surface-600 rounded-full h-1.5">
                      <div className={`h-1.5 rounded-full ${riskBarColor(1 - result.cross_document_analysis.overall.score)}`}
                        style={{ width: `${result.cross_document_analysis.overall.score * 100}%` }} />
                    </div>
                    <span className="text-xs font-medium text-surface-500">
                      {(result.cross_document_analysis.overall.score * 100).toFixed(0)}%
                    </span>
                  </div>
                )}

                {/* Critical issues */}
                {result.cross_document_analysis.critical_issues?.length > 0 && (
                  <div className="space-y-2 mb-3">
                    {result.cross_document_analysis.critical_issues.map((issue, i) => (
                      <div key={i} className="flex items-start gap-2 p-3 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
                        <AlertTriangle className="h-4 w-4 text-red-500 flex-shrink-0 mt-0.5" />
                        <div className="text-xs text-red-700 dark:text-red-300">
                          <span className="font-semibold uppercase">{issue.severity}</span>
                          <span className="mx-1">—</span>
                          {issue.description}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Warnings */}
                {result.cross_document_analysis.warnings?.length > 0 && (
                  <div className="space-y-2 mb-3">
                    {result.cross_document_analysis.warnings.map((warn, i) => (
                      <div key={i} className="flex items-start gap-2 p-3 rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
                        <Info className="h-4 w-4 text-amber-500 flex-shrink-0 mt-0.5" />
                        <span className="text-xs text-amber-700 dark:text-amber-300">{warn.description}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Consistency checks summary */}
                {result.cross_document_analysis.checks && (
                  <div className="flex items-center gap-4 text-xs text-surface-500">
                    <span>{result.cross_document_analysis.checks.total} check(s) performed</span>
                    <span className="text-emerald-600">✓ {result.cross_document_analysis.checks.passed} passed</span>
                    {result.cross_document_analysis.checks.failed > 0 && (
                      <span className="text-red-600">✗ {result.cross_document_analysis.checks.failed} failed</span>
                    )}
                  </div>
                )}

                {/* Cross-doc recommendation */}
                {result.cross_document_analysis.recommendation && (
                  <div className="mt-3 p-3 rounded-xl bg-surface-50 dark:bg-surface-700 text-xs text-surface-600 dark:text-surface-300">
                    {result.cross_document_analysis.recommendation}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </LandlordLayout>
  )
}

export default TenantVerificationNew
