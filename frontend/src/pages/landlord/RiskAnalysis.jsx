import { useState, useEffect } from 'react'
import LandlordLayout from '../../components/layouts/LandlordLayout'
import { landlordAPI } from '../../services/api'
import {
  Shield, AlertTriangle, CheckCircle, Loader, FileText,
  Upload, FilePlus, UserCheck, X, Eye, Layers
} from 'lucide-react'
import toast from 'react-hot-toast'

const RiskAnalysis = () => {
  const [activeTab, setActiveTab] = useState('history')
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)

  // Cross-document verification
  const [crossDocFiles, setCrossDocFiles] = useState([])
  const [crossDocLoading, setCrossDocLoading] = useState(false)
  const [crossDocResult, setCrossDocResult] = useState(null)

  // Full application verification
  const [appFiles, setAppFiles] = useState([])
  const [applicantName, setApplicantName] = useState('')
  const [appLoading, setAppLoading] = useState(false)
  const [appResult, setAppResult] = useState(null)

  useEffect(() => {
    const load = async () => {
      try {
        const { data } = await landlordAPI.getVerificationHistory(0, 50)
        setHistory(data.history || [])
      } catch (err) {
        console.error(err)
        toast.error('Failed to load risk analyses')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const guessDocType = (filename) => {
    const lower = filename.toLowerCase()
    if (lower.includes('id') || lower.includes('passport') || lower.includes('license') || lower.includes('driver')) return 'id'
    if (lower.includes('pay') || lower.includes('stub') || lower.includes('income') || lower.includes('salary')) return 'paystub'
    if (lower.includes('lease') || lower.includes('agreement') || lower.includes('contract')) return 'lease'
    if (lower.includes('bank') || lower.includes('statement')) return 'bank_statement'
    if (lower.includes('employ') || lower.includes('letter') || lower.includes('reference')) return 'employment'
    return 'other'
  }

  // Cross-document: compare multiple tenant documents for consistency
  const handleCrossDocVerification = async (e) => {
    e.preventDefault()
    if (crossDocFiles.length < 2) return toast.error('Upload at least 2 documents to cross-verify')
    setCrossDocLoading(true)
    setCrossDocResult(null)
    try {
      const documents = []
      for (const file of crossDocFiles) {
        const base64 = await fileToBase64(file)
        documents.push({ name: file.name, type: guessDocType(file.name), text: base64 })
      }
      const response = await landlordAPI.verifyCrossDocument({ documents })
      setCrossDocResult(response.data)
      toast.success('Cross-document verification complete!')
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Cross-document verification failed')
    } finally {
      setCrossDocLoading(false)
    }
  }

  // Full application verification: OCR + cross-doc + image analysis combined
  const handleFullAppVerification = async (e) => {
    e.preventDefault()
    if (appFiles.length === 0) return toast.error('Upload application documents')
    setAppLoading(true)
    setAppResult(null)
    try {
      const documents = []
      for (const file of appFiles) {
        const base64 = await fileToBase64(file)
        documents.push({ name: file.name, type: guessDocType(file.name), image_base64: base64 })
      }
      const response = await landlordAPI.verifyFullApplication({
        applicant_name: applicantName || null,
        documents,
      })
      setAppResult(response.data)
      toast.success('Full application verification complete!')
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Full application verification failed')
    } finally {
      setAppLoading(false)
    }
  }

  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.readAsDataURL(file)
      reader.onload = () => resolve(reader.result.split(',')[1])
      reader.onerror = reject
    })
  }

  const riskBadge = (score) => {
    if (score == null) return { label: 'N/A', cls: 'badge-secondary' }
    if (score < 0.3) return { label: 'Low Risk', cls: 'badge-success' }
    if (score < 0.6) return { label: 'Medium Risk', cls: 'badge-warning' }
    return { label: 'High Risk', cls: 'badge-danger' }
  }

  const tabs = [
    { id: 'history', icon: Shield, label: 'Risk History' },
    { id: 'crossdoc', icon: Layers, label: 'Cross-Document Check' },
    { id: 'fullapp', icon: UserCheck, label: 'Full Application Verify' },
  ]

  return (
    <LandlordLayout title="Risk Analysis" subtitle="AI-powered fraud risk assessments">
      {/* Tab selector */}
      <div className="flex items-center gap-2 mb-6 flex-wrap">
        {tabs.map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)}
            className={`btn btn-md ${activeTab === t.id ? 'btn-primary' : 'btn-secondary'}`}>
            <t.icon className="h-4 w-4" /> {t.label}
          </button>
        ))}
      </div>

      {/* ====== HISTORY TAB ====== */}
      {activeTab === 'history' && (
        loading ? (
          <div className="card p-16 text-center"><Loader className="h-8 w-8 animate-spin text-landlord-500 mx-auto mb-3" /><p className="text-surface-500">Loading...</p></div>
        ) : history.length === 0 ? (
          <div className="card p-16 text-center">
            <Shield className="h-12 w-12 text-surface-300 dark:text-surface-600 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-surface-900 dark:text-white mb-2">No risk analyses yet</h3>
            <p className="text-sm text-surface-500">Run document, tenant, or property verifications to see risk analysis results here.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {history.map((item, idx) => {
              const riskScore = item.details?.risk_score ?? item.details?.overall_risk_score ?? null
              const risk = riskBadge(riskScore)
              const verType = item.action || item.entity_type || 'Verification'
              const flags = item.details?.flags || item.details?.inconsistencies?.map(i => i.field || i.type) || []
              const summary = item.details?.summary || item.details?.analysis || null
              return (
                <div key={item.id || idx} className="card p-5">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-xl ${riskScore != null && riskScore > 0.5 ? 'bg-red-100 dark:bg-red-900/30' : 'bg-green-100 dark:bg-green-900/30'}`}>
                        {riskScore != null && riskScore > 0.5 ? <AlertTriangle className="h-5 w-5 text-red-500" /> : <CheckCircle className="h-5 w-5 text-green-500" />}
                      </div>
                      <div>
                        <h4 className="font-semibold text-surface-900 dark:text-white">{verType}</h4>
                        <p className="text-xs text-surface-500">{item.created_at ? new Date(item.created_at).toLocaleDateString() : ''}</p>
                      </div>
                    </div>
                    <span className={risk.cls}>{risk.label}</span>
                  </div>
                  {riskScore != null && (
                    <div className="mb-3">
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-surface-500">Risk Score</span>
                        <span className="font-medium">{(riskScore * 100).toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-surface-100 dark:bg-surface-800 rounded-full h-2">
                        <div className={`h-2 rounded-full ${riskScore > 0.6 ? 'bg-red-500' : riskScore > 0.3 ? 'bg-yellow-500' : 'bg-green-500'}`}
                          style={{ width: `${riskScore * 100}%` }} />
                      </div>
                    </div>
                  )}
                  {flags.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {flags.map((flag, fi) => (
                        <span key={fi} className="text-xs bg-red-50 text-red-600 dark:bg-red-900/20 dark:text-red-400 px-2 py-1 rounded-lg">{typeof flag === 'string' ? flag : flag.description || JSON.stringify(flag)}</span>
                      ))}
                    </div>
                  )}
                  {summary && (
                    <p className="text-xs text-surface-500 mt-2">{typeof summary === 'string' ? summary : JSON.stringify(summary).slice(0, 200)}</p>
                  )}
                </div>
              )
            })}
          </div>
        )
      )}

      {/* ====== CROSS-DOCUMENT VERIFICATION (cross_document_engine) ====== */}
      {activeTab === 'crossdoc' && (
        <>
          <div className="card p-6 mb-6">
            <div className="flex items-center gap-2 mb-4">
              <Layers className="h-5 w-5 text-landlord-500" />
              <h3 className="text-base font-display font-bold text-surface-900 dark:text-white">Cross-Document Verification</h3>
            </div>
            <p className="text-sm text-surface-500 dark:text-surface-400 mb-4">
              Upload multiple tenant documents (ID, pay stubs, bank statements, employment letter) to check for consistency across documents. The AI uses NER and fuzzy matching to detect discrepancies in names, addresses, dates, and financial figures.
            </p>
            <form onSubmit={handleCrossDocVerification} className="space-y-4">
              <div>
                <label className="input-label">Upload Documents (at least 2)</label>
                <div className="border-2 border-dashed border-surface-300 dark:border-surface-600 rounded-xl p-6 text-center hover:border-landlord-400 transition-colors">
                  <Upload className="h-8 w-8 text-surface-400 mx-auto mb-2" />
                  <p className="text-sm text-surface-500 mb-2">Drop files or click to browse</p>
                  <input type="file" multiple accept=".pdf,.jpg,.jpeg,.png,.doc,.docx" className="w-full opacity-0 absolute inset-0 cursor-pointer"
                    style={{ position: 'relative', opacity: 1 }}
                    onChange={(e) => setCrossDocFiles(Array.from(e.target.files || []))} />
                </div>
                {crossDocFiles.length > 0 && (
                  <div className="mt-3 space-y-1">
                    {crossDocFiles.map((f, i) => (
                      <div key={i} className="flex items-center gap-2 text-sm text-surface-700 dark:text-surface-300">
                        <FileText className="h-4 w-4 text-landlord-500" /> {f.name}
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <button type="submit" disabled={crossDocLoading} className="btn btn-lg btn-primary w-full">
                {crossDocLoading ? <><Loader className="h-5 w-5 animate-spin" /> Verifying...</> : <><Layers className="h-5 w-5" /> Run Cross-Document Check</>}
              </button>
            </form>
          </div>

          {/* Cross-doc Result */}
          {crossDocResult && <VerificationResultCard result={crossDocResult} title="Cross-Document Verification" riskBadge={riskBadge} />}
        </>
      )}

      {/* ====== FULL APPLICATION VERIFICATION (ocr + cross_doc + real_image) ====== */}
      {activeTab === 'fullapp' && (
        <>
          <div className="card p-6 mb-6">
            <div className="flex items-center gap-2 mb-4">
              <UserCheck className="h-5 w-5 text-landlord-500" />
              <h3 className="text-base font-display font-bold text-surface-900 dark:text-white">Full Application Verification</h3>
            </div>
            <p className="text-sm text-surface-500 dark:text-surface-400 mb-4">
              Upload all tenant application documents for a comprehensive verification. This combines OCR text extraction, cross-document consistency checking, and image authenticity analysis in one pass.
            </p>
            <form onSubmit={handleFullAppVerification} className="space-y-4">
              <div>
                <label className="input-label">Applicant Name (optional)</label>
                <input type="text" value={applicantName} onChange={e => setApplicantName(e.target.value)}
                  className="input-field" placeholder="e.g., John Smith" />
              </div>
              <div>
                <label className="input-label">Application Documents</label>
                <div className="border-2 border-dashed border-surface-300 dark:border-surface-600 rounded-xl p-6 text-center hover:border-landlord-400 transition-colors">
                  <FilePlus className="h-8 w-8 text-surface-400 mx-auto mb-2" />
                  <p className="text-sm text-surface-500 mb-2">Upload ID, income proof, references, photos</p>
                  <input type="file" multiple accept=".pdf,.jpg,.jpeg,.png,.doc,.docx" className="w-full"
                    onChange={(e) => setAppFiles(Array.from(e.target.files || []))} />
                </div>
                {appFiles.length > 0 && (
                  <div className="mt-3 space-y-1">
                    {appFiles.map((f, i) => (
                      <div key={i} className="flex items-center gap-2 text-sm text-surface-700 dark:text-surface-300">
                        <FileText className="h-4 w-4 text-landlord-500" /> {f.name}
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <button type="submit" disabled={appLoading} className="btn btn-lg btn-primary w-full">
                {appLoading ? <><Loader className="h-5 w-5 animate-spin" /> Processing...</> : <><UserCheck className="h-5 w-5" /> Verify Full Application</>}
              </button>
            </form>
          </div>

          {/* Full App Result */}
          {appResult && <VerificationResultCard result={appResult} title="Full Application Verification" riskBadge={riskBadge} />}
        </>
      )}
    </LandlordLayout>
  )
}

/* Reusable verification result card */
const VerificationResultCard = ({ result, title, riskBadge }) => {
  const risk = riskBadge(result.risk_score)
  return (
    <div className="card animate-fade-in-up">
      <div className="p-6 border-b border-surface-200 dark:border-surface-700">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white">{title}</h3>
          <span className={risk.cls}>{risk.label}</span>
        </div>
        {result.risk_score != null && (
          <div className="mt-3">
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-surface-500">Overall Risk</span>
              <span className="font-bold">{(result.risk_score * 100).toFixed(0)}%</span>
            </div>
            <div className="w-full bg-surface-100 dark:bg-surface-800 rounded-full h-2.5">
              <div className={`h-2.5 rounded-full ${result.risk_score > 0.6 ? 'bg-red-500' : result.risk_score > 0.3 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                style={{ width: `${result.risk_score * 100}%` }} />
            </div>
          </div>
        )}
      </div>

      {/* Inconsistencies / Issues */}
      {result.inconsistencies && result.inconsistencies.length > 0 && (
        <div className="p-6 border-b border-surface-200 dark:border-surface-700">
          <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Inconsistencies Found ({result.inconsistencies.length})</h4>
          <div className="space-y-2">
            {result.inconsistencies.map((item, i) => (
              <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-red-50 dark:bg-red-900/10">
                <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-surface-900 dark:text-white">{item.field || item.type || 'Discrepancy'}</p>
                  <p className="text-xs text-surface-500 mt-0.5">{item.description || item.detail || JSON.stringify(item)}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Flags */}
      {result.flags && result.flags.length > 0 && (
        <div className="p-6 border-b border-surface-200 dark:border-surface-700">
          <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Flags</h4>
          <div className="flex flex-wrap gap-2">
            {result.flags.map((flag, i) => (
              <span key={i} className="text-xs bg-amber-50 text-amber-700 dark:bg-amber-900/20 dark:text-amber-400 px-2.5 py-1 rounded-lg">
                {typeof flag === 'string' ? flag : flag.description || JSON.stringify(flag)}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Documents Analyzed */}
      {result.documents_analyzed && result.documents_analyzed.length > 0 && (
        <div className="p-6 border-b border-surface-200 dark:border-surface-700">
          <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-3">Documents Analyzed</h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {result.documents_analyzed.map((doc, i) => (
              <div key={i} className="flex items-center gap-2 p-2 rounded-lg bg-surface-50 dark:bg-surface-700 text-sm">
                <FileText className="h-4 w-4 text-landlord-500" />
                <span className="text-surface-700 dark:text-surface-300 truncate">{doc.filename || doc.name || doc}</span>
                {doc.status && <span className={`badge ${doc.status === 'verified' ? 'badge-success' : 'badge-warning'}`}>{doc.status}</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary */}
      {(result.summary || result.analysis) && (
        <div className="p-6">
          <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-2">Summary</h4>
          <p className="text-sm text-surface-600 dark:text-surface-400 leading-relaxed whitespace-pre-line">
            {result.summary || result.analysis}
          </p>
        </div>
      )}
    </div>
  )
}

export default RiskAnalysis
