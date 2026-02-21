import { useState, useRef } from 'react'
import LandlordLayout from '../../components/layouts/LandlordLayout'
import { landlordAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  FileCheck, Upload, Loader, AlertTriangle, CheckCircle,
  XCircle, Shield, Trash2, FileText
} from 'lucide-react'

const DocumentVerificationNew = () => {
  const [documentType, setDocumentType] = useState('paystub')
  const [applicantName, setApplicantName] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const fileInputRef = useRef(null)

  const documentTypes = [
    { value: 'paystub', label: 'Pay Stub' },
    { value: 'id_card', label: 'Government ID' },
    { value: 'bank_statement', label: 'Bank Statement' },
    { value: 'rental_application', label: 'Rental Application' },
    { value: 'employment_letter', label: 'Employment Letter' },
    { value: 'tax_document', label: 'Tax Document' },
    { value: 'utility_bill', label: 'Utility Bill' },
  ]

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'application/pdf']
    if (!validTypes.includes(file.type)) return toast.error('Please upload an image or PDF')
    if (file.size > 15 * 1024 * 1024) return toast.error('File too large (max 15MB)')
    setSelectedFile(file)
    setPreview(file.type.startsWith('image/') ? URL.createObjectURL(file) : null)
    setResult(null)
  }

  const handleVerify = async () => {
    if (!selectedFile) return toast.error('Please select a file')
    setLoading(true)
    setResult(null)
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('document_type', documentType)
      if (applicantName.trim()) formData.append('applicant_name', applicantName)
      const res = await landlordAPI.verifyDocumentUpload(formData)
      setResult(res.data.result || res.data)
      toast.success('Document verification complete!')
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Document verification failed')
    } finally {
      setLoading(false)
    }
  }

  const clearForm = () => {
    setSelectedFile(null)
    setPreview(null)
    setResult(null)
    setApplicantName('')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const getRiskBadge = (level) => {
    const m = { low: 'badge-success', medium: 'badge-warning', high: 'badge-danger', critical: 'badge-danger' }
    return m[level] || 'badge-info'
  }

  return (
    <LandlordLayout title="Document Verification" subtitle="AI-powered document fraud analysis using OCR">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Upload */}
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-blue-500 rounded-xl flex items-center justify-center">
              <FileCheck className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white">Upload Document</h3>
              <p className="text-sm text-surface-500 dark:text-surface-400">Upload a tenant document for AI fraud analysis</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="input-label">Document Type</label>
                <select value={documentType} onChange={(e) => setDocumentType(e.target.value)} className="input-field">
                  {documentTypes.map(dt => <option key={dt.value} value={dt.value}>{dt.label}</option>)}
                </select>
              </div>
              <div>
                <label className="input-label">Applicant Name (optional)</label>
                <input type="text" value={applicantName} onChange={(e) => setApplicantName(e.target.value)}
                  className="input-field" placeholder="Enter applicant name" />
              </div>
            </div>

            {/* Drop zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-surface-300 dark:border-surface-600 rounded-2xl p-8 text-center cursor-pointer hover:border-landlord-500 transition-colors group"
            >
              <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileSelect} accept="image/*,.pdf" />
              {selectedFile ? (
                <div className="space-y-3">
                  {preview ? (
                    <img src={preview} alt="Preview" className="max-h-48 mx-auto rounded-xl shadow-md" />
                  ) : (
                    <FileText className="h-16 w-16 mx-auto text-blue-500" />
                  )}
                  <p className="text-sm font-medium text-surface-900 dark:text-white">{selectedFile.name}</p>
                  <p className="text-xs text-surface-500">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                </div>
              ) : (
                <>
                  <Upload className="h-12 w-12 mx-auto text-surface-400 group-hover:text-landlord-500 transition-colors mb-3" />
                  <p className="text-sm font-medium text-surface-700 dark:text-surface-300">Click to upload or drag & drop</p>
                  <p className="text-xs text-surface-400">Images or PDF, max 15MB</p>
                </>
              )}
            </div>

            <div className="flex gap-3">
              <button onClick={handleVerify} disabled={loading || !selectedFile} className="btn btn-lg btn-primary flex-1">
                {loading ? <><Loader className="h-5 w-5 animate-spin" /> Analyzing...</> : <><Shield className="h-5 w-5" /> Verify Document</>}
              </button>
              {selectedFile && (
                <button onClick={clearForm} className="btn btn-lg btn-secondary">
                  <Trash2 className="h-5 w-5" />
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Result */}
        {result && (
          <div className="card p-6 animate-fade-in-up">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white">Verification Result</h3>
              <span className={getRiskBadge(result.risk_level)}>{result.risk_level || 'analyzed'}</span>
            </div>

            {result.risk_score !== undefined && (
              <div className="p-4 rounded-xl bg-surface-50 dark:bg-surface-700 mb-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-surface-500 dark:text-surface-400">Risk Score</span>
                  <span className="text-2xl font-display font-bold text-surface-900 dark:text-white">
                    {typeof result.risk_score === 'number' ? (result.risk_score * 100).toFixed(0) + '%' : result.risk_score}
                  </span>
                </div>
                <div className="w-full bg-surface-200 dark:bg-surface-600 rounded-full h-2 mt-2">
                  <div className={`h-2 rounded-full ${
                    (result.risk_score || 0) < 0.3 ? 'bg-emerald-500' : (result.risk_score || 0) < 0.6 ? 'bg-amber-500' : 'bg-red-500'
                  }`} style={{ width: `${Math.min((result.risk_score || 0) * 100, 100)}%` }} />
                </div>
              </div>
            )}

            {result.summary && (
              <div className="p-4 rounded-xl bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 mb-4">
                <p className="text-sm text-surface-700 dark:text-surface-300">{result.summary}</p>
              </div>
            )}

            {result.indicators?.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-surface-900 dark:text-white mb-2">Risk Indicators</h4>
                {result.indicators.map((ind, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-surface-50 dark:bg-surface-700/50">
                    {ind.severity >= 3 ? <AlertTriangle className="h-4 w-4 text-accent-red mt-0.5" /> : <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5" />}
                    <div>
                      <p className="text-sm text-surface-900 dark:text-white font-medium">{ind.description || ind.type}</p>
                      {ind.evidence?.length > 0 && (
                        <p className="text-xs text-surface-500 mt-1">Evidence: {ind.evidence.join(', ')}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </LandlordLayout>
  )
}

export default DocumentVerificationNew
