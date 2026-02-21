import { useState, useRef } from 'react'
import Layout from '../../components/Layout'
import { landlordAPI } from '../../services/api'
import toast from 'react-hot-toast'
import { 
  FileCheck, Upload, Loader, AlertTriangle, CheckCircle, 
  XCircle, Shield, Eye, Trash2, FileText
} from 'lucide-react'

const DocumentVerification = () => {
  const [mode, setMode] = useState('upload') // 'upload' or 'base64'
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
    if (!validTypes.includes(file.type)) {
      toast.error('Please upload an image or PDF file')
      return
    }
    if (file.size > 15 * 1024 * 1024) {
      toast.error('File is too large (max 15MB)')
      return
    }

    setSelectedFile(file)
    if (file.type.startsWith('image/')) {
      setPreview(URL.createObjectURL(file))
    } else {
      setPreview(null)
    }
    setResult(null)
  }

  const handleVerifyUpload = async () => {
    if (!selectedFile) {
      toast.error('Please select a file')
      return
    }

    setLoading(true)
    setResult(null)
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('document_type', documentType)
      if (applicantName.trim()) {
        formData.append('applicant_name', applicantName)
      }
      const res = await landlordAPI.verifyDocumentUpload(formData)
      setResult(res.data.result || res.data)
      toast.success('Document verification complete!')
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Document verification failed')
    } finally {
      setLoading(false)
    }
  }

  const handleVerifyBase64 = async () => {
    if (!selectedFile) {
      toast.error('Please select a file')
      return
    }

    setLoading(true)
    setResult(null)
    try {
      const reader = new FileReader()
      reader.onload = async () => {
        try {
          const base64 = reader.result
          const res = await landlordAPI.verifyDocument({
            document_base64: base64,
            document_type: documentType,
            applicant_name: applicantName || null,
            filename: selectedFile.name
          })
          setResult(res.data)
          toast.success('Document verification complete!')
        } catch (error) {
          toast.error(error.response?.data?.detail || 'Document verification failed')
        } finally {
          setLoading(false)
        }
      }
      reader.readAsDataURL(selectedFile)
    } catch (error) {
      toast.error('Failed to read file')
      setLoading(false)
    }
  }

  const handleVerify = () => {
    if (mode === 'upload') {
      handleVerifyUpload()
    } else {
      handleVerifyBase64()
    }
  }

  const clearForm = () => {
    setSelectedFile(null)
    setPreview(null)
    setResult(null)
    setApplicantName('')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const getRiskColor = (level) => {
    const colors = {
      low: 'bg-green-100 text-green-800 border-green-300',
      medium: 'bg-yellow-100 text-yellow-800 border-yellow-300',
      high: 'bg-orange-100 text-orange-800 border-orange-300',
      critical: 'bg-red-100 text-red-800 border-red-300'
    }
    return colors[level] || 'bg-gray-100 text-gray-800 border-gray-300'
  }

  const getRiskGradient = (score) => {
    if (score < 0.3) return 'from-green-500 to-green-600'
    if (score < 0.5) return 'from-yellow-500 to-yellow-600'
    if (score < 0.7) return 'from-orange-500 to-orange-600'
    return 'from-red-500 to-red-600'
  }

  return (
    <Layout title="Document Verification">
      <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
        {/* Upload Section */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-6">
            <FileCheck className="h-8 w-8 text-blue-600" />
            <div>
              <h3 className="text-2xl font-bold">Verify Document</h3>
              <p className="text-gray-600">Upload a tenant document for AI-powered fraud analysis using OCR</p>
            </div>
          </div>

          <div className="space-y-4">
            {/* Document Type */}
            <div>
              <label className="block text-sm font-medium mb-2">Document Type *</label>
              <select
                value={documentType}
                onChange={(e) => setDocumentType(e.target.value)}
                className="input-field"
              >
                {documentTypes.map((dt) => (
                  <option key={dt.value} value={dt.value}>{dt.label}</option>
                ))}
              </select>
            </div>

            {/* Applicant Name */}
            <div>
              <label className="block text-sm font-medium mb-2">Applicant Name (Optional)</label>
              <input
                type="text"
                value={applicantName}
                onChange={(e) => setApplicantName(e.target.value)}
                className="input-field"
                placeholder="John Doe — used to verify name consistency"
              />
            </div>

            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium mb-2">Document Image *</label>
              <div 
                className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary-500 transition cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                {preview ? (
                  <div className="space-y-3">
                    <img src={preview} alt="Document preview" className="max-h-48 mx-auto rounded-lg shadow" />
                    <p className="text-sm text-gray-600">{selectedFile?.name}</p>
                  </div>
                ) : selectedFile ? (
                  <div className="space-y-2">
                    <FileText className="h-12 w-12 text-gray-400 mx-auto" />
                    <p className="text-sm text-gray-600">{selectedFile.name}</p>
                    <p className="text-xs text-gray-400">{(selectedFile.size / 1024).toFixed(0)} KB</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Upload className="h-12 w-12 text-gray-400 mx-auto" />
                    <p className="text-gray-600">Click to upload or drag & drop</p>
                    <p className="text-xs text-gray-400">Supports JPEG, PNG, GIF, WebP, PDF (max 15MB)</p>
                  </div>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.pdf"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>

            {/* Actions */}
            <div className="flex space-x-3">
              <button
                onClick={handleVerify}
                disabled={loading || !selectedFile}
                className="flex-1 btn-primary py-3"
              >
                {loading ? (
                  <><Loader className="inline h-5 w-5 mr-2 animate-spin" />Verifying...</>
                ) : (
                  <><Shield className="inline h-5 w-5 mr-2" />Verify Document</>
                )}
              </button>
              {selectedFile && (
                <button onClick={clearForm} className="btn-secondary py-3">
                  <Trash2 className="inline h-4 w-4 mr-1" />
                  Clear
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Verification Result */}
        {result && (
          <div className="space-y-4 animate-fade-in">
            {/* Risk Score Banner */}
            <div className={`rounded-xl p-6 text-white bg-gradient-to-r ${getRiskGradient(result.risk_score)}`}>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm opacity-80">Document Risk Score</p>
                  <p className="text-5xl font-bold">{((result.risk_score || 0) * 100).toFixed(0)}%</p>
                </div>
                <div className="text-right">
                  <p className="text-sm opacity-80">Risk Level</p>
                  <p className="text-3xl font-bold capitalize">{result.risk_level}</p>
                </div>
              </div>
            </div>

            {/* Details Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Extracted Data */}
              <div className="card">
                <h4 className="font-bold mb-3 flex items-center">
                  <Eye className="h-5 w-5 mr-2 text-blue-600" />
                  Extracted Information
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Document Type</span>
                    <span className="font-medium capitalize">{result.document_type?.replace(/_/g, ' ')}</span>
                  </div>
                  {result.extracted_name && (
                    <div className="flex justify-between">
                      <span className="text-gray-500">Extracted Name</span>
                      <span className="font-medium">{result.extracted_name}</span>
                    </div>
                  )}
                  {result.extracted_employer && (
                    <div className="flex justify-between">
                      <span className="text-gray-500">Employer</span>
                      <span className="font-medium">{result.extracted_employer}</span>
                    </div>
                  )}
                  {result.extracted_amounts && result.extracted_amounts.length > 0 && (
                    <div className="flex justify-between">
                      <span className="text-gray-500">Amounts Found</span>
                      <span className="font-medium">{result.extracted_amounts.map(a => `$${a.toFixed(2)}`).join(', ')}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Quality Scores */}
              <div className="card">
                <h4 className="font-bold mb-3 flex items-center">
                  <CheckCircle className="h-5 w-5 mr-2 text-green-600" />
                  Quality Assessment
                </h4>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-500">Document Quality</span>
                      <span className="font-medium">{((result.quality_score || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${(result.quality_score || 0) * 100}%` }}></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-500">Consistency Score</span>
                      <span className="font-medium">{((result.consistency_score || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-green-600 h-2 rounded-full" style={{ width: `${(result.consistency_score || 0) * 100}%` }}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Explanation */}
            {result.explanation && (
              <div className="card">
                <h4 className="font-bold mb-2 flex items-center">
                  <FileText className="h-5 w-5 mr-2 text-primary-600" />
                  Analysis Summary
                </h4>
                <p className="text-gray-700">{result.explanation}</p>
              </div>
            )}

            {/* Fraud Indicators */}
            {result.indicators && result.indicators.length > 0 && (
              <div className="card">
                <h4 className="font-bold mb-3 flex items-center">
                  <AlertTriangle className="h-5 w-5 mr-2 text-yellow-600" />
                  Fraud Indicators ({result.indicators.length})
                </h4>
                <div className="space-y-2">
                  {result.indicators.map((indicator, i) => (
                    <div key={i} className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                      <div className="flex items-start">
                        <AlertTriangle className="h-4 w-4 text-yellow-600 mr-2 mt-0.5 flex-shrink-0" />
                        <div>
                          <p className="font-medium text-yellow-800 text-sm">{indicator.name || indicator.type || 'Indicator'}</p>
                          <p className="text-xs text-yellow-700 mt-1">{indicator.description || indicator.detail || JSON.stringify(indicator)}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </Layout>
  )
}

export default DocumentVerification
