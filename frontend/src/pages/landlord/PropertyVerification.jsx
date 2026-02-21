import { useState, useRef } from 'react'
import Layout from '../../components/Layout'
import { landlordAPI } from '../../services/api'
import toast from 'react-hot-toast'
import { 
  Image, Upload, Loader, AlertTriangle, CheckCircle, 
  XCircle, Shield, Trash2, Plus, Camera
} from 'lucide-react'

const PropertyVerification = () => {
  const [images, setImages] = useState([])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files)
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
    
    const newImages = files
      .filter(file => {
        if (!validTypes.includes(file.type)) {
          toast.error(`${file.name} is not a valid image format`)
          return false
        }
        if (file.size > 10 * 1024 * 1024) {
          toast.error(`${file.name} is too large (max 10MB)`)
          return false
        }
        return true
      })
      .map(file => ({
        file,
        preview: URL.createObjectURL(file),
        name: file.name
      }))

    setImages(prev => [...prev, ...newImages].slice(0, 20))
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const removeImage = (index) => {
    URL.revokeObjectURL(images[index].preview)
    setImages(images.filter((_, i) => i !== index))
  }

  const clearAll = () => {
    images.forEach(img => URL.revokeObjectURL(img.preview))
    setImages([])
    setResult(null)
  }

  const handleVerify = async () => {
    if (images.length === 0) {
      toast.error('Please upload at least one image')
      return
    }

    setLoading(true)
    setResult(null)

    try {
      const formData = new FormData()
      images.forEach(img => {
        formData.append('files', img.file)
      })
      const res = await landlordAPI.verifyListingImagesUpload(formData)
      setResult(res.data.result || res.data)
      toast.success('Image verification complete!')
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Image verification failed')
    } finally {
      setLoading(false)
    }
  }

  const getRiskGradient = (score) => {
    if (score < 0.3) return 'from-green-500 to-green-600'
    if (score < 0.5) return 'from-yellow-500 to-yellow-600'
    if (score < 0.7) return 'from-orange-500 to-orange-600'
    return 'from-red-500 to-red-600'
  }

  return (
    <Layout title="Property Image Verification">
      <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
        {/* Upload Section */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-6">
            <Camera className="h-8 w-8 text-green-600" />
            <div>
              <h3 className="text-2xl font-bold">Verify Property Images</h3>
              <p className="text-gray-600">Upload listing photos to detect AI-generated, stock, or stolen images using CNN analysis</p>
            </div>
          </div>

          {/* Drop Zone */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary-500 transition cursor-pointer mb-4"
          >
            <Upload className="h-12 w-12 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-600">Click to upload or drag & drop property images</p>
            <p className="text-xs text-gray-400 mt-1">JPEG, PNG, GIF, WebP (max 10MB each, up to 20 images)</p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            onChange={handleFileSelect}
            className="hidden"
          />

          {/* Image Previews */}
          {images.length > 0 && (
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">{images.length} image{images.length > 1 ? 's' : ''} selected</span>
                <button onClick={clearAll} className="text-sm text-red-500 hover:text-red-700">Clear All</button>
              </div>
              <div className="grid grid-cols-3 md:grid-cols-5 gap-3">
                {images.map((img, idx) => (
                  <div key={idx} className="relative group">
                    <img src={img.preview} alt={img.name} className="w-full h-24 object-cover rounded-lg border" />
                    <button
                      onClick={() => removeImage(idx)}
                      className="absolute top-1 right-1 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition"
                    >
                      <XCircle className="h-3 w-3" />
                    </button>
                    <p className="text-xs text-gray-500 truncate mt-1">{img.name}</p>
                  </div>
                ))}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full h-24 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center hover:border-primary-500 cursor-pointer"
                >
                  <Plus className="h-6 w-6 text-gray-400" />
                </div>
              </div>
            </div>
          )}

          {/* Verify Button */}
          <button
            onClick={handleVerify}
            disabled={loading || images.length === 0}
            className="w-full btn-primary py-3"
          >
            {loading ? (
              <><Loader className="inline h-5 w-5 mr-2 animate-spin" />Analyzing Images...</>
            ) : (
              <><Shield className="inline h-5 w-5 mr-2" />Verify Images ({images.length})</>
            )}
          </button>
        </div>

        {/* Results */}
        {result && (
          <div className="space-y-4 animate-fade-in">
            {/* Overall Risk Banner */}
            <div className={`rounded-xl p-6 text-white bg-gradient-to-r ${getRiskGradient(result.overall_risk_score || 0)}`}>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm opacity-80">Overall Image Risk</p>
                  <p className="text-5xl font-bold">{((result.overall_risk_score || 0) * 100).toFixed(0)}%</p>
                </div>
                <div className="text-right">
                  <p className="text-sm opacity-80">Risk Level</p>
                  <p className="text-3xl font-bold capitalize">{result.overall_risk_level}</p>
                  <p className="text-sm opacity-80 mt-1">{result.image_count} images analyzed</p>
                </div>
              </div>
            </div>

            {/* Summary Stats */}
            <div className="card">
              <h4 className="font-bold mb-3">Analysis Summary</h4>
              {typeof result.summary === 'string' && (
                <p className="text-gray-700 mb-3">{result.summary}</p>
              )}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
                {result.property_images_count != null && (
                  <div className="bg-green-50 rounded-lg p-3 text-center">
                    <p className="text-xs text-gray-500">Property Images</p>
                    <p className="text-xl font-bold text-green-700">{result.property_images_count}</p>
                  </div>
                )}
                {result.suspicious_images_count != null && (
                  <div className="bg-red-50 rounded-lg p-3 text-center">
                    <p className="text-xs text-gray-500">Suspicious</p>
                    <p className="text-xl font-bold text-red-700">{result.suspicious_images_count}</p>
                  </div>
                )}
                {result.ai_suspected_count != null && (
                  <div className={`rounded-lg p-3 text-center ${result.ai_suspected_count > 0 ? 'bg-purple-50' : 'bg-gray-50'}`}>
                    <p className="text-xs text-gray-500">AI-Generated</p>
                    <p className={`text-xl font-bold ${result.ai_suspected_count > 0 ? 'text-purple-700' : 'text-gray-400'}`}>{result.ai_suspected_count}</p>
                  </div>
                )}
                {result.average_ai_score != null && (
                  <div className={`rounded-lg p-3 text-center ${result.average_ai_score >= 0.25 ? 'bg-purple-50' : 'bg-gray-50'}`}>
                    <p className="text-xs text-gray-500">AI Score</p>
                    <p className={`text-xl font-bold ${result.average_ai_score >= 0.25 ? 'text-purple-700' : 'text-gray-400'}`}>{(result.average_ai_score * 100).toFixed(0)}%</p>
                  </div>
                )}
                {result.image_count != null && (
                  <div className="bg-blue-50 rounded-lg p-3 text-center">
                    <p className="text-xs text-gray-500">Total Images</p>
                    <p className="text-xl font-bold text-blue-700">{result.image_count}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Per-Image Results */}
            {result.images && result.images.length > 0 && (
              <div className="card">
                <h4 className="font-bold mb-4">Per-Image Analysis</h4>
                <div className="space-y-3">
                  {result.images.map((img, i) => (
                    <div key={i} className="bg-gray-50 rounded-lg p-4 border">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <Image className="h-4 w-4 text-gray-500" />
                          <span className="font-medium text-sm">{img.metadata?.filename || `Image ${i + 1}`}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          {img.is_property_image ? (
                            <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">Property</span>
                          ) : (
                            <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-0.5 rounded-full">Non-Property</span>
                          )}
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            (img.risk_score || 0) < 0.3 ? 'bg-green-100 text-green-800' :
                            (img.risk_score || 0) < 0.6 ? 'bg-yellow-100 text-yellow-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            {((img.risk_score || 0) * 100).toFixed(0)}% risk
                          </span>
                          {img.ai_detection_score != null && img.ai_detection_score >= 0.15 && (
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              img.ai_detection_score >= 0.4 ? 'bg-purple-200 text-purple-800' :
                              img.ai_detection_score >= 0.25 ? 'bg-purple-100 text-purple-700' :
                              'bg-purple-50 text-purple-600'
                            }`}>
                              🤖 AI {(img.ai_detection_score * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs text-gray-600">
                        {img.classification?.is_property_related != null && (
                          <div><span className="text-gray-400">Property:</span> {img.classification.is_property_related ? 'Yes' : 'No'}</div>
                        )}
                        {img.property_confidence != null && (
                          <div><span className="text-gray-400">Confidence:</span> {(img.property_confidence * 100).toFixed(0)}%</div>
                        )}
                        {img.risk_level && (
                          <div><span className="text-gray-400">Level:</span> <span className="capitalize">{img.risk_level}</span></div>
                        )}
                        {img.quality_score != null && (
                          <div><span className="text-gray-400">Quality:</span> {(img.quality_score * 100).toFixed(0)}%</div>
                        )}
                        {img.ai_detection_score != null && (
                          <div>
                            <span className="text-gray-400">AI Score:</span>{' '}
                            <span className={img.ai_detection_score >= 0.25 ? 'text-purple-700 font-medium' : ''}>
                              {(img.ai_detection_score * 100).toFixed(0)}%
                            </span>
                          </div>
                        )}
                      </div>
                      {/* Classification top classes */}
                      {img.classification?.top_classes && img.classification.top_classes.length > 0 && (
                        <div className="mt-2">
                          <span className="text-xs text-gray-400">Top Classes: </span>
                          {img.classification.top_classes.slice(0, 3).map((cls, j) => (
                            <span key={j} className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full mr-1">
                              {cls.label || cls.class || cls.name || `Class ${j+1}`} {cls.confidence ? `(${(cls.confidence * 100).toFixed(0)}%)` : ''}
                            </span>
                          ))}
                        </div>
                      )}
                      {/* Risk indicators */}
                      {img.indicators && img.indicators.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {img.indicators.map((ind, j) => (
                            <span key={j} className="text-xs bg-yellow-100 text-yellow-700 px-2 py-0.5 rounded-full" title={typeof ind === 'object' ? (ind.description || '') : ''}>
                              {typeof ind === 'string' ? ind : (ind.code || ind.name || ind.description || 'Indicator')}
                            </span>
                          ))}
                        </div>
                      )}
                      {/* Explanation */}
                      {img.explanation && (
                        <p className="mt-2 text-xs text-gray-500">{img.explanation}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Warnings */}
            {result.warnings && result.warnings.length > 0 && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <h4 className="font-bold text-yellow-800 mb-2 flex items-center">
                  <AlertTriangle className="h-5 w-5 mr-2" />
                  Warnings
                </h4>
                <ul className="space-y-1">
                  {result.warnings.map((w, i) => (
                    <li key={i} className="text-sm text-yellow-700">{typeof w === 'string' ? w : (w.message || w.description || JSON.stringify(w))}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </Layout>
  )
}

export default PropertyVerification
