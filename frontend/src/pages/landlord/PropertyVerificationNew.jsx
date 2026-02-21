import { useState, useRef } from 'react'
import LandlordLayout from '../../components/layouts/LandlordLayout'
import { landlordAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  Image, Upload, Loader, AlertTriangle, CheckCircle, Shield, Trash2, X, Plus, Info, Bot, Eye, Globe, ExternalLink,
  Fingerprint, Microscope, Scissors, Activity, Layers, BarChart3
} from 'lucide-react'

const getRiskBadge = (level) => {
  const config = {
    authentic: { cls: 'badge-success', label: 'Authentic' },
    likely_authentic: { cls: 'badge-success', label: 'Likely Authentic' },
    uncertain: { cls: 'badge-warning', label: 'Uncertain' },
    suspicious: { cls: 'badge-danger', label: 'Suspicious' },
    likely_fake: { cls: 'badge-danger', label: 'Likely Fake' },
  }
  return config[level] || { cls: 'badge-info', label: level || 'Unknown' }
}

const PropertyVerificationNew = () => {
  const [files, setFiles] = useState([])
  const [previews, setPreviews] = useState([])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const fileInputRef = useRef(null)

  const handleFilesSelect = (e) => {
    const selected = Array.from(e.target.files)
    const validFiles = selected.filter(f => f.type.startsWith('image/') && f.size <= 15 * 1024 * 1024)
    if (validFiles.length < selected.length) toast.error('Some files were skipped (invalid type or too large)')
    const newPreviews = validFiles.map(f => URL.createObjectURL(f))
    setFiles(prev => [...prev, ...validFiles])
    setPreviews(prev => [...prev, ...newPreviews])
    setResult(null)
  }

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
    setPreviews(prev => { URL.revokeObjectURL(prev[index]); return prev.filter((_, i) => i !== index) })
  }

  const handleVerify = async () => {
    if (files.length === 0) return toast.error('Please select images')
    setLoading(true)
    setResult(null)
    try {
      const formData = new FormData()
      files.forEach(f => formData.append('files', f))
      const res = await landlordAPI.verifyListingImagesUpload(formData)
      setResult(res.data?.result || res.data)
      toast.success('Image verification complete!')
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Image verification failed')
    } finally {
      setLoading(false)
    }
  }

  const clearAll = () => {
    previews.forEach(p => URL.revokeObjectURL(p))
    setFiles([])
    setPreviews([])
    setResult(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <LandlordLayout title="Property Image Verification" subtitle="Detect fake or misleading property photos">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Upload */}
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-emerald-500 rounded-xl flex items-center justify-center">
              <Image className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white">Upload Property Images</h3>
              <p className="text-sm text-surface-500 dark:text-surface-400">Upload listing photos to detect stock images, AI-generated, or stolen photos</p>
            </div>
          </div>

          {/* Image grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 mb-4">
            {previews.map((p, i) => (
              <div key={i} className="relative group rounded-xl overflow-hidden aspect-square">
                <img src={p} alt="" className="w-full h-full object-cover" />
                <button onClick={() => removeFile(i)}
                  className="absolute top-2 right-2 w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
            <button
              onClick={() => fileInputRef.current?.click()}
              className="aspect-square rounded-xl border-2 border-dashed border-surface-300 dark:border-surface-600 flex flex-col items-center justify-center gap-2 hover:border-landlord-500 transition-colors cursor-pointer"
            >
              <Plus className="h-6 w-6 text-surface-400" />
              <span className="text-xs text-surface-500">Add Images</span>
            </button>
          </div>
          <input ref={fileInputRef} type="file" className="hidden" multiple accept="image/*" onChange={handleFilesSelect} />

          <div className="flex gap-3">
            <button onClick={handleVerify} disabled={loading || files.length === 0} className="btn btn-lg btn-primary flex-1">
              {loading ? <><Loader className="h-5 w-5 animate-spin" /> Analyzing {files.length} images...</> : <><Shield className="h-5 w-5" /> Verify {files.length} Image{files.length !== 1 ? 's' : ''}</>}
            </button>
            {files.length > 0 && (
              <button onClick={clearAll} className="btn btn-lg btn-secondary"><Trash2 className="h-5 w-5" /></button>
            )}
          </div>
        </div>

        {/* Result */}
        {result && (
          <div className="card p-6 animate-fade-in-up">
            <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white mb-4">Verification Results</h3>

            {/* Overall Assessment */}
            <div className={`flex items-center justify-between p-4 rounded-xl mb-4 ${
              result.overall_risk_level === 'authentic' || result.overall_risk_level === 'likely_authentic'
                ? 'bg-emerald-50 dark:bg-emerald-900/10 border border-emerald-200 dark:border-emerald-800'
                : result.overall_risk_level === 'uncertain'
                ? 'bg-amber-50 dark:bg-amber-900/10 border border-amber-200 dark:border-amber-800'
                : 'bg-red-50 dark:bg-red-900/10 border border-red-200 dark:border-red-800'
            }`}>
              <div className="flex items-center gap-3">
                {result.overall_risk_level === 'authentic' || result.overall_risk_level === 'likely_authentic'
                  ? <CheckCircle className="h-6 w-6 text-emerald-500" />
                  : result.overall_risk_level === 'uncertain'
                  ? <Info className="h-6 w-6 text-amber-500" />
                  : <AlertTriangle className="h-6 w-6 text-red-500" />
                }
                <div>
                  <h4 className="text-sm font-semibold text-surface-900 dark:text-white">Overall Assessment</h4>
                  <p className="text-xs text-surface-500">{result.image_count} image{result.image_count !== 1 ? 's' : ''} analyzed</p>
                </div>
              </div>
              <span className={getRiskBadge(result.overall_risk_level).cls}>
                {getRiskBadge(result.overall_risk_level).label}
              </span>
            </div>

            {/* Stats Row */}
            {(result.ai_suspected_count > 0 || result.suspicious_images_count > 0 || result.web_match_count > 0 || result.forensics) && (
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 mb-4">
                <div className="text-center p-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                  <p className="text-lg font-bold text-surface-900 dark:text-white">{result.image_count}</p>
                  <p className="text-xs text-surface-500">Total</p>
                </div>
                <div className="text-center p-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                  <p className="text-lg font-bold text-emerald-600">{result.property_images_count}</p>
                  <p className="text-xs text-surface-500">Property</p>
                </div>
                <div className="text-center p-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                  <p className={`text-lg font-bold ${result.ai_suspected_count > 0 ? 'text-red-600' : 'text-surface-900 dark:text-white'}`}>{result.ai_suspected_count || 0}</p>
                  <p className="text-xs text-surface-500">AI Suspected</p>
                </div>
                <div className="text-center p-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                  <p className={`text-lg font-bold ${(result.forensics?.gan_flagged_count || 0) > 0 ? 'text-red-600' : 'text-surface-900 dark:text-white'}`}>{result.forensics?.gan_flagged_count || 0}</p>
                  <p className="text-xs text-surface-500">GAN Flagged</p>
                </div>
                <div className="text-center p-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                  <p className={`text-lg font-bold ${(result.forensics?.tampering_flagged_count || 0) > 0 ? 'text-red-600' : 'text-surface-900 dark:text-white'}`}>{result.forensics?.tampering_flagged_count || 0}</p>
                  <p className="text-xs text-surface-500">Tampered</p>
                </div>
                <div className="text-center p-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                  <p className={`text-lg font-bold ${(result.web_match_count || 0) > 0 ? 'text-orange-600' : 'text-surface-900 dark:text-white'}`}>{result.web_match_count || 0}</p>
                  <p className="text-xs text-surface-500">Web Matches</p>
                </div>
                <div className="text-center p-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                  <p className={`text-lg font-bold ${result.suspicious_images_count > 0 ? 'text-red-600' : 'text-surface-900 dark:text-white'}`}>{result.suspicious_images_count}</p>
                  <p className="text-xs text-surface-500">Suspicious</p>
                </div>
              </div>
            )}

            {/* Summary */}
            {result.summary && (
              <div className="flex items-start gap-2 p-3 rounded-xl bg-surface-50 dark:bg-surface-700/50 mb-4">
                <CheckCircle className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-surface-700 dark:text-surface-300">{result.summary}</p>
              </div>
            )}

            {/* Per-Image Analysis */}
            {result.images?.length > 0 && (
              <div className="space-y-3">
                <h4 className="text-sm font-semibold text-surface-900 dark:text-white">Per-Image Analysis</h4>
                {result.images.map((img, i) => {
                  const riskBadge = getRiskBadge(img.risk_level)
                  const isRisky = img.risk_level === 'suspicious' || img.risk_level === 'likely_fake' || img.risk_level === 'uncertain'
                  return (
                    <div key={i} className={`p-4 rounded-xl border ${isRisky ? 'border-red-200 dark:border-red-800 bg-red-50/50 dark:bg-red-900/5' : 'border-surface-200 dark:border-surface-700 bg-surface-50 dark:bg-surface-700/50'}`}>
                      <div className="flex items-start gap-4">
                        {previews[i] && <img src={previews[i]} alt="" className="w-20 h-20 rounded-lg object-cover flex-shrink-0" />}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium text-surface-900 dark:text-white">Image {i + 1}</span>
                            <span className={riskBadge.cls}>{riskBadge.label}</span>
                          </div>

                          {/* Risk Score Bar */}
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-xs text-surface-500 w-16">Risk</span>
                            <div className="flex-1 bg-surface-100 dark:bg-surface-800 rounded-full h-2">
                              <div className={`h-2 rounded-full ${img.risk_score > 0.5 ? 'bg-red-500' : img.risk_score > 0.3 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                                style={{ width: `${Math.max(img.risk_score * 100, 3)}%` }} />
                            </div>
                            <span className="text-xs font-bold text-surface-700 dark:text-surface-300 w-10 text-right">{(img.risk_score * 100).toFixed(0)}%</span>
                          </div>

                          {/* AI Detection Score Bar */}
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-xs text-surface-500 w-16 flex items-center gap-1"><Bot className="h-3 w-3" /> AI</span>
                            <div className="flex-1 bg-surface-100 dark:bg-surface-800 rounded-full h-2">
                              <div className={`h-2 rounded-full ${img.ai_detection_score > 0.3 ? 'bg-red-500' : img.ai_detection_score > 0.15 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                                style={{ width: `${Math.max(img.ai_detection_score * 100, 3)}%` }} />
                            </div>
                            <span className="text-xs font-bold text-surface-700 dark:text-surface-300 w-10 text-right">{(img.ai_detection_score * 100).toFixed(0)}%</span>
                          </div>

                          {/* Property Classification */}
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-xs text-surface-500 w-16 flex items-center gap-1"><Eye className="h-3 w-3" /> Type</span>
                            <span className={`text-xs font-medium ${img.is_property_image ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}`}>
                              {img.is_property_image ? `Property image (${(img.property_confidence * 100).toFixed(0)}% conf.)` : 'Not property-related'}
                            </span>
                          </div>

                          {/* Explanation */}
                          {img.explanation && (
                            <p className="text-xs text-surface-500 dark:text-surface-400 mt-1">{img.explanation}</p>
                          )}

                          {/* Indicators */}
                          {img.indicators?.length > 0 && (
                            <div className="mt-2 space-y-1">
                              {img.indicators.map((ind, j) => (
                                <div key={j} className="flex items-start gap-2 text-xs">
                                  <AlertTriangle className={`h-3.5 w-3.5 mt-0.5 flex-shrink-0 ${ind.severity >= 3 ? 'text-red-500' : ind.severity >= 2 ? 'text-amber-500' : 'text-surface-400'}`} />
                                  <span className="text-surface-600 dark:text-surface-400">{ind.description}</span>
                                </div>
                              ))}
                            </div>
                          )}

                          {/* Top CNN Classes */}
                          {img.classification?.top_classes?.length > 0 && (
                            <div className="mt-2">
                              <p className="text-[10px] text-surface-400 mb-1">CNN Classification:</p>
                              <div className="flex flex-wrap gap-1">
                                {img.classification.top_classes.slice(0, 5).map((cls, k) => (
                                  <span key={k} className="text-[10px] px-1.5 py-0.5 rounded bg-surface-100 dark:bg-surface-700 text-surface-500">
                                    {cls.label || cls.class_name} ({(cls.probability * 100).toFixed(0)}%)
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Forensic Analysis Detail */}
                          {(img.forensics || img.confidence_calibration || img.explainability) && (
                            <div className="mt-3 p-3 rounded-lg bg-indigo-50 dark:bg-indigo-900/10 border border-indigo-200 dark:border-indigo-800">
                              <p className="text-xs font-semibold text-indigo-700 dark:text-indigo-400 flex items-center gap-1 mb-2.5">
                                <Microscope className="h-3.5 w-3.5" /> Deep Forensic Analysis
                              </p>

                              {/* Forensic score bars */}
                              {img.forensics && (
                                <div className="space-y-1.5 mb-3">
                                  {[
                                    { label: 'AI Generation Patterns', icon: <Fingerprint className="h-3 w-3" />, score: img.forensics.gan_fingerprint_score, threshold: 0.15, tip: 'The image contains patterns commonly found in AI-generated pictures.' },
                                    { label: 'AI Editing Traces', icon: <Layers className="h-3 w-3" />, score: img.forensics.diffusion_artifact_score, threshold: 0.15, tip: 'Subtle pixel patterns suggest AI editing tools may have been used.' },
                                    { label: 'Camera Signature', icon: <Activity className="h-3 w-3" />, score: img.forensics.sensor_noise_score, threshold: 0.12, tip: 'Real cameras leave unique noise patterns. This image may not show natural camera behavior.' },
                                    { label: 'Signs of Editing', icon: <Scissors className="h-3 w-3" />, score: img.forensics.tampering_score, threshold: 0.15, tip: 'Parts of the image may have been digitally altered.' },
                                    { label: 'Scene Consistency', icon: <Eye className="h-3 w-3" />, score: img.forensics.content_consistency_score != null ? (1 - img.forensics.content_consistency_score) : null, threshold: 0.15, invert: true, tip: 'We checked whether the lighting, shadows, and structure look natural.' },
                                  ].filter(s => s.score != null && s.score !== undefined).map((sig, si) => (
                                    <div key={si} className="flex items-center gap-2 group/tip relative">
                                      <span className="text-[10px] text-surface-500 w-28 flex items-center gap-1 truncate">
                                        {sig.icon} {sig.label}
                                        <span className="cursor-help text-surface-400 hover:text-indigo-500 transition-colors shrink-0">
                                          <Info className="h-2.5 w-2.5" />
                                        </span>
                                      </span>
                                      <div className="absolute left-0 bottom-full mb-1.5 z-50 hidden group-hover/tip:block w-72 max-w-xs p-2.5 text-[11px] text-white bg-surface-800 dark:bg-surface-700 rounded-lg shadow-lg leading-relaxed whitespace-normal pointer-events-none">
                                        {sig.tip}
                                      </div>
                                      <div className="flex-1 bg-surface-100 dark:bg-surface-800 rounded-full h-1.5">
                                        <div className={`h-1.5 rounded-full ${sig.score >= sig.threshold ? 'bg-red-500' : sig.score >= sig.threshold * 0.6 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                                          style={{ width: `${Math.max(sig.score * 100, 2)}%` }} />
                                      </div>
                                      <span className={`text-[10px] font-bold w-8 text-right ${sig.score >= sig.threshold ? 'text-red-600 dark:text-red-400' : 'text-surface-500'}`}>
                                        {(sig.score * 100).toFixed(0)}%
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              )}

                              {/* Duplicate detection */}
                              {img.forensics?.duplicate_listings?.length > 0 && (
                                <div className="mb-2 flex items-center gap-1 text-[10px] text-red-600 dark:text-red-400 font-medium">
                                  <AlertTriangle className="h-3 w-3" /> Image reused in {img.forensics.duplicate_listings.length} other listing{img.forensics.duplicate_listings.length !== 1 ? 's' : ''}
                                </div>
                              )}

                              {/* Perceptual hash */}
                              {img.forensics?.perceptual_hash && (
                                <p className="text-[10px] text-surface-400 mb-2">pHash: <span className="font-mono text-surface-500">{img.forensics.perceptual_hash}</span></p>
                              )}

                              {/* Confidence calibration */}
                              {img.confidence_calibration && (
                                <div className="flex items-center gap-3 mb-2 text-[10px]">
                                  <BarChart3 className="h-3 w-3 text-indigo-500" />
                                  <span className="text-surface-500">Calibrated probability:</span>
                                  <span className={`font-bold ${img.confidence_calibration.calibrated_probability >= 0.5 ? 'text-red-600 dark:text-red-400' : 'text-surface-700 dark:text-surface-300'}`}>
                                    {(img.confidence_calibration.calibrated_probability * 100).toFixed(0)}%
                                  </span>
                                  <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
                                    img.confidence_calibration.confidence_level === 'high' ? 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400' :
                                    img.confidence_calibration.confidence_level === 'medium' ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400' :
                                    'bg-surface-100 dark:bg-surface-700 text-surface-500'
                                  }`}>
                                    {img.confidence_calibration.confidence_level} confidence
                                  </span>
                                  <span className="text-surface-400">{img.confidence_calibration.active_signals}/{img.confidence_calibration.total_signals} signals active</span>
                                </div>
                              )}

                              {/* Explainability — top drivers */}
                              {img.explainability?.top_risk_drivers?.length > 0 && (
                                <div className="mt-2">
                                  <p className="text-[10px] text-surface-400 mb-1">Top risk drivers:</p>
                                  <div className="flex flex-wrap gap-1">
                                    {img.explainability.top_risk_drivers.map((drv, di) => (
                                      <span key={di} className="text-[10px] px-1.5 py-0.5 rounded bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 border border-red-200 dark:border-red-800">
                                        {drv}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {/* Explainability — reasoning chain */}
                              {img.explainability?.reasoning_chain?.length > 0 && (
                                <div className="mt-2 space-y-0.5">
                                  {img.explainability.reasoning_chain.slice(0, 4).map((reason, ri) => (
                                    <p key={ri} className="text-[10px] text-surface-500 dark:text-surface-400">• {reason}</p>
                                  ))}
                                </div>
                              )}

                              {/* Deep EXIF anomalies */}
                              {img.deep_exif?.anomalies?.length > 0 && (
                                <div className="mt-2">
                                  <p className="text-[10px] text-surface-400 mb-1">EXIF anomalies:</p>
                                  {img.deep_exif.anomalies.slice(0, 3).map((a, ai) => (
                                    <p key={ai} className="text-[10px] text-amber-600 dark:text-amber-400">⚠ {a}</p>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}

                          {/* Web Detection (Reverse Image Search) */}
                          {img.web_detection?.has_web_matches && (
                            <div className="mt-3 p-3 rounded-lg bg-orange-50 dark:bg-orange-900/10 border border-orange-200 dark:border-orange-800">
                              <p className="text-xs font-semibold text-orange-700 dark:text-orange-400 flex items-center gap-1 mb-2">
                                <Globe className="h-3.5 w-3.5" /> Reverse Image Search — {img.web_detection.total_matches} match{img.web_detection.total_matches !== 1 ? 'es' : ''} found
                              </p>
                              {img.web_detection.full_matching_images?.length > 0 && (
                                <div className="mb-1.5">
                                  <p className="text-[10px] text-surface-500 mb-0.5">Exact matches:</p>
                                  {img.web_detection.full_matching_images.slice(0, 3).map((m, mi) => (
                                    <a key={mi} href={m.url} target="_blank" rel="noopener noreferrer"
                                      className="flex items-center gap-1 text-[10px] text-blue-600 dark:text-blue-400 hover:underline truncate">
                                      <ExternalLink className="h-2.5 w-2.5 flex-shrink-0" /> {m.url}
                                    </a>
                                  ))}
                                </div>
                              )}
                              {img.web_detection.pages_with_matching_images?.length > 0 && (
                                <div className="mb-1.5">
                                  <p className="text-[10px] text-surface-500 mb-0.5">Found on pages:</p>
                                  {img.web_detection.pages_with_matching_images.slice(0, 3).map((pg, pi) => (
                                    <a key={pi} href={pg.url} target="_blank" rel="noopener noreferrer"
                                      className="flex items-center gap-1 text-[10px] text-blue-600 dark:text-blue-400 hover:underline truncate">
                                      <ExternalLink className="h-2.5 w-2.5 flex-shrink-0" /> {pg.page_title || pg.url}
                                    </a>
                                  ))}
                                </div>
                              )}
                              {img.web_detection.best_guess_labels?.length > 0 && (
                                <p className="text-[10px] text-surface-500">
                                  Google's best guess: <span className="font-medium text-surface-700 dark:text-surface-300">{img.web_detection.best_guess_labels.join(', ')}</span>
                                </p>
                              )}
                            </div>
                          )}
                          {img.web_detection && !img.web_detection.has_web_matches && (
                            <div className="mt-2 flex items-center gap-1 text-[10px] text-emerald-600 dark:text-emerald-400">
                              <Globe className="h-3 w-3" /> No web matches found — image appears original
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </LandlordLayout>
  )
}

export default PropertyVerificationNew
