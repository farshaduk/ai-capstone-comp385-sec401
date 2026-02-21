import { useState, useEffect } from 'react'
import Layout from '../../components/Layout'
import { adminAPI } from '../../services/api'
import toast from 'react-hot-toast'
import { Brain, Play, Square, Loader, CheckCircle, XCircle, TrendingUp, BarChart3, FileText, X, Trash2, Zap, Target, AlertCircle, Award, Activity } from 'lucide-react'

const Models = () => {
  const [models, setModels] = useState([])
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(true)
  const [showTrain, setShowTrain] = useState(false)
  const [trainForm, setTrainForm] = useState({ name: '', dataset_id: '' })
  const [training, setTraining] = useState(false)
  const [selectedModel, setSelectedModel] = useState(null)
  const [analysisReport, setAnalysisReport] = useState(null)
  const [loadingAnalysis, setLoadingAnalysis] = useState(false)

  // BERT state
  const [bertStatus, setBertStatus] = useState(null)
  const [showBertTrain, setShowBertTrain] = useState(false)
  const [bertTrainForm, setBertTrainForm] = useState({ epochs: 3, batch_size: 8, learning_rate: 0.00002 })
  const [bertTraining, setBertTraining] = useState(false)
  const [bertMetrics, setBertMetrics] = useState(null)

  useEffect(() => {
    fetchData()
    fetchBertStatus()
  }, [])

  const fetchData = async () => {
    try {
      const [modelsRes, datasetsRes] = await Promise.all([
        adminAPI.getModels(),
        adminAPI.getDatasets()
      ])
      setModels(modelsRes.data)
      setDatasets(datasetsRes.data)
    } catch (error) {
      toast.error('Failed to fetch data')
    } finally {
      setLoading(false)
    }
  }

  const fetchBertStatus = async () => {
    try {
      const res = await adminAPI.getBertStatus()
      setBertStatus(res.data)
    } catch (error) {
      setBertStatus({ bert_available: false, status: 'unavailable' })
    }
  }

  const handleBertTrain = async (e) => {
    e.preventDefault()
    setBertTraining(true)
    setBertMetrics(null)
    try {
      const formData = new FormData()
      formData.append('epochs', bertTrainForm.epochs)
      formData.append('batch_size', bertTrainForm.batch_size)
      formData.append('learning_rate', bertTrainForm.learning_rate)
      const res = await adminAPI.trainBert(formData)
      setBertMetrics(res.data.metrics || res.data)
      toast.success('BERT model training complete!')
      setShowBertTrain(false)
      fetchBertStatus()
    } catch (error) {
      toast.error(error.response?.data?.detail || 'BERT training failed')
    } finally {
      setBertTraining(false)
    }
  }

  const handleTrain = async (e) => {
    e.preventDefault()
    setTraining(true)

    try {
      await adminAPI.trainModel(trainForm)
      toast.success('Model training started!')
      setShowTrain(false)
      setTrainForm({ name: '', dataset_id: '' })
      fetchData()
    } catch (error) {
      toast.error('Failed to start training')
    } finally {
      setTraining(false)
    }
  }

  const handleActivate = async (id) => {
    try {
      await adminAPI.activateModel(id)
      toast.success('Model activated!')
      fetchData()
    } catch (error) {
      toast.error('Failed to activate model')
    }
  }

  const handleDeactivate = async (id) => {
    try {
      await adminAPI.deactivateModel(id)
      toast.success('Model deactivated!')
      fetchData()
    } catch (error) {
      toast.error('Failed to deactivate model')
    }
  }

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
      return
    }

    try {
      await adminAPI.deleteModel(id)
      toast.success('Model deleted successfully!')
      fetchData()
    } catch (error) {
      toast.error('Failed to delete model')
    }
  }

  const viewAnalysis = async (model) => {
    setSelectedModel(model)
    setLoadingAnalysis(true)
    try {
      const response = await adminAPI.getModelAnalysis(model.id)
      setAnalysisReport(response.data)
    } catch (error) {
      toast.error('Failed to load analysis report')
      setSelectedModel(null)
    } finally {
      setLoadingAnalysis(false)
    }
  }

  const closeAnalysis = () => {
    setSelectedModel(null)
    setAnalysisReport(null)
  }

  const getStatusBadge = (status) => {
    const badges = {
      training: 'badge bg-blue-100 text-blue-800',
      completed: 'badge bg-green-100 text-green-800',
      active: 'badge bg-emerald-100 text-emerald-800',
      inactive: 'badge bg-gray-100 text-gray-800',
      failed: 'badge bg-red-100 text-red-800'
    }
    return badges[status] || badges.inactive
  }

  return (
    <Layout title="Model Management">
      <div className="space-y-6">
        <div className="flex space-x-3">
          <button onClick={() => setShowTrain(true)} className="btn-primary">
            <Brain className="inline h-5 w-5 mr-2" />
            Train Isolation Forest
          </button>
          <button onClick={() => setShowBertTrain(true)} className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition">
            <Zap className="inline h-5 w-5 mr-2" />
            Train BERT Classifier
          </button>
        </div>

        {/* BERT Status Card */}
        {bertStatus && (
          <div className="card border-l-4 border-purple-500">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                  bertStatus.is_trained ? 'bg-green-100' : bertStatus.bert_available ? 'bg-yellow-100' : 'bg-red-100'
                }`}>
                  <Zap className={`h-6 w-6 ${
                    bertStatus.is_trained ? 'text-green-600' : bertStatus.bert_available ? 'text-yellow-600' : 'text-red-600'
                  }`} />
                </div>
                <div>
                  <h3 className="font-bold text-lg">BERT Fraud Classifier</h3>
                  <p className="text-sm text-gray-500">
                    {bertStatus.model_name || 'DistilBERT'} — {bertStatus.is_trained ? 'Trained & Ready' : bertStatus.bert_available ? 'Available (Not Trained)' : 'Dependencies Missing'}
                  </p>
                </div>
              </div>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                bertStatus.is_trained ? 'bg-green-100 text-green-800' : bertStatus.bert_available ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800'
              }`}>
                {bertStatus.status || 'unknown'}
              </span>
            </div>

            {/* BERT Metrics Display */}
            {bertMetrics && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <h4 className="font-semibold mb-3 flex items-center">
                  <TrendingUp className="h-5 w-5 text-purple-600 mr-2" />
                  BERT Training Results
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  {bertMetrics.test_accuracy != null && (
                    <div className="bg-purple-50 rounded-lg p-3 text-center">
                      <p className="text-xs text-gray-500">Accuracy</p>
                      <p className="text-xl font-bold text-purple-700">{(bertMetrics.test_accuracy * 100).toFixed(1)}%</p>
                    </div>
                  )}
                  {bertMetrics.test_precision != null && (
                    <div className="bg-blue-50 rounded-lg p-3 text-center">
                      <p className="text-xs text-gray-500">Precision</p>
                      <p className="text-xl font-bold text-blue-700">{(bertMetrics.test_precision * 100).toFixed(1)}%</p>
                    </div>
                  )}
                  {bertMetrics.test_recall != null && (
                    <div className="bg-green-50 rounded-lg p-3 text-center">
                      <p className="text-xs text-gray-500">Recall</p>
                      <p className="text-xl font-bold text-green-700">{(bertMetrics.test_recall * 100).toFixed(1)}%</p>
                    </div>
                  )}
                  {bertMetrics.test_f1 != null && (
                    <div className="bg-indigo-50 rounded-lg p-3 text-center">
                      <p className="text-xs text-gray-500">F1 Score</p>
                      <p className="text-xl font-bold text-indigo-700">{(bertMetrics.test_f1 * 100).toFixed(1)}%</p>
                    </div>
                  )}
                  {bertMetrics.test_roc_auc != null && (
                    <div className="bg-amber-50 rounded-lg p-3 text-center">
                      <p className="text-xs text-gray-500">ROC AUC</p>
                      <p className="text-xl font-bold text-amber-700">{(bertMetrics.test_roc_auc * 100).toFixed(1)}%</p>
                    </div>
                  )}
                </div>
                {bertMetrics.training_time_seconds && (
                  <p className="text-sm text-gray-500 mt-2">Training time: {bertMetrics.training_time_seconds.toFixed(1)}s | Best epoch: {bertMetrics.best_epoch}</p>
                )}
              </div>
            )}
          </div>
        )}

        {loading ? (
          <div className="flex justify-center p-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6">
            {models.map((model) => (
              <div key={model.id} className="card">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h3 className="font-bold text-xl">{model.name}</h3>
                      <span className={getStatusBadge(model.status)}>{model.status}</span>
                      {model.is_active && <span className="badge-success">ACTIVE IN PRODUCTION</span>}
                    </div>
                    <p className="text-gray-600 text-sm">Version: {model.version}</p>
                    <p className="text-gray-500 text-xs mt-1">
                      Trained: {new Date(model.created_at).toLocaleString()}
                    </p>
                  </div>
                  <div className="flex space-x-2">
                    {(model.status === 'completed' || model.status === 'active' || model.status === 'inactive') && (
                      <button
                        onClick={() => viewAnalysis(model)}
                        className="py-2 px-4 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200"
                      >
                        <BarChart3 className="inline h-4 w-4 mr-1" />
                        View Analysis
                      </button>
                    )}
                    {!model.is_active && (model.status === 'completed' || model.status === 'inactive') && (
                      <button
                        onClick={() => handleActivate(model.id)}
                        className="btn-primary py-2"
                      >
                        <Play className="inline h-4 w-4 mr-1" />
                        Activate
                      </button>
                    )}
                    {model.is_active && (
                      <button
                        onClick={() => handleDeactivate(model.id)}
                        className="py-2 px-4 bg-orange-100 text-orange-700 rounded-lg hover:bg-orange-200"
                      >
                        <Square className="inline h-4 w-4 mr-1" />
                        Deactivate
                      </button>
                    )}
                    {!model.is_active && model.status !== 'training' && (
                      <button
                        onClick={() => handleDelete(model.id)}
                        className="py-2 px-4 bg-red-100 text-red-700 rounded-lg hover:bg-red-200"
                      >
                        <Trash2 className="inline h-4 w-4 mr-1" />
                        Delete
                      </button>
                    )}
                  </div>
                </div>

                {model.metrics && Object.keys(model.metrics).length > 0 && (
                  <div className="bg-gray-50 rounded-lg p-4 mt-4">
                    <h4 className="font-semibold mb-3 flex items-center">
                      <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
                      Performance Metrics
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {model.metrics.accuracy && (
                        <div>
                          <p className="text-sm text-gray-600">Accuracy</p>
                          <p className="text-2xl font-bold text-green-600">
                            {(model.metrics.accuracy * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                      {model.metrics.precision && (
                        <div>
                          <p className="text-sm text-gray-600">Precision</p>
                          <p className="text-2xl font-bold text-blue-600">
                            {(model.metrics.precision * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                      {model.metrics.recall && (
                        <div>
                          <p className="text-sm text-gray-600">Recall</p>
                          <p className="text-2xl font-bold text-purple-600">
                            {(model.metrics.recall * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                      {model.metrics.f1_score && (
                        <div>
                          <p className="text-sm text-gray-600">F1 Score</p>
                          <p className="text-2xl font-bold text-indigo-600">
                            {(model.metrics.f1_score * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                    </div>

                    {model.metrics.n_samples && (
                      <div className="mt-4 pt-4 border-t border-gray-200">
                        <p className="text-sm text-gray-600">
                          Trained on <span className="font-semibold">{model.metrics.n_samples}</span> samples
                          with <span className="font-semibold">{model.metrics.n_features}</span> features
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Train Modal */}
      {showTrain && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl max-w-md w-full p-6">
            <h3 className="text-2xl font-bold mb-4">Train New Model</h3>
            <form onSubmit={handleTrain} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Model Name</label>
                <input
                  type="text"
                  value={trainForm.name}
                  onChange={(e) => setTrainForm({ ...trainForm, name: e.target.value })}
                  className="input-field"
                  placeholder="Fraud Detection Model v1"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Training Dataset</label>
                <select
                  value={trainForm.dataset_id}
                  onChange={(e) => setTrainForm({ ...trainForm, dataset_id: parseInt(e.target.value) })}
                  className="input-field"
                  required
                >
                  <option value="">Select dataset...</option>
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name} ({d.record_count} records)
                    </option>
                  ))}
                </select>
              </div>
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                <p className="text-sm text-yellow-800">
                  ⚠️ Model training may take several minutes depending on dataset size.
                </p>
              </div>
              <div className="flex space-x-2">
                <button type="submit" disabled={training} className="btn-primary flex-1">
                  {training ? (
                    <>
                      <Loader className="inline h-4 w-4 mr-2 animate-spin" />
                      Training...
                    </>
                  ) : (
                    'Start Training'
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => setShowTrain(false)}
                  className="btn-secondary flex-1"
                  disabled={training}
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* BERT Train Modal */}
      {showBertTrain && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl max-w-md w-full p-6">
            <h3 className="text-2xl font-bold mb-2">Train BERT Classifier</h3>
            <p className="text-sm text-gray-500 mb-4">Fine-tune DistilBERT for rental fraud text classification</p>
            <form onSubmit={handleBertTrain} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Epochs</label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={bertTrainForm.epochs}
                  onChange={(e) => setBertTrainForm({ ...bertTrainForm, epochs: parseInt(e.target.value) })}
                  className="input-field"
                  required
                />
                <p className="text-xs text-gray-400 mt-1">Recommended: 3-5 for small datasets</p>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Batch Size</label>
                <select
                  value={bertTrainForm.batch_size}
                  onChange={(e) => setBertTrainForm({ ...bertTrainForm, batch_size: parseInt(e.target.value) })}
                  className="input-field"
                >
                  <option value={4}>4</option>
                  <option value={8}>8 (recommended)</option>
                  <option value={16}>16</option>
                  <option value={32}>32</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Learning Rate</label>
                <select
                  value={bertTrainForm.learning_rate}
                  onChange={(e) => setBertTrainForm({ ...bertTrainForm, learning_rate: parseFloat(e.target.value) })}
                  className="input-field"
                >
                  <option value={0.00001}>1e-5 (conservative)</option>
                  <option value={0.00002}>2e-5 (recommended)</option>
                  <option value={0.00003}>3e-5</option>
                  <option value={0.00005}>5e-5 (aggressive)</option>
                </select>
              </div>
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
                <p className="text-sm text-purple-800">
                  <Zap className="inline h-4 w-4 mr-1" />
                  BERT training requires PyTorch and transformers. Training may take several minutes on CPU.
                </p>
              </div>
              <div className="flex space-x-2">
                <button type="submit" disabled={bertTraining} className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50">
                  {bertTraining ? (
                    <>
                      <Loader className="inline h-4 w-4 mr-2 animate-spin" />
                      Training BERT...
                    </>
                  ) : (
                    'Start BERT Training'
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => setShowBertTrain(false)}
                  className="btn-secondary flex-1"
                  disabled={bertTraining}
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Analysis Report Modal */}
      {selectedModel && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-white rounded-xl max-w-6xl w-full my-8">
            <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between rounded-t-xl">
              <div className="flex items-center space-x-3">
                <BarChart3 className="h-6 w-6 text-primary-600" />
                <h3 className="text-2xl font-bold">Comprehensive Model Analysis</h3>
              </div>
              <button onClick={closeAnalysis} className="text-gray-400 hover:text-gray-600">
                <X className="h-6 w-6" />
              </button>
            </div>

            <div className="p-6 max-h-[80vh] overflow-y-auto">
              {loadingAnalysis ? (
                <div className="flex justify-center items-center py-12">
                  <Loader className="h-12 w-12 animate-spin text-primary-600" />
                </div>
              ) : analysisReport && (
                <div className="space-y-6">
                  {/* Model Overview */}
                  <div className="bg-gradient-to-r from-primary-50 to-blue-50 rounded-lg p-6">
                    <h4 className="text-xl font-bold mb-4 flex items-center">
                      <Brain className="h-6 w-6 mr-2 text-primary-600" />
                      Model Overview
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <p className="text-sm text-gray-600">Model Name</p>
                        <p className="font-semibold">{analysisReport.name}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Version</p>
                        <p className="font-semibold">{analysisReport.version}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Status</p>
                        <span className={getStatusBadge(analysisReport.status)}>{analysisReport.status}</span>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Created</p>
                        <p className="font-semibold text-sm">{new Date(analysisReport.created_at).toLocaleDateString()}</p>
                      </div>
                    </div>
                  </div>

                  {/* Algorithm Details */}
                  <div className="card">
                    <h4 className="text-xl font-bold mb-4 flex items-center">
                      <FileText className="h-6 w-6 mr-2 text-purple-600" />
                      Algorithm Details
                    </h4>
                    <div className="space-y-4">
                      <div>
                        <p className="text-sm text-gray-600 mb-1">Algorithm</p>
                        <p className="text-lg font-bold text-purple-600">{analysisReport.algorithm_details.algorithm_name}</p>
                        <p className="text-sm text-gray-700 mt-1">{analysisReport.algorithm_details.algorithm_type}</p>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-4">
                        <p className="text-sm text-gray-700">{analysisReport.algorithm_details.algorithm_description}</p>
                      </div>
                      <div>
                        <p className="font-semibold mb-2">Use Case:</p>
                        <p className="text-sm text-gray-700">{analysisReport.algorithm_details.use_case}</p>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <p className="font-semibold mb-2 text-green-700">✓ Advantages:</p>
                          <ul className="text-sm text-gray-700 space-y-1">
                            {analysisReport.algorithm_details.advantages.map((adv, idx) => (
                              <li key={idx} className="flex items-start">
                                <span className="text-green-600 mr-2">•</span>
                                {adv}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <p className="font-semibold mb-2 text-orange-700">⚠ Limitations:</p>
                          <ul className="text-sm text-gray-700 space-y-1">
                            {analysisReport.algorithm_details.limitations.map((lim, idx) => (
                              <li key={idx} className="flex items-start">
                                <span className="text-orange-600 mr-2">•</span>
                                {lim}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                      <div>
                        <p className="font-semibold mb-2">Hyperparameters:</p>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                          {Object.entries(analysisReport.algorithm_details.hyperparameters).map(([key, value]) => (
                            <div key={key} className="bg-gray-50 rounded p-2">
                              <p className="text-xs text-gray-600">{key}</p>
                              <p className="text-sm font-semibold">{String(value)}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Performance Metrics */}
                  <div className="card">
                    <h4 className="text-xl font-bold mb-4 flex items-center">
                      <TrendingUp className="h-6 w-6 mr-2 text-green-600" />
                      Performance Metrics
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                      <div className="bg-green-50 rounded-lg p-4 text-center">
                        <p className="text-sm text-gray-600 mb-1">Accuracy</p>
                        <p className="text-3xl font-bold text-green-600">
                          {(analysisReport.performance_metrics.accuracy * 100).toFixed(1)}%
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          {analysisReport.performance_metrics.interpretation.accuracy}
                        </p>
                      </div>
                      <div className="bg-blue-50 rounded-lg p-4 text-center">
                        <p className="text-sm text-gray-600 mb-1">Precision</p>
                        <p className="text-3xl font-bold text-blue-600">
                          {(analysisReport.performance_metrics.precision * 100).toFixed(1)}%
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          {analysisReport.performance_metrics.interpretation.precision}
                        </p>
                      </div>
                      <div className="bg-purple-50 rounded-lg p-4 text-center">
                        <p className="text-sm text-gray-600 mb-1">Recall</p>
                        <p className="text-3xl font-bold text-purple-600">
                          {(analysisReport.performance_metrics.recall * 100).toFixed(1)}%
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          {analysisReport.performance_metrics.interpretation.recall}
                        </p>
                      </div>
                      <div className="bg-indigo-50 rounded-lg p-4 text-center">
                        <p className="text-sm text-gray-600 mb-1">F1 Score</p>
                        <p className="text-3xl font-bold text-indigo-600">
                          {(analysisReport.performance_metrics.f1_score * 100).toFixed(1)}%
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          {analysisReport.performance_metrics.interpretation.f1_score}
                        </p>
                      </div>
                    </div>
                    {analysisReport.performance_metrics.confusion_matrix && 
                     analysisReport.performance_metrics.confusion_matrix.length > 0 && (
                      <div className="bg-gray-50 rounded-lg p-4">
                        <p className="font-semibold mb-3">Confusion Matrix:</p>
                        <div className="overflow-x-auto">
                          <table className="min-w-full text-center">
                            <thead>
                              <tr>
                                <th className="p-2"></th>
                                <th className="p-2 text-sm font-semibold">Predicted Normal</th>
                                <th className="p-2 text-sm font-semibold">Predicted Fraud</th>
                              </tr>
                            </thead>
                            <tbody>
                              <tr>
                                <td className="p-2 text-sm font-semibold">Actual Normal</td>
                                <td className="p-2 bg-green-100 font-bold">{analysisReport.performance_metrics.confusion_matrix[0]?.[0] || 0}</td>
                                <td className="p-2 bg-red-100 font-bold">{analysisReport.performance_metrics.confusion_matrix[0]?.[1] || 0}</td>
                              </tr>
                              <tr>
                                <td className="p-2 text-sm font-semibold">Actual Fraud</td>
                                <td className="p-2 bg-red-100 font-bold">{analysisReport.performance_metrics.confusion_matrix[1]?.[0] || 0}</td>
                                <td className="p-2 bg-green-100 font-bold">{analysisReport.performance_metrics.confusion_matrix[1]?.[1] || 0}</td>
                              </tr>
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Training Details */}
                  <div className="card">
                    <h4 className="text-xl font-bold mb-4">Training Details</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                      <div className="bg-blue-50 rounded-lg p-4">
                        <p className="text-sm text-gray-600">Training Samples</p>
                        <p className="text-2xl font-bold text-blue-600">{analysisReport.training_details.n_samples.toLocaleString()}</p>
                      </div>
                      <div className="bg-purple-50 rounded-lg p-4">
                        <p className="text-sm text-gray-600">Features Engineered</p>
                        <p className="text-2xl font-bold text-purple-600">{analysisReport.training_details.n_features}</p>
                      </div>
                      <div className="bg-green-50 rounded-lg p-4">
                        <p className="text-sm text-gray-600">Training Status</p>
                        <p className="text-2xl font-bold text-green-600">{analysisReport.training_details.training_time}</p>
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div>
                        <p className="font-semibold mb-2">Feature Engineering:</p>
                        <div className="flex flex-wrap gap-2">
                          {analysisReport.training_details.feature_engineering.map((feature, idx) => (
                            <span key={idx} className="badge bg-blue-100 text-blue-800">{feature}</span>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="font-semibold mb-2">Preprocessing Steps:</p>
                        <div className="flex flex-wrap gap-2">
                          {analysisReport.training_details.preprocessing.map((step, idx) => (
                            <span key={idx} className="badge bg-purple-100 text-purple-800">{step}</span>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="font-semibold mb-2">Model Artifacts:</p>
                        <div className="flex flex-wrap gap-2">
                          {analysisReport.training_details.model_artifacts.map((artifact, idx) => (
                            <span key={idx} className="badge bg-gray-100 text-gray-800">{artifact}</span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Visualizations */}
                  {analysisReport.visualizations && Object.keys(analysisReport.visualizations).length > 0 && (
                    <div className="card">
                      <h4 className="text-xl font-bold mb-4">Visualizations</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {Object.entries(analysisReport.visualizations).map(([name, path]) => (
                          <div key={name} className="bg-gray-50 rounded-lg p-4">
                            <p className="font-semibold mb-2 capitalize">{name.replace(/_/g, ' ')}</p>
                            <div className="bg-white rounded border border-gray-200 p-2">
                              <img 
                                src={`http://localhost:8000/${path}`} 
                                alt={name}
                                className="w-full h-auto"
                                onError={(e) => {
                                  e.target.style.display = 'none'
                                  e.target.nextSibling.style.display = 'block'
                                }}
                              />
                              <p className="text-sm text-gray-500 text-center py-4" style={{display: 'none'}}>
                                Visualization not available
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* ROC & PR Curves */}
                  {(analysisReport.visualizations.roc_curve || analysisReport.visualizations.precision_recall_curve) && (
                    <div className="card">
                      <h4 className="text-xl font-bold mb-4">ROC & Precision-Recall Curves</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {analysisReport.visualizations.roc_curve && (
                          <div>
                            <p className="font-semibold mb-2">ROC Curve (AUC: {(analysisReport.performance_metrics.roc_auc * 100).toFixed(1)}%)</p>
                            <img src={`http://localhost:8000/${analysisReport.visualizations.roc_curve}`} alt="ROC Curve" className="w-full rounded border" />
                          </div>
                        )}
                        {analysisReport.visualizations.precision_recall_curve && (
                          <div>
                            <p className="font-semibold mb-2">Precision-Recall Curve (AUC: {(analysisReport.performance_metrics.pr_auc * 100).toFixed(1)}%)</p>
                            <img src={`http://localhost:8000/${analysisReport.visualizations.precision_recall_curve}`} alt="PR Curve" className="w-full rounded border" />
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Threshold Analysis */}
                  {analysisReport.threshold_analysis && analysisReport.threshold_analysis.length > 0 && (
                    <div className="card">
                      <h4 className="text-xl font-bold mb-4 flex items-center">
                        <Target className="h-6 w-6 mr-2 text-orange-600" />
                        Threshold Analysis
                      </h4>
                      {analysisReport.visualizations.threshold_analysis && (
                        <img src={`http://localhost:8000/${analysisReport.visualizations.threshold_analysis}`} alt="Threshold Analysis" className="w-full rounded border mb-4" />
                      )}
                      <div className="overflow-x-auto">
                        <table className="min-w-full text-sm">
                          <thead className="bg-gray-100">
                            <tr>
                              <th className="px-4 py-2 text-left">Threshold</th>
                              <th className="px-4 py-2 text-left">Precision</th>
                              <th className="px-4 py-2 text-left">Recall</th>
                              <th className="px-4 py-2 text-left">F1 Score</th>
                            </tr>
                          </thead>
                          <tbody>
                            {analysisReport.threshold_analysis.map((t, idx) => (
                              <tr key={idx} className="border-b hover:bg-gray-50">
                                <td className="px-4 py-2 font-medium">{t.threshold.toFixed(3)}</td>
                                <td className="px-4 py-2">{(t.precision * 100).toFixed(1)}%</td>
                                <td className="px-4 py-2">{(t.recall * 100).toFixed(1)}%</td>
                                <td className="px-4 py-2">{(t.f1_score * 100).toFixed(1)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Performance Benchmarks */}
                  {analysisReport.performance_benchmarks && (
                    <div className="card">
                      <h4 className="text-xl font-bold mb-4 flex items-center">
                        <Zap className="h-6 w-6 mr-2 text-yellow-600" />
                        Performance Benchmarks
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="bg-yellow-50 rounded-lg p-4 text-center">
                          <p className="text-sm text-gray-600 mb-1">Inference Time</p>
                          <p className="text-3xl font-bold text-yellow-600">{analysisReport.performance_benchmarks.inference_time_ms.toFixed(2)} ms</p>
                          <p className="text-xs text-gray-500 mt-1">Per prediction</p>
                        </div>
                        <div className="bg-blue-50 rounded-lg p-4 text-center">
                          <p className="text-sm text-gray-600 mb-1">Throughput</p>
                          <p className="text-3xl font-bold text-blue-600">{Math.round(analysisReport.performance_benchmarks.throughput_per_second)}</p>
                          <p className="text-xs text-gray-500 mt-1">Predictions/second</p>
                        </div>
                        <div className="bg-purple-50 rounded-lg p-4 text-center">
                          <p className="text-sm text-gray-600 mb-1">Model Size</p>
                          <p className="text-3xl font-bold text-purple-600">{analysisReport.performance_benchmarks.model_size_mb.toFixed(2)} MB</p>
                          <p className="text-xs text-gray-500 mt-1">Disk space</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Business Impact */}
                  {analysisReport.business_impact && (
                    <div className="card">
                      <h4 className="text-xl font-bold mb-4 flex items-center">
                        <Activity className="h-6 w-6 mr-2 text-green-600" />
                        Business Impact Analysis
                      </h4>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
                        <div className="bg-green-50 rounded-lg p-4">
                          <p className="text-sm text-gray-600">Fraud Detection Rate</p>
                          <p className="text-2xl font-bold text-green-600">{(analysisReport.business_impact.fraud_detection_rate * 100).toFixed(1)}%</p>
                        </div>
                        <div className="bg-red-50 rounded-lg p-4">
                          <p className="text-sm text-gray-600">False Alarm Rate</p>
                          <p className="text-2xl font-bold text-red-600">{(analysisReport.business_impact.false_alarm_rate * 100).toFixed(1)}%</p>
                        </div>
                        <div className="bg-blue-50 rounded-lg p-4">
                          <p className="text-sm text-gray-600">Recommended Threshold</p>
                          <p className="text-2xl font-bold text-blue-600">{analysisReport.business_impact.recommended_threshold.toFixed(3)}</p>
                        </div>
                      </div>
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <p className="font-semibold text-blue-800 mb-2">Deployment Recommendation:</p>
                        <p className="text-blue-700">{analysisReport.business_impact.deployment_recommendation}</p>
                      </div>
                    </div>
                  )}

                  {/* Model Comparison */}
                  {analysisReport.model_comparison && analysisReport.model_comparison.length > 1 && (
                    <div className="card">
                      <h4 className="text-xl font-bold mb-4 flex items-center">
                        <Award className="h-6 w-6 mr-2 text-yellow-600" />
                        Model Comparison
                      </h4>
                      <div className="overflow-x-auto">
                        <table className="min-w-full text-sm">
                          <thead className="bg-gray-100">
                            <tr>
                              <th className="px-4 py-2 text-left">Model</th>
                              <th className="px-4 py-2 text-left">Version</th>
                              <th className="px-4 py-2 text-left">Accuracy</th>
                              <th className="px-4 py-2 text-left">F1 Score</th>
                              <th className="px-4 py-2 text-left">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {analysisReport.model_comparison.slice(0, 5).map((m, idx) => (
                              <tr key={m.id} className={`border-b hover:bg-gray-50 ${m.id === analysisReport.id ? 'bg-blue-50' : ''}`}>
                                <td className="px-4 py-2 font-medium">{m.name} {m.id === analysisReport.id && '(Current)'}</td>
                                <td className="px-4 py-2">{m.version}</td>
                                <td className="px-4 py-2">{(m.accuracy * 100).toFixed(1)}%</td>
                                <td className="px-4 py-2">{(m.f1_score * 100).toFixed(1)}%</td>
                                <td className="px-4 py-2">
                                  {m.is_active && <span className="badge bg-green-100 text-green-800">Active</span>}
                                  {idx === 0 && !m.is_active && <span className="badge bg-yellow-100 text-yellow-800">Best</span>}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Deployment Readiness */}
                  {analysisReport.deployment_readiness && (
                    <div className="card">
                      <h4 className="text-xl font-bold mb-4 flex items-center">
                        <CheckCircle className="h-6 w-6 mr-2 text-green-600" />
                        Deployment Readiness Checklist
                      </h4>
                      <div className="space-y-3">
                        {analysisReport.deployment_readiness.map((item, idx) => (
                          <div key={idx} className={`flex items-start space-x-3 p-3 rounded-lg ${item.status ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
                            {item.status ? (
                              <CheckCircle className="h-5 w-5 text-green-600 mt-0.5" />
                            ) : (
                              <XCircle className="h-5 w-5 text-red-600 mt-0.5" />
                            )}
                            <div className="flex-1">
                              <p className="font-semibold">{item.item}</p>
                              <p className="text-sm text-gray-600">{item.details}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Error Analysis */}
                  {analysisReport.error_analysis && (
                    <div className="card">
                      <h4 className="text-xl font-bold mb-4 flex items-center">
                        <AlertCircle className="h-6 w-6 mr-2 text-orange-600" />
                        Error Analysis & Improvement Suggestions
                      </h4>
                      <div className="space-y-4">
                        <div className="bg-orange-50 rounded-lg p-4">
                          <p className="text-sm text-gray-600">Total Errors</p>
                          <p className="text-2xl font-bold text-orange-600">{analysisReport.error_analysis.total_errors.toLocaleString()}</p>
                        </div>
                        <div>
                          <p className="font-semibold mb-2">Common Error Patterns:</p>
                          <ul className="space-y-1">
                            {analysisReport.error_analysis.common_patterns.map((pattern, idx) => (
                              <li key={idx} className="flex items-start">
                                <span className="text-orange-600 mr-2">•</span>
                                <span className="text-sm">{pattern}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <p className="font-semibold mb-2">Improvement Suggestions:</p>
                          <div className="space-y-2">
                            {analysisReport.error_analysis.improvement_suggestions.map((suggestion, idx) => (
                              <div key={idx} className="bg-blue-50 border border-blue-200 rounded p-2 text-sm">
                                {suggestion}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Monitoring Recommendations */}
                  {analysisReport.monitoring_recommendations && (
                    <div className="card">
                      <h4 className="text-xl font-bold mb-4">Monitoring & Maintenance Recommendations</h4>
                      <div className="space-y-4">
                        <div>
                          <p className="font-semibold mb-2">Key Metrics to Monitor:</p>
                          <div className="flex flex-wrap gap-2">
                            {analysisReport.monitoring_recommendations.key_metrics.map((metric, idx) => (
                              <span key={idx} className="badge bg-purple-100 text-purple-800">{metric}</span>
                            ))}
                          </div>
                        </div>
                        <div>
                          <p className="font-semibold mb-2">Alert Thresholds:</p>
                          <div className="space-y-2">
                            {Object.entries(analysisReport.monitoring_recommendations.alert_thresholds).map(([key, value]) => (
                              <div key={key} className="bg-yellow-50 border border-yellow-200 rounded p-2 text-sm">
                                <span className="font-medium">{key.replace(/_/g, ' ')}:</span> {value}
                              </div>
                            ))}
                          </div>
                        </div>
                        <div>
                          <p className="font-semibold mb-2">Retraining Triggers:</p>
                          <ul className="space-y-1">
                            {analysisReport.monitoring_recommendations.retraining_triggers.map((trigger, idx) => (
                              <li key={idx} className="flex items-start">
                                <span className="text-blue-600 mr-2">→</span>
                                <span className="text-sm">{trigger}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* A/B Test Recommendations */}
                  {analysisReport.ab_test_recommendations && (
                    <div className="card">
                      <h4 className="text-xl font-bold mb-4">A/B Testing Recommendations</h4>
                      <div className="space-y-3">
                        <div className="bg-blue-50 rounded-lg p-4">
                          <p className="font-semibold mb-1">Test Scenario:</p>
                          <p className="text-sm">{analysisReport.ab_test_recommendations.test_scenario}</p>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-gray-50 rounded-lg p-3">
                            <p className="text-sm text-gray-600">Duration</p>
                            <p className="font-semibold">{analysisReport.ab_test_recommendations.duration}</p>
                          </div>
                          <div className="bg-gray-50 rounded-lg p-3">
                            <p className="text-sm text-gray-600">Sample Size</p>
                            <p className="font-semibold">{analysisReport.ab_test_recommendations.sample_size}</p>
                          </div>
                        </div>
                        <div>
                          <p className="font-semibold mb-2">Success Metrics:</p>
                          <div className="flex flex-wrap gap-2">
                            {analysisReport.ab_test_recommendations.success_metrics.map((metric, idx) => (
                              <span key={idx} className="badge bg-green-100 text-green-800">{metric}</span>
                            ))}
                          </div>
                        </div>
                        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                          <p className="font-semibold text-red-800 mb-1">Risk Mitigation:</p>
                          <p className="text-sm text-red-700">{analysisReport.ab_test_recommendations.risk_mitigation}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Feature Importance */}
                  {analysisReport.feature_importance && analysisReport.feature_importance.length > 0 && (
                    <div className="card">
                      <h4 className="text-xl font-bold mb-4">Feature Importance</h4>
                      <div className="space-y-2">
                        {analysisReport.feature_importance.map((feat, idx) => (
                          <div key={idx} className="flex items-center space-x-2">
                            <span className="text-sm font-medium w-24">{feat.feature}</span>
                            <div className="flex-1 bg-gray-200 rounded-full h-4">
                              <div className="bg-blue-600 h-4 rounded-full" style={{width: `${feat.importance * 100}%`}}></div>
                            </div>
                            <span className="text-sm text-gray-600 w-16 text-right">{(feat.importance * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Additional Visualizations */}
                  {(analysisReport.visualizations.score_distribution_by_class || analysisReport.visualizations.correlation_heatmap) && (
                    <div className="card">
                      <h4 className="text-xl font-bold mb-4">Additional Analysis</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {analysisReport.visualizations.score_distribution_by_class && (
                          <div>
                            <p className="font-semibold mb-2">Score Distribution by Class</p>
                            <img src={`http://localhost:8000/${analysisReport.visualizations.score_distribution_by_class}`} alt="Score Distribution" className="w-full rounded border" />
                          </div>
                        )}
                        {analysisReport.visualizations.correlation_heatmap && (
                          <div>
                            <p className="font-semibold mb-2">Feature Correlation Heatmap</p>
                            <img src={`http://localhost:8000/${analysisReport.visualizations.correlation_heatmap}`} alt="Correlation Heatmap" className="w-full rounded border" />
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Dataset Info */}
                  {analysisReport.dataset_info && analysisReport.dataset_info.dataset_id && (
                    <div className="card">
                      <h4 className="text-xl font-bold mb-4">Training Dataset Information</h4>
                      <div className="space-y-3">
                        <div>
                          <p className="text-sm text-gray-600">Dataset Name</p>
                          <p className="font-semibold">{analysisReport.dataset_info.dataset_name}</p>
                        </div>
                        {analysisReport.dataset_info.dataset_description && (
                          <div>
                            <p className="text-sm text-gray-600">Description</p>
                            <p className="text-sm">{analysisReport.dataset_info.dataset_description}</p>
                          </div>
                        )}
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-blue-50 rounded-lg p-3">
                            <p className="text-sm text-gray-600">Records</p>
                            <p className="text-xl font-bold text-blue-600">{analysisReport.dataset_info.record_count.toLocaleString()}</p>
                          </div>
                          <div className="bg-purple-50 rounded-lg p-3">
                            <p className="text-sm text-gray-600">Columns</p>
                            <p className="text-xl font-bold text-purple-600">{analysisReport.dataset_info.column_count}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </Layout>
  )
}

export default Models

