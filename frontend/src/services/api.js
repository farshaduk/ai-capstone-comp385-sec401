import axios from 'axios'
import { useAuthStore } from '../store/authStore'
import toast from 'react-hot-toast'

const API_URL = 'http://localhost:8000/api'

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    const token = useAuthStore.getState().token
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout()
      window.location.href = '/login'
      toast.error('Session expired. Please login again.')
    }
    return Promise.reject(error)
  }
)

// Auth API
export const authAPI = {
  login: (email, password) => api.post('/auth/login', { email, password }),
  register: (data) => api.post('/auth/register', data),
  getMe: () => api.get('/auth/me'),
}

// Admin API
export const adminAPI = {
  // Dashboard
  getDashboard: () => api.get('/admin/dashboard'),
  
  // Datasets (system dataset discovery + reports)
  getDatasets: () => api.get('/admin/datasets'),
  getDatasetReport: (datasetId) => api.get(`/admin/datasets/${datasetId}/report`),
  
  // Users
  getUsers: (role = null) => api.get('/admin/users', { params: { role } }),
  getUser: (id) => api.get(`/admin/users/${id}`),
  updateUser: (id, data) => api.patch(`/admin/users/${id}`, data),
  deactivateUser: (id) => api.post(`/admin/users/${id}/deactivate`),
  
  // Settings
  getSettings: () => api.get('/admin/settings'),
  updateRiskTuning: (data) => api.put('/admin/settings/risk-tuning', data),
  
  // Audit Logs
  getAuditLogs: (params = {}) => api.get('/admin/audit-logs', { params }),
  
  // Feedback (FR14)
  getFeedback: (params = {}) => api.get('/admin/feedback', { params }),
  getFeedbackStats: () => api.get('/admin/feedback/stats'),
  reviewFeedback: (id, data) => api.put(`/admin/feedback/${id}/review`, data),
  
  // Listing Approval
  getAdminListings: (params = {}) => api.get('/admin/listings', { params }),
  getAdminListingStats: () => api.get('/admin/listings/stats'),
  reviewListing: (id, data) => api.put(`/admin/listings/${id}/review`, data),
  
  // Auto-Learning
  runAutoLearning: (daysBack = 30) => api.post(`/admin/learning/run?days_back=${daysBack}`),
  getLearningStats: () => api.get('/admin/learning/stats'),
  getRetrainingDataset: () => api.get('/admin/learning/retraining-dataset'),
  
  // Auto-Learning (aliases for dashboard)
  getAutoLearnInsights: () => api.get('/admin/learning/stats'),
  triggerAutoLearn: (daysBack = 30) => api.post(`/admin/learning/run?days_back=${daysBack}`),
  
  // Subscription Plans Management
  getSubscriptionPlans: () => api.get('/admin/subscription-plans'),
  getSubscriptionPlan: (id) => api.get(`/admin/subscription-plans/${id}`),
  createSubscriptionPlan: (data) => api.post('/admin/subscription-plans', data),
  updateSubscriptionPlan: (id, data) => api.patch(`/admin/subscription-plans/${id}`, data),
  deleteSubscriptionPlan: (id) => api.delete(`/admin/subscription-plans/${id}`),

  // AI Engines Dashboard
  getAIEnginesStatus: () => api.get('/admin/ai-engines/status'),
  testBert: (data) => api.post('/admin/ai-engines/test-bert', data, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  testMessageAnalysis: (data) => api.post('/admin/ai-engines/test-message-analysis', data, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  testCrossDocument: (data) => api.post('/admin/ai-engines/test-cross-document', data),
  getTrainingDataStats: () => api.get('/admin/ai-engines/training-data-stats'),

  // Preprocessing Pipeline
  getPreprocessingStatus: () => api.get('/admin/preprocessing/status'),
  analyzeDatasetPreprocessing: (id) => api.post(`/admin/preprocessing/analyze-dataset/${id}`),
  processDataset: (id) => api.post(`/admin/preprocessing/process-dataset/${id}`),

  // Trained Models (on-disk discovery + reports)
  getTrainedModels: () => api.get('/admin/trained-models'),
  getTrainedModelReport: (modelId) => api.get(`/admin/trained-models/${modelId}/report`),

  // Analytics
  getAnalyticsOverview: (days = 30) => api.get('/admin/analytics/overview', { params: { days } }),
  getModelAccuracy: () => api.get('/admin/analytics/model-accuracy'),
  getTopIndicators: (limit = 15) => api.get('/admin/analytics/top-indicators', { params: { limit } }),

  // Monitoring
  getSystemHealth: () => api.get('/admin/monitoring/system-health'),
  getAIEnginesHealth: () => api.get('/admin/monitoring/ai-engines-health'),
  getRecentErrors: (limit = 50) => api.get('/admin/monitoring/recent-errors', { params: { limit } }),
  getActivityFeed: (limit = 30) => api.get('/admin/monitoring/activity-feed', { params: { limit } }),
  getDependencyVersions: () => api.get('/admin/monitoring/dependency-versions'),
}

// Renter API
export const renterAPI = {
  analyzeListing: (data) => api.post('/renter/analyze', data),
  analyzeUrl: (data) => api.post('/renter/analyze-url', data),
  analyzeImages: (imageUrls) => api.post('/renter/analyze-images', null, { params: { image_urls: imageUrls } }),
  analyzeUploadedImages: (formData) => api.post('/renter/analyze-uploaded-images', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  getHistory: (skip = 0, limit = 50) => api.get('/renter/history', { params: { skip, limit } }),
  getAnalysisDetail: (id) => api.get(`/renter/history/${id}`),
  getExplanation: (id) => api.get(`/renter/history/${id}/explain`),
  getPlans: () => api.get('/renter/subscription/plans'),
  upgradePlan: (planName) => api.post(`/renter/subscription/upgrade?plan_name=${planName}`),
  processPayment: (data) => api.post('/renter/subscription/payment', data),
  getPaymentHistory: () => api.get('/renter/subscription/payments'),
  getCurrentSubscription: () => api.get('/renter/subscription/current'),
  getStats: () => api.get('/renter/stats'),
  submitFeedback: (data) => api.post('/renter/feedback', data),
  getFeedbackHistory: () => api.get('/renter/feedback'),
  exportReport: (data) => api.post('/renter/report/export', data),

  // Message & Conversation Analysis
  analyzeMessage: (data) => api.post('/renter/analyze-message', data, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  analyzeConversation: (data) => api.post('/renter/analyze-conversation', data),
  explainText: (data) => api.post('/renter/explain-text', data, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),

  // Listings / Property browsing
  browseListings: (params) => api.get('/listings', { params }),
  getListing: (id) => api.get(`/listings/${id}`),

  // Saved listings
  getSavedListings: () => api.get('/renter/saved-listings'),
  saveListing: (listing_id, notes = '') => api.post('/renter/saved-listings', { listing_id, notes }),
  unsaveListing: (savedId) => api.delete(`/renter/saved-listings/${savedId}`),

  // Applications
  getApplications: () => api.get('/renter/applications'),
  applyToListing: (listing_id, message = '') => api.post('/renter/applications', { listing_id, message }),
  getAppMessages: (appId) => api.get(`/applications/${appId}/messages`),
  sendAppMessage: (appId, text) => api.post(`/applications/${appId}/messages`, { text }),

  // Address verification
  verifyAddress: (data) => api.post('/renter/verify-address', data),
}

// Landlord API - Document verification and tenant screening
export const landlordAPI = {
  // Document verification
  verifyDocument: (data) => api.post('/landlord/verify-document', data),
  verifyDocumentUpload: (formData) => api.post('/landlord/verify-document-upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  
  // Tenant verification
  verifyTenant: (data) => api.post('/landlord/verify-tenant', data),
  
  // Property image verification (for landlords verifying their own listings)
  verifyPropertyImage: (data) => api.post('/landlord/verify-property-image', data),
  verifyPropertyImages: (data) => api.post('/landlord/verify-property-images', data),
  verifyListingImagesUpload: (formData) => api.post('/landlord/verify-listing-images-upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  
  // History and stats
  getVerificationHistory: (skip = 0, limit = 50) => api.get('/landlord/verification-history', { params: { skip, limit } }),
  getDashboardStats: () => api.get('/landlord/dashboard-stats'),

  // Cross-document & full application verification
  verifyCrossDocument: (data) => api.post('/landlord/verify-cross-document', data),
  verifyFullApplication: (data) => api.post('/landlord/verify-full-application', data),

  // Listings CRUD
  getMyListings: () => api.get('/landlord/listings'),
  createListing: (data) => api.post('/landlord/listings', data),
  updateListing: (id, data) => api.patch(`/landlord/listings/${id}`, data),
  deleteListing: (id) => api.delete(`/landlord/listings/${id}`),

  // Applicants
  getApplicants: (status = '') => api.get('/landlord/applicants', { params: { status_filter: status } }),
  updateApplicationStatus: (id, status) => api.patch(`/landlord/applicants/${id}`, { status }),
  getAppMessages: (appId) => api.get(`/applications/${appId}/messages`),
  sendAppMessage: (appId, text) => api.post(`/applications/${appId}/messages`, { text }),

  // Leases
  getLeases: () => api.get('/landlord/leases'),
  createLease: (data) => api.post('/landlord/leases', data),

  // Analytics
  getAnalytics: () => api.get('/landlord/analytics'),
}

// Profile API
export const profileAPI = {
  getProfile: () => api.get('/auth/profile'),
  updateProfile: (data) => api.patch('/auth/profile', data),
  changePassword: (data) => api.post('/auth/change-password', data),
}

export default api

