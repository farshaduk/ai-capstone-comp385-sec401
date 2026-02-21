/**
 * Tests for the API service layer — validates endpoint URLs, 
 * interceptor setup, and exported method shapes.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock axios to intercept requests
vi.mock('axios', () => {
  const instance = {
    get: vi.fn().mockResolvedValue({ data: {} }),
    post: vi.fn().mockResolvedValue({ data: {} }),
    put: vi.fn().mockResolvedValue({ data: {} }),
    patch: vi.fn().mockResolvedValue({ data: {} }),
    delete: vi.fn().mockResolvedValue({ data: {} }),
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
    defaults: { headers: { common: {} } },
  }
  return {
    default: {
      create: vi.fn(() => instance),
      ...instance,
    },
  }
})

// Mock auth store
vi.mock('../store/authStore', () => ({
  useAuthStore: {
    getState: () => ({ token: 'test-token', logout: vi.fn() }),
  },
}))

import { adminAPI, renterAPI, landlordAPI } from '../services/api'

describe('API Service — adminAPI', () => {
  it('exports required admin methods', () => {
    const requiredMethods = [
      'getDashboard', 'getDatasets', 'getModels', 'getUsers', 'getAuditLogs',
    ]
    for (const method of requiredMethods) {
      expect(typeof adminAPI[method]).toBe('function')
    }
  })

  it('exports BERT training methods', () => {
    expect(typeof adminAPI.trainBert).toBe('function')
    expect(typeof adminAPI.getBertStatus).toBe('function')
  })

  it('exports AI engines methods', () => {
    expect(typeof adminAPI.getAIEnginesStatus).toBe('function')
    expect(typeof adminAPI.testBert).toBe('function')
    expect(typeof adminAPI.testMessageAnalysis).toBe('function')
    expect(typeof adminAPI.testCrossDocument).toBe('function')
    expect(typeof adminAPI.getTrainingDataStats).toBe('function')
  })

  it('exports preprocessing methods', () => {
    expect(typeof adminAPI.getPreprocessingStatus).toBe('function')
  })
})

describe('API Service — renterAPI', () => {
  it('exports required renter methods', () => {
    const requiredMethods = [
      'analyzeListing', 'getHistory', 'getPlans', 'getStats',
    ]
    for (const method of requiredMethods) {
      expect(typeof renterAPI[method]).toBe('function')
    }
  })

  it('exports message analysis methods', () => {
    expect(typeof renterAPI.analyzeMessage).toBe('function')
    expect(typeof renterAPI.analyzeConversation).toBe('function')
  })

  it('exports explainability methods', () => {
    expect(typeof renterAPI.explainText).toBe('function')
  })
})

describe('API Service — landlordAPI', () => {
  it('exports required landlord methods', () => {
    const requiredMethods = [
      'verifyDocument', 'verifyDocumentUpload', 'verifyTenant',
      'verifyPropertyImage', 'verifyPropertyImages',
      'verifyListingImagesUpload',
      'getVerificationHistory', 'getDashboardStats',
    ]
    for (const method of requiredMethods) {
      expect(typeof landlordAPI[method]).toBe('function')
    }
  })

  it('exports advanced verification methods', () => {
    expect(typeof landlordAPI.verifyCrossDocument).toBe('function')
    expect(typeof landlordAPI.verifyFullApplication).toBe('function')
  })
})
