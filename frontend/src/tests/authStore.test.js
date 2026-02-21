/**
 * Tests for the Zustand auth store.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock localStorage
const localStorageMock = (() => {
  let store = {}
  return {
    getItem: (key) => store[key] || null,
    setItem: (key, value) => { store[key] = String(value) },
    removeItem: (key) => { delete store[key] },
    clear: () => { store = {} },
  }
})()
Object.defineProperty(window, 'localStorage', { value: localStorageMock })

import { useAuthStore } from '../store/authStore'

describe('Auth Store', () => {
  beforeEach(() => {
    localStorageMock.clear()
    // Reset the store state
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
    })
  })

  it('starts with no authentication', () => {
    const state = useAuthStore.getState()
    expect(state.isAuthenticated).toBe(false)
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
  })

  it('has a login function', () => {
    expect(typeof useAuthStore.getState().login).toBe('function')
  })

  it('has a logout function', () => {
    expect(typeof useAuthStore.getState().logout).toBe('function')
  })

  it('logout clears state', () => {
    const { logout } = useAuthStore.getState()
    // Set some state first
    useAuthStore.setState({ user: { id: 1 }, token: 'abc', isAuthenticated: true })
    logout()
    const state = useAuthStore.getState()
    expect(state.isAuthenticated).toBe(false)
    expect(state.token).toBeNull()
  })
})
