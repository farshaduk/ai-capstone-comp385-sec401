import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useAuthStore = create(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      isAuthenticated: false,
      selectedRole: null, // 'tenant' | 'landlord' | 'admin' — set BEFORE login

      setSelectedRole: (role) => set({ selectedRole: role }),

      login: (token, user) => set({
        token,
        user,
        isAuthenticated: true,
        selectedRole: user?.role || get().selectedRole,
      }),

      logout: () => set({
        token: null,
        user: null,
        isAuthenticated: false,
        selectedRole: null,
      }),

      updateUser: (user) => set({ user }),

      // Role check helpers
      isAdmin: () => get().user?.role === 'admin',
      isLandlord: () => get().user?.role === 'landlord',
      isTenant: () => get().user?.role === 'renter' || get().user?.role === 'tenant',
      getRole: () => get().user?.role,
      getDashboardPath: () => {
        const role = get().user?.role
        if (role === 'admin') return '/admin'
        if (role === 'landlord') return '/landlord'
        return '/tenant'
      },
    }),
    {
      name: 'auth-storage',
    }
  )
)

