import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useThemeStore = create(
  persist(
    (set, get) => ({
      theme: 'system', // 'light' | 'dark' | 'system'
      resolvedTheme: 'light',

      setTheme: (theme) => {
        set({ theme })
        get().applyTheme(theme)
      },

      toggleTheme: () => {
        const current = get().resolvedTheme
        const next = current === 'dark' ? 'light' : 'dark'
        set({ theme: next })
        get().applyTheme(next)
      },

      applyTheme: (theme) => {
        let resolved = theme
        if (theme === 'system') {
          resolved = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
        }
        set({ resolvedTheme: resolved })
        
        if (resolved === 'dark') {
          document.documentElement.classList.add('dark')
        } else {
          document.documentElement.classList.remove('dark')
        }
      },

      initTheme: () => {
        const { theme, applyTheme } = get()
        applyTheme(theme)

        // Listen for system theme changes
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
        const handler = () => {
          if (get().theme === 'system') {
            applyTheme('system')
          }
        }
        mediaQuery.addEventListener('change', handler)
        return () => mediaQuery.removeEventListener('change', handler)
      },
    }),
    {
      name: 'theme-storage',
      partialize: (state) => ({ theme: state.theme }),
    }
  )
)
