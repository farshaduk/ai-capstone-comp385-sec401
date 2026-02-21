import { useThemeStore } from '../../store/themeStore'
import { Sun, Moon, Monitor } from 'lucide-react'

const ThemeToggle = ({ compact = false }) => {
  const { theme, setTheme, resolvedTheme } = useThemeStore()

  if (compact) {
    return (
      <button
        onClick={() => {
          const next = resolvedTheme === 'dark' ? 'light' : 'dark'
          setTheme(next)
        }}
        className="relative p-2 rounded-xl bg-surface-100 dark:bg-surface-800 
                   hover:bg-surface-200 dark:hover:bg-surface-700
                   transition-all duration-300 group"
        aria-label="Toggle theme"
      >
        <Sun className="h-5 w-5 text-amber-500 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
        <Moon className="absolute top-2 left-2 h-5 w-5 text-primary-400 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
      </button>
    )
  }

  const options = [
    { value: 'light', icon: Sun, label: 'Light' },
    { value: 'dark', icon: Moon, label: 'Dark' },
    { value: 'system', icon: Monitor, label: 'System' },
  ]

  return (
    <div className="flex items-center gap-1 p-1 rounded-xl bg-surface-100 dark:bg-surface-800">
      {options.map(({ value, icon: Icon, label }) => (
        <button
          key={value}
          onClick={() => setTheme(value)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium
                     transition-all duration-200 ${
            theme === value
              ? 'bg-white dark:bg-surface-700 text-surface-900 dark:text-white shadow-soft-sm'
              : 'text-surface-500 dark:text-surface-400 hover:text-surface-700 dark:hover:text-surface-200'
          }`}
          aria-label={`Set ${label} theme`}
        >
          <Icon className="h-4 w-4" />
          <span className="hidden sm:inline">{label}</span>
        </button>
      ))}
    </div>
  )
}

export default ThemeToggle
