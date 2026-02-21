import { Loader } from 'lucide-react'

const LoadingSpinner = ({ size = 'md', text = 'Loading...', fullPage = false }) => {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-8 w-8',
    lg: 'h-12 w-12',
  }

  const spinner = (
    <div className="flex flex-col items-center justify-center gap-3">
      <div className="relative">
        <div className={`${sizeClasses[size]} border-4 border-surface-200 dark:border-surface-700 rounded-full`} />
        <div className={`absolute inset-0 ${sizeClasses[size]} border-4 border-transparent border-t-primary-500 rounded-full animate-spin`} />
      </div>
      {text && <p className="text-sm font-medium text-surface-500 dark:text-surface-400">{text}</p>}
    </div>
  )

  if (fullPage) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-surface-50 dark:bg-surface-950">
        {spinner}
      </div>
    )
  }

  return spinner
}

export default LoadingSpinner
