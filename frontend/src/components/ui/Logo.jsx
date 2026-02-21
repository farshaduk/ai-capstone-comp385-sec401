import { Shield } from 'lucide-react'
import { Link } from 'react-router-dom'

const Logo = ({ size = 'md', showText = true, linkTo = '/' }) => {
  const sizes = {
    sm: { icon: 'h-6 w-6', text: 'text-lg', tagline: 'text-[10px]' },
    md: { icon: 'h-8 w-8', text: 'text-xl', tagline: 'text-xs' },
    lg: { icon: 'h-10 w-10', text: 'text-2xl', tagline: 'text-sm' },
    xl: { icon: 'h-12 w-12', text: 'text-3xl', tagline: 'text-sm' },
  }

  const s = sizes[size] || sizes.md

  const content = (
    <div className="flex items-center gap-3">
      <div className="relative">
        <div className="bg-gradient-to-br from-primary-500 to-primary-700 p-2 rounded-xl shadow-glow-primary">
          <Shield className={`${s.icon} text-white`} />
        </div>
        <div className="absolute -top-1 -right-1 w-3 h-3 bg-accent-green rounded-full border-2 border-white dark:border-surface-900 animate-pulse-soft" />
      </div>
      {showText && (
        <div>
          <h1 className={`font-display font-bold ${s.text} text-surface-900 dark:text-white`}>
            Rental<span className="text-primary-600 dark:text-primary-400">Guard</span>
          </h1>
          <p className={`${s.tagline} text-surface-500 dark:text-surface-400 font-medium -mt-0.5`}>
            AI-Powered Protection
          </p>
        </div>
      )}
    </div>
  )

  if (linkTo) {
    return <Link to={linkTo} className="inline-flex hover:opacity-90 transition-opacity">{content}</Link>
  }
  return content
}

export default Logo
