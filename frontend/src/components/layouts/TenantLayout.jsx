import { useState, useEffect } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuthStore } from '../../store/authStore'
import { renterAPI } from '../../services/api'
import ThemeToggle from '../ui/ThemeToggle'
import Logo from '../ui/Logo'
import {
  LayoutDashboard, Search, History, CreditCard, Heart,
  MessageSquare, User, LogOut, ChevronLeft, ChevronRight,
  Bell, Settings, HelpCircle, Menu, X, Shield, Camera, MapPin
} from 'lucide-react'

const menuItems = [
  { path: '/tenant', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/tenant/listings', icon: Search, label: 'Browse Listings' },
  { path: '/tenant/analyze', icon: Shield, label: 'Analyze Listing' },
  { path: '/tenant/verify-images', icon: Camera, label: 'Verify Images' },
  { path: '/tenant/verify-address', icon: MapPin, label: 'Verify Address' },
  { path: '/tenant/saved', icon: Heart, label: 'Saved Listings' },
  { path: '/tenant/applications', icon: MessageSquare, label: 'Applications' },
  { path: '/tenant/payments', icon: CreditCard, label: 'Payments' },
  { path: '/tenant/history', icon: History, label: 'Analysis History' },
  { path: '/tenant/subscription', icon: Settings, label: 'Subscription' },
  { path: '/tenant/profile', icon: User, label: 'Profile' },
]

const TenantLayout = ({ children, title, subtitle }) => {
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [trustScore, setTrustScore] = useState(0)

  useEffect(() => {
    renterAPI.getStats().then(res => {
      const s = res.data
      if (!s) return
      const total = s.total_analyses || 0
      const safe = s.safe_count || 0
      const fraud = s.fraud_count || 0
      // Trust score rewards engagement: scanning = due diligence.
      // Finding fraud is GOOD (tenant protected themselves), not a penalty.
      const engagementBonus = Math.min(30, total * 3)
      const verifiedRatio = total > 0 ? (safe + fraud) / total : 0
      const diligenceBonus = verifiedRatio * 30
      setTrustScore(total === 0 ? 0 : Math.min(100, Math.max(0, Math.round(40 + engagementBonus + diligenceBonus))))
    }).catch(() => {})
  }, [location.pathname])

  const handleLogout = () => {
    logout()
    navigate('/')
  }

  return (
    <div className="min-h-screen bg-surface-50 dark:bg-surface-950 flex">
      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed top-0 left-0 h-full z-50
          bg-white dark:bg-surface-900 border-r border-surface-200 dark:border-surface-800
          transition-all duration-300 ease-out flex flex-col
          ${collapsed ? 'w-[72px]' : 'w-[260px]'}
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Logo */}
        <div className={`flex items-center h-16 px-4 border-b border-surface-200 dark:border-surface-800 ${collapsed ? 'justify-center' : 'justify-between'}`}>
          {!collapsed && <Logo size="sm" linkTo="/tenant" />}
          {collapsed && (
            <div className="bg-gradient-to-br from-tenant-500 to-tenant-700 p-1.5 rounded-lg">
              <Shield className="h-5 w-5 text-white" />
            </div>
          )}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="hidden lg:flex p-1.5 rounded-lg hover:bg-surface-100 dark:hover:bg-surface-800 text-surface-400"
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
          <button
            onClick={() => setMobileOpen(false)}
            className="lg:hidden p-1.5 rounded-lg hover:bg-surface-100 dark:hover:bg-surface-800 text-surface-400"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Trust Score */}
        {!collapsed && (
          <div className="px-4 py-3 border-b border-surface-200 dark:border-surface-800">
            <div className="bg-gradient-to-r from-tenant-50 to-emerald-50 dark:from-tenant-950/30 dark:to-emerald-950/30 rounded-xl p-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold text-tenant-700 dark:text-tenant-300">Trust Score</span>
                <span className="text-lg font-bold text-tenant-600 dark:text-tenant-400">{trustScore}</span>
              </div>
              <div className="w-full bg-tenant-200 dark:bg-tenant-900 rounded-full h-1.5">
                <div className="bg-tenant-500 h-1.5 rounded-full transition-all" style={{ width: `${trustScore}%` }} />
              </div>
            </div>
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
          {menuItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path
            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => setMobileOpen(false)}
                title={collapsed ? item.label : undefined}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium
                  transition-all duration-200 group
                  ${isActive
                    ? 'bg-tenant-50 text-tenant-700 dark:bg-tenant-950/50 dark:text-tenant-300 shadow-soft-xs'
                    : 'text-surface-600 hover:text-surface-900 hover:bg-surface-100 dark:text-surface-400 dark:hover:text-surface-100 dark:hover:bg-surface-800'
                  }
                  ${collapsed ? 'justify-center px-2' : ''}
                `}
              >
                <Icon className={`h-5 w-5 flex-shrink-0 ${isActive ? 'text-tenant-600 dark:text-tenant-400' : ''}`} />
                {!collapsed && <span>{item.label}</span>}
              </Link>
            )
          })}
        </nav>

        {/* User section */}
        <div className="border-t border-surface-200 dark:border-surface-800 p-3">
          {!collapsed && <ThemeToggle compact />}
          <div className={`flex items-center gap-3 mt-3 ${collapsed ? 'justify-center' : ''}`}>
            <div className="w-9 h-9 bg-gradient-to-br from-tenant-500 to-tenant-600 rounded-xl flex items-center justify-center flex-shrink-0">
              <span className="text-sm font-bold text-white">
                {user?.full_name?.charAt(0) || 'T'}
              </span>
            </div>
            {!collapsed && (
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-surface-900 dark:text-white truncate">
                  {user?.full_name}
                </p>
                <p className="text-xs text-surface-500 dark:text-surface-400">Tenant</p>
              </div>
            )}
            {!collapsed && (
              <button
                onClick={handleLogout}
                className="p-2 rounded-lg hover:bg-red-50 dark:hover:bg-red-950/30 text-surface-400 hover:text-red-500 transition-colors"
                title="Logout"
              >
                <LogOut className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div className={`flex-1 transition-all duration-300 ${collapsed ? 'lg:ml-[72px]' : 'lg:ml-[260px]'}`}>
        {/* Top bar */}
        <header className="sticky top-0 z-30 bg-white/80 dark:bg-surface-900/80 backdrop-blur-xl border-b border-surface-200 dark:border-surface-800">
          <div className="flex items-center justify-between h-16 px-4 lg:px-8">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setMobileOpen(true)}
                className="lg:hidden p-2 rounded-xl hover:bg-surface-100 dark:hover:bg-surface-800 text-surface-500"
              >
                <Menu className="h-5 w-5" />
              </button>
              <div>
                {title && <h2 className="text-lg font-bold text-surface-900 dark:text-white">{title}</h2>}
                {subtitle && <p className="text-sm text-surface-500 dark:text-surface-400">{subtitle}</p>}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button className="p-2 rounded-xl hover:bg-surface-100 dark:hover:bg-surface-800 text-surface-400 relative">
                <Bell className="h-5 w-5" />
                <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-accent-red rounded-full" />
              </button>
              <button className="p-2 rounded-xl hover:bg-surface-100 dark:hover:bg-surface-800 text-surface-400">
                <HelpCircle className="h-5 w-5" />
              </button>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="p-4 lg:p-8 page-enter">
          {children}
        </main>
      </div>
    </div>
  )
}

export default TenantLayout
