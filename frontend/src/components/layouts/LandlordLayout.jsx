import { useState } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuthStore } from '../../store/authStore'
import ThemeToggle from '../ui/ThemeToggle'
import Logo from '../ui/Logo'
import {
  LayoutDashboard, Plus, Users, FileText, CreditCard,
  BarChart3, Settings, LogOut, ChevronLeft,
  ChevronRight, Bell, HelpCircle, Menu, X, Shield,
  Home, AlertTriangle, FileCheck, Image
} from 'lucide-react'

const menuItems = [
  { path: '/landlord', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/landlord/listings', icon: Home, label: 'My Listings' },
  { path: '/landlord/listings/new', icon: Plus, label: 'Create Listing' },
  { path: '/landlord/applicants', icon: Users, label: 'Applicants' },
  { path: '/landlord/risk-analysis', icon: AlertTriangle, label: 'Risk Analysis' },
  { path: '/landlord/documents', icon: FileCheck, label: 'Verify Documents' },
  { path: '/landlord/tenants', icon: Users, label: 'Verify Tenants' },
  { path: '/landlord/property-images', icon: Image, label: 'Verify Images' },
  { path: '/landlord/leases', icon: FileText, label: 'Leases' },
  { path: '/landlord/payments', icon: CreditCard, label: 'Payments' },
  { path: '/landlord/analytics', icon: BarChart3, label: 'Analytics' },
  { path: '/landlord/history', icon: FileText, label: 'History' },
  { path: '/landlord/settings', icon: Settings, label: 'Settings' },
]

const LandlordLayout = ({ children, title, subtitle }) => {
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

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
          {!collapsed && <Logo size="sm" linkTo="/landlord" />}
          {collapsed && (
            <div className="bg-gradient-to-br from-landlord-500 to-landlord-700 p-1.5 rounded-lg">
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

        {/* Quick Stats */}
        {!collapsed && (
          <div className="px-4 py-3 border-b border-surface-200 dark:border-surface-800">
            <div className="bg-gradient-to-r from-landlord-50 to-blue-50 dark:from-landlord-950/30 dark:to-blue-950/30 rounded-xl p-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-surface-500 dark:text-surface-400">Listings</p>
                  <p className="text-lg font-bold text-landlord-600 dark:text-landlord-400">12</p>
                </div>
                <div>
                  <p className="text-xs text-surface-500 dark:text-surface-400">Applicants</p>
                  <p className="text-lg font-bold text-landlord-600 dark:text-landlord-400">28</p>
                </div>
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
                    ? 'bg-landlord-50 text-landlord-700 dark:bg-landlord-950/50 dark:text-landlord-200 shadow-soft-xs'
                    : 'text-surface-600 hover:text-surface-900 hover:bg-surface-100 dark:text-surface-400 dark:hover:text-surface-100 dark:hover:bg-surface-800'
                  }
                  ${collapsed ? 'justify-center px-2' : ''}
                `}
              >
                <Icon className={`h-5 w-5 flex-shrink-0 ${isActive ? 'text-landlord-600 dark:text-landlord-400' : ''}`} />
                {!collapsed && <span>{item.label}</span>}
              </Link>
            )
          })}
        </nav>

        {/* User section */}
        <div className="border-t border-surface-200 dark:border-surface-800 p-3">
          {!collapsed && <ThemeToggle compact />}
          <div className={`flex items-center gap-3 mt-3 ${collapsed ? 'justify-center' : ''}`}>
            <div className="w-9 h-9 bg-gradient-to-br from-landlord-500 to-landlord-600 rounded-xl flex items-center justify-center flex-shrink-0">
              <span className="text-sm font-bold text-white">
                {user?.full_name?.charAt(0) || 'L'}
              </span>
            </div>
            {!collapsed && (
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-surface-900 dark:text-white truncate">
                  {user?.full_name}
                </p>
                <p className="text-xs text-surface-500 dark:text-surface-400">Landlord</p>
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

export default LandlordLayout
