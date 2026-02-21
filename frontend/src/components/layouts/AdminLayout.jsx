import { useState } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuthStore } from '../../store/authStore'
import ThemeToggle from '../ui/ThemeToggle'
import Logo from '../ui/Logo'
import {
  LayoutDashboard, Database, Users, FileText, Cpu,
  Package, Settings, LogOut, ChevronLeft, ChevronRight,
  Bell, Menu, X, Shield, Activity,
  BarChart3, HardDrive, MessageSquare, Home
} from 'lucide-react'

const menuItems = [
  { 
    label: 'Overview', 
    items: [
      { path: '/admin', icon: LayoutDashboard, label: 'Dashboard' },
      { path: '/admin/analytics', icon: BarChart3, label: 'Analytics' },
      { path: '/admin/monitoring', icon: Activity, label: 'Monitoring' },
    ]
  },
  {
    label: 'AI & Data',
    items: [
      { path: '/admin/datasets', icon: Database, label: 'Datasets' },
      { path: '/admin/trained-models', icon: HardDrive, label: 'Trained Models' },
      { path: '/admin/ai-engines', icon: Cpu, label: 'AI Engines' },
    ]
  },
  {
    label: 'Management',
    items: [
      { path: '/admin/users', icon: Users, label: 'Users' },
      { path: '/admin/plans', icon: Package, label: 'Plans' },
      { path: '/admin/feedback-review', icon: MessageSquare, label: 'Feedback Review' },
      { path: '/admin/listing-approval', icon: Home, label: 'Listing Approval' },
    ]
  },
  {
    label: 'System',
    items: [
      { path: '/admin/audit-logs', icon: FileText, label: 'Audit Logs' },
      { path: '/admin/settings', icon: Settings, label: 'Settings' },
    ]
  },
]

const AdminLayout = ({ children, title, subtitle }) => {
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
          bg-surface-900 dark:bg-surface-950 border-r border-surface-800
          transition-all duration-300 ease-out flex flex-col
          ${collapsed ? 'w-[72px]' : 'w-[260px]'}
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Logo */}
        <div className={`flex items-center h-16 px-4 border-b border-surface-800 ${collapsed ? 'justify-center' : 'justify-between'}`}>
          {!collapsed && (
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-br from-primary-500 to-primary-700 p-1.5 rounded-lg">
                <Shield className="h-5 w-5 text-white" />
              </div>
              <div>
                <p className="font-display font-bold text-white text-sm">RentalGuard</p>
                <p className="text-[10px] text-surface-400">Admin Console</p>
              </div>
            </div>
          )}
          {collapsed && (
            <div className="bg-gradient-to-br from-primary-500 to-primary-700 p-1.5 rounded-lg">
              <Shield className="h-5 w-5 text-white" />
            </div>
          )}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="hidden lg:flex p-1.5 rounded-lg hover:bg-surface-800 text-surface-500"
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
          <button
            onClick={() => setMobileOpen(false)}
            className="lg:hidden p-1.5 rounded-lg hover:bg-surface-800 text-surface-500"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* System Status */}
        {!collapsed && (
          <div className="px-4 py-3 border-b border-surface-800">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-accent-green animate-pulse-soft" />
              <span className="text-xs font-medium text-surface-400">System Online</span>
            </div>
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
          {menuItems.map((group) => (
            <div key={group.label}>
              {!collapsed && (
                <p className="px-3 mb-2 text-[10px] font-bold text-surface-500 uppercase tracking-widest">
                  {group.label}
                </p>
              )}
              <div className="space-y-1">
                {group.items.map((item) => {
                  const Icon = item.icon
                  const isActive = location.pathname === item.path
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      onClick={() => setMobileOpen(false)}
                      title={collapsed ? item.label : undefined}
                      className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium
                        transition-all duration-200
                        ${isActive
                          ? 'bg-primary-600/20 text-primary-300 shadow-soft-xs'
                          : 'text-surface-400 hover:text-surface-100 hover:bg-surface-800'
                        }
                        ${collapsed ? 'justify-center px-2' : ''}
                      `}
                    >
                      <Icon className={`h-5 w-5 flex-shrink-0 ${isActive ? 'text-primary-400' : ''}`} />
                      {!collapsed && <span>{item.label}</span>}
                    </Link>
                  )
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* User section */}
        <div className="border-t border-surface-800 p-3">
          {!collapsed && <ThemeToggle compact />}
          <div className={`flex items-center gap-3 mt-3 ${collapsed ? 'justify-center' : ''}`}>
            <div className="w-9 h-9 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl flex items-center justify-center flex-shrink-0">
              <span className="text-sm font-bold text-white">
                {user?.full_name?.charAt(0) || 'A'}
              </span>
            </div>
            {!collapsed && (
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-white truncate">{user?.full_name}</p>
                <p className="text-xs text-surface-500">Administrator</p>
              </div>
            )}
            {!collapsed && (
              <button
                onClick={handleLogout}
                className="p-2 rounded-lg hover:bg-red-500/20 text-surface-500 hover:text-red-400 transition-colors"
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

export default AdminLayout
