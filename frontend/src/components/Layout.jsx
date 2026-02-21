import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import { 
  LayoutDashboard, Database, Brain, Users, FileText, 
  LogOut, Shield, Scan, History, CreditCard, Package,
  Cpu, FileCheck, Image, MessageSquare
} from 'lucide-react'

const Layout = ({ children, title }) => {
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()
  const location = useLocation()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  const isAdmin = user?.role === 'admin'

  const adminMenuItems = [
    { path: '/admin', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/admin/datasets', icon: Database, label: 'Datasets' },
    { path: '/admin/models', icon: Brain, label: 'Models' },
    { path: '/admin/ai-engines', icon: Cpu, label: 'AI Engines' },
    { path: '/admin/users', icon: Users, label: 'Users' },
    { path: '/admin/plans', icon: Package, label: 'Plans' },
    { path: '/admin/audit-logs', icon: FileText, label: 'Audit Logs' },
  ]

  const renterMenuItems = [
    { path: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/analyze', icon: Scan, label: 'Analyze Listing' },
    { path: '/history', icon: History, label: 'History' },
    { path: '/subscription', icon: CreditCard, label: 'Subscription' },
  ]

  const landlordMenuItems = [
    { path: '/landlord', icon: Shield, label: 'Landlord Hub' },
    { path: '/landlord/documents', icon: FileCheck, label: 'Verify Documents' },
    { path: '/landlord/tenants', icon: Users, label: 'Verify Tenant' },
    { path: '/landlord/property-images', icon: Image, label: 'Verify Images' },
    { path: '/landlord/history', icon: History, label: 'Verification History' },
  ]

  const menuItems = isAdmin ? adminMenuItems : renterMenuItems

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <aside className="w-64 bg-gradient-to-b from-primary-800 to-primary-900 text-white fixed h-full">
        <div className="p-6">
          <div className="flex items-center space-x-2 mb-8">
            <Shield className="h-8 w-8" />
            <div>
              <h1 className="font-bold text-xl">Rental Guard</h1>
              <p className="text-xs text-primary-200">AI-Powered Protection</p>
            </div>
          </div>

          <nav className="space-y-2">
            {menuItems.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.path
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition ${
                    isActive
                      ? 'bg-white text-primary-900 shadow-lg'
                      : 'text-primary-100 hover:bg-primary-700'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <span className="font-medium">{item.label}</span>
                </Link>
              )
            })}
          </nav>

          {/* Landlord Tools - available to all authenticated users */}
          <div className="mt-6 pt-4 border-t border-primary-700">
            <p className="px-4 text-xs font-semibold text-primary-300 uppercase tracking-wider mb-2">
              Landlord Tools
            </p>
            <nav className="space-y-1">
              {landlordMenuItems.map((item) => {
                const Icon = item.icon
                const isActive = location.pathname === item.path
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center space-x-3 px-4 py-2.5 rounded-lg transition text-sm ${
                      isActive
                        ? 'bg-white text-primary-900 shadow-lg'
                        : 'text-primary-100 hover:bg-primary-700'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="font-medium">{item.label}</span>
                  </Link>
                )
              })}
            </nav>
          </div>
        </div>

        <div className="absolute bottom-0 w-full p-6 border-t border-primary-700">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 bg-primary-600 rounded-full flex items-center justify-center">
              <span className="font-bold text-lg">
                {user?.full_name?.charAt(0) || 'U'}
              </span>
            </div>
            <div className="flex-1">
              <p className="font-medium text-sm truncate">{user?.full_name}</p>
              <p className="text-xs text-primary-200 capitalize">{user?.role}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-primary-700 hover:bg-primary-600 rounded-lg transition"
          >
            <LogOut className="h-4 w-4" />
            <span>Logout</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 ml-64">
        <header className="bg-white shadow-sm border-b border-gray-200 px-8 py-6">
          <h2 className="text-3xl font-bold text-gray-800">{title}</h2>
        </header>
        <div className="p-8">
          {children}
        </div>
      </main>
    </div>
  )
}

export default Layout

