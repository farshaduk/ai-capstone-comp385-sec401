import { useState, useEffect } from 'react'
import { useNavigate, Link, useSearchParams } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import { useThemeStore } from '../store/themeStore'
import { authAPI } from '../services/api'
import toast from 'react-hot-toast'
import { Shield, Mail, Lock, Loader, ArrowLeft, Building2, UserCheck, Eye, EyeOff } from 'lucide-react'
import ThemeToggle from '../components/ui/ThemeToggle'

const Login = () => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const { login, setSelectedRole } = useAuthStore()

  const roleParam = searchParams.get('role')

  useEffect(() => {
    if (roleParam) {
      setSelectedRole(roleParam)
    }
  }, [roleParam, setSelectedRole])

  const getRoleConfig = () => {
    switch (roleParam) {
      case 'landlord':
        return {
          icon: Building2,
          label: 'Landlord',
          gradient: 'from-landlord-600 via-landlord-700 to-blue-800',
          accentBg: 'bg-landlord-100 dark:bg-landlord-950/50',
          accentText: 'text-landlord-600 dark:text-landlord-400',
        }
      case 'admin':
        return {
          icon: Shield,
          label: 'Administrator',
          gradient: 'from-surface-800 via-surface-900 to-surface-950',
          accentBg: 'bg-primary-100 dark:bg-primary-950/50',
          accentText: 'text-primary-600 dark:text-primary-400',
        }
      default:
        return {
          icon: UserCheck,
          label: 'Tenant',
          gradient: 'from-tenant-600 via-tenant-700 to-emerald-800',
          accentBg: 'bg-tenant-100 dark:bg-tenant-950/50',
          accentText: 'text-tenant-600 dark:text-tenant-400',
        }
    }
  }

  const roleConfig = getRoleConfig()
  const RoleIcon = roleConfig.icon

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)

    try {
      const response = await authAPI.login(email, password)
      const { access_token } = response.data

      login(access_token, null)

      const userResponse = await authAPI.getMe()
      const user = userResponse.data

      login(access_token, user)
      toast.success('Welcome back!')

      // Strict role-based redirect
      if (user.role === 'admin') {
        navigate('/admin')
      } else if (user.role === 'landlord') {
        navigate('/landlord')
      } else {
        navigate('/tenant')
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex">
      {/* Left side — branding */}
      <div className={`hidden lg:flex lg:w-1/2 bg-gradient-to-br ${roleConfig.gradient} relative overflow-hidden`}>
        <div className="absolute inset-0 bg-black/10" />
        <div className="absolute top-0 right-0 w-96 h-96 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-white/5 rounded-full translate-y-1/2 -translate-x-1/2" />
        
        <div className="relative z-10 flex flex-col justify-between p-12 w-full">
          <div>
            <Link to="/" className="inline-flex items-center gap-2 text-white/80 hover:text-white transition-colors text-sm">
              <ArrowLeft className="h-4 w-4" />
              Back to home
            </Link>
          </div>

          <div className="space-y-8">
            <div className="flex items-center gap-4">
              <div className="bg-white/20 backdrop-blur p-3 rounded-2xl">
                <Shield className="h-10 w-10 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-display font-bold text-white">RentalGuard</h1>
                <p className="text-white/60 text-sm">AI-Powered Protection</p>
              </div>
            </div>
            
            <div>
              <h2 className="text-2xl font-display font-bold text-white mb-3">
                Welcome Back, {roleConfig.label}
              </h2>
              <p className="text-white/70 max-w-md">
                Sign in to access your personalized dashboard with AI-powered fraud detection and trust scoring tools.
              </p>
            </div>

            <div className="flex items-center gap-3 text-white/50 text-sm">
              <Shield className="h-4 w-4" />
              <span>256-bit SSL encrypted connection</span>
            </div>
          </div>

          <div className="text-white/30 text-sm">
            &copy; 2026 RentalGuard. All rights reserved.
          </div>
        </div>
      </div>

      {/* Right side — form */}
      <div className="flex-1 flex flex-col bg-white dark:bg-surface-950">
        {/* Top bar */}
        <div className="flex items-center justify-between p-4 lg:p-6">
          <Link to="/" className="lg:hidden inline-flex items-center gap-2 text-surface-500 hover:text-surface-700 dark:text-surface-400 dark:hover:text-surface-200 text-sm">
            <ArrowLeft className="h-4 w-4" />
            Home
          </Link>
          <div className="ml-auto">
            <ThemeToggle compact />
          </div>
        </div>

        {/* Form */}
        <div className="flex-1 flex items-center justify-center p-4 lg:p-8">
          <div className="w-full max-w-md animate-fade-in-up">
            {/* Role indicator */}
            <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full ${roleConfig.accentBg} mb-6`}>
              <RoleIcon className={`h-4 w-4 ${roleConfig.accentText}`} />
              <span className={`text-sm font-semibold ${roleConfig.accentText}`}>{roleConfig.label} Login</span>
            </div>

            <h2 className="text-display-sm font-display font-bold text-surface-900 dark:text-white mb-2">
              Sign in to your account
            </h2>
            <p className="text-surface-500 dark:text-surface-400 mb-8">
              Don't have an account?{' '}
              <Link to={`/register${roleParam ? `?role=${roleParam}` : ''}`} className={`font-semibold ${roleConfig.accentText} hover:underline`}>
                Create one
              </Link>
            </p>

            <form onSubmit={handleSubmit} className="space-y-5">
              <div>
                <label className="input-label">Email Address</label>
                <div className="relative">
                  <Mail className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="input-field pl-12"
                    placeholder="you@example.com"
                    required
                    autoFocus
                  />
                </div>
              </div>

              <div>
                <label className="input-label">Password</label>
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="input-field pl-12 pr-12"
                    placeholder="Enter your password"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-surface-400 hover:text-surface-600 dark:hover:text-surface-300"
                  >
                    {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" className="w-4 h-4 rounded border-surface-300 text-primary-600 focus:ring-primary-500" />
                  <span className="text-sm text-surface-600 dark:text-surface-400">Remember me</span>
                </label>
                <a href="#" className="text-sm font-medium text-primary-600 dark:text-primary-400 hover:underline">
                  Forgot password?
                </a>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full btn btn-lg btn-primary"
              >
                {loading ? (
                  <>
                    <Loader className="h-5 w-5 animate-spin" />
                    <span>Signing in...</span>
                  </>
                ) : (
                  <span>Sign In</span>
                )}
              </button>
            </form>

            {/* Demo credentials */}
            <div className="mt-8 p-4 rounded-xl bg-surface-50 dark:bg-surface-800/50 border border-surface-200 dark:border-surface-700">
              <p className="text-xs font-semibold text-surface-500 dark:text-surface-400 mb-2">Demo Credentials</p>
              <div className="text-xs text-surface-500 dark:text-surface-400 space-y-1">
                <p><span className="font-medium text-surface-700 dark:text-surface-300">Admin:</span> admin@rentalfraud.com / admin123</p>
                <p><span className="font-medium text-surface-700 dark:text-surface-300">Landlord:</span> landlord@example.com / landlord123</p>
                <p><span className="font-medium text-surface-700 dark:text-surface-300">Tenant:</span> renter1@example.com / renter123</p>
              </div>
            </div>

            {/* Switch role */}
            <div className="mt-6 text-center">
              <p className="text-xs text-surface-400 dark:text-surface-500">
                Wrong role?{' '}
                <Link to="/get-started" className="font-medium text-primary-600 dark:text-primary-400 hover:underline">
                  Select a different role
                </Link>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Login

