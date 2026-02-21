import { useState, useEffect } from 'react'
import { useNavigate, Link, useSearchParams } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import { authAPI } from '../services/api'
import toast from 'react-hot-toast'
import { Shield, Mail, Lock, User, Loader, ArrowLeft, Building2, UserCheck, Eye, EyeOff, Check } from 'lucide-react'
import ThemeToggle from '../components/ui/ThemeToggle'

const Register = () => {
  const [searchParams] = useSearchParams()
  const roleParam = searchParams.get('role') || 'renter'
  const { setSelectedRole } = useAuthStore()

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    full_name: '',
    role: roleParam,
  })
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [agreed, setAgreed] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    setSelectedRole(roleParam)
    setFormData(prev => ({ ...prev, role: roleParam }))
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

  const passwordStrength = (pwd) => {
    let score = 0
    if (pwd.length >= 8) score++
    if (/[A-Z]/.test(pwd)) score++
    if (/[0-9]/.test(pwd)) score++
    if (/[^A-Za-z0-9]/.test(pwd)) score++
    return score
  }

  const strength = passwordStrength(formData.password)
  const strengthLabels = ['Weak', 'Fair', 'Good', 'Strong']
  const strengthColors = ['bg-red-500', 'bg-amber-500', 'bg-blue-500', 'bg-green-500']

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!agreed) {
      toast.error('Please agree to the Terms of Service')
      return
    }
    setLoading(true)

    try {
      await authAPI.register(formData)
      toast.success('Account created! Please sign in.')
      navigate(`/login?role=${roleParam}`)
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Registration failed')
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
                Join as a {roleConfig.label}
              </h2>
              <p className="text-white/70 max-w-md">
                Create your free account and start protecting your rental journey with enterprise-grade AI.
              </p>
            </div>

            <ul className="space-y-3">
              {['Free to get started', '1,000 scans included', 'No credit card required'].map(item => (
                <li key={item} className="flex items-center gap-2 text-white/80 text-sm">
                  <Check className="h-4 w-4 text-white/60" />
                  {item}
                </li>
              ))}
            </ul>
          </div>

          <div className="text-white/30 text-sm">
            &copy; 2026 RentalGuard. All rights reserved.
          </div>
        </div>
      </div>

      {/* Right side — form */}
      <div className="flex-1 flex flex-col bg-white dark:bg-surface-950">
        <div className="flex items-center justify-between p-4 lg:p-6">
          <Link to="/get-started" className="lg:hidden inline-flex items-center gap-2 text-surface-500 hover:text-surface-700 dark:text-surface-400 dark:hover:text-surface-200 text-sm">
            <ArrowLeft className="h-4 w-4" />
            Back
          </Link>
          <div className="ml-auto">
            <ThemeToggle compact />
          </div>
        </div>

        <div className="flex-1 flex items-center justify-center p-4 lg:p-8">
          <div className="w-full max-w-md animate-fade-in-up">
            <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full ${roleConfig.accentBg} mb-6`}>
              <RoleIcon className={`h-4 w-4 ${roleConfig.accentText}`} />
              <span className={`text-sm font-semibold ${roleConfig.accentText}`}>{roleConfig.label} Registration</span>
            </div>

            <h2 className="text-display-sm font-display font-bold text-surface-900 dark:text-white mb-2">
              Create your account
            </h2>
            <p className="text-surface-500 dark:text-surface-400 mb-8">
              Already have an account?{' '}
              <Link to={`/login?role=${roleParam}`} className={`font-semibold ${roleConfig.accentText} hover:underline`}>
                Sign in
              </Link>
            </p>

            <form onSubmit={handleSubmit} className="space-y-5">
              <div>
                <label className="input-label">Full Name</label>
                <div className="relative">
                  <User className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
                  <input
                    type="text"
                    value={formData.full_name}
                    onChange={(e) => setFormData({ ...formData, full_name: e.target.value })}
                    className="input-field pl-12"
                    placeholder="John Doe"
                    required
                    autoFocus
                  />
                </div>
              </div>

              <div>
                <label className="input-label">Email Address</label>
                <div className="relative">
                  <Mail className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
                  <input
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    className="input-field pl-12"
                    placeholder="you@example.com"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="input-label">Password</label>
                <div className="relative">
                  <Lock className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={formData.password}
                    onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                    className="input-field pl-12 pr-12"
                    placeholder="Min. 8 characters"
                    required
                    minLength={6}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-surface-400 hover:text-surface-600 dark:hover:text-surface-300"
                  >
                    {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                  </button>
                </div>
                {formData.password && (
                  <div className="mt-2">
                    <div className="flex gap-1">
                      {[0, 1, 2, 3].map(i => (
                        <div key={i} className={`h-1 flex-1 rounded-full ${i < strength ? strengthColors[strength - 1] : 'bg-surface-200 dark:bg-surface-700'}`} />
                      ))}
                    </div>
                    <p className="text-xs text-surface-500 dark:text-surface-400 mt-1">
                      {strength > 0 ? strengthLabels[strength - 1] : 'Too weak'}
                    </p>
                  </div>
                )}
              </div>

              <label className="flex items-start gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={agreed}
                  onChange={(e) => setAgreed(e.target.checked)}
                  className="w-4 h-4 mt-0.5 rounded border-surface-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="text-sm text-surface-600 dark:text-surface-400">
                  I agree to the{' '}
                  <Link to="/terms" className="text-primary-600 dark:text-primary-400 hover:underline">Terms of Service</Link>
                  {' '}and{' '}
                  <Link to="/privacy" className="text-primary-600 dark:text-primary-400 hover:underline">Privacy Policy</Link>
                </span>
              </label>

              <button
                type="submit"
                disabled={loading || !agreed}
                className="w-full btn btn-lg btn-primary"
              >
                {loading ? (
                  <>
                    <Loader className="h-5 w-5 animate-spin" />
                    <span>Creating account...</span>
                  </>
                ) : (
                  <span>Create Account</span>
                )}
              </button>
            </form>

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

export default Register

