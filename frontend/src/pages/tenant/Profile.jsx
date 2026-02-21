import { useState, useEffect } from 'react'
import TenantLayout from '../../components/layouts/TenantLayout'
import { useAuthStore } from '../../store/authStore'
import { profileAPI } from '../../services/api'
import { User, Mail, Phone, MapPin, Shield, Camera, Save, Key, Bell, Eye, EyeOff, CheckCircle } from 'lucide-react'
import toast from 'react-hot-toast'

const Profile = () => {
  const { user, updateUser } = useAuthStore()
  const [tab, setTab] = useState('profile')
  const [saving, setSaving] = useState(false)

  // Profile form
  const [fullName, setFullName] = useState(user?.full_name || '')
  const [phone, setPhone] = useState('')
  const [address, setAddress] = useState('')
  const [bio, setBio] = useState('')

  // Password form
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPasswords, setShowPasswords] = useState(false)

  // Notification preferences
  const [notifications, setNotifications] = useState({
    emailAlerts: true,
    riskAlerts: true,
    applicationUpdates: true,
    newsletter: false,
    marketingEmails: false,
  })

  useEffect(() => {
    const loadProfile = async () => {
      try {
        const res = await profileAPI.getProfile()
        if (res.data) {
          setFullName(res.data.full_name || '')
          setPhone(res.data.phone || '')
          setAddress(res.data.address || '')
          setBio(res.data.bio || '')
        }
      } catch { /* use defaults from auth store */ }
    }
    loadProfile()
  }, [])

  const handleSaveProfile = async () => {
    setSaving(true)
    try {
      await profileAPI.updateProfile({ full_name: fullName, phone, address, bio })
      updateUser({ full_name: fullName })
      toast.success('Profile updated successfully')
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to update profile')
    } finally {
      setSaving(false)
    }
  }

  const handleChangePassword = async () => {
    if (newPassword !== confirmPassword) {
      toast.error('Passwords do not match')
      return
    }
    if (newPassword.length < 8) {
      toast.error('Password must be at least 8 characters')
      return
    }
    setSaving(true)
    try {
      await profileAPI.changePassword({ current_password: currentPassword, new_password: newPassword })
      toast.success('Password changed successfully')
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to change password')
    } finally {
      setSaving(false)
    }
  }

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'security', label: 'Security', icon: Key },
    { id: 'notifications', label: 'Notifications', icon: Bell },
  ]

  return (
    <TenantLayout title="Profile" subtitle="Manage your account settings">
      <div className="grid lg:grid-cols-4 gap-6">
        {/* Sidebar */}
        <div className="lg:col-span-1">
          {/* Avatar */}
          <div className="card p-6 text-center mb-4">
            <div className="relative inline-block">
              <div className="w-24 h-24 bg-gradient-to-br from-tenant-500 to-tenant-600 rounded-2xl flex items-center justify-center mx-auto">
                <span className="text-3xl font-bold text-white">
                  {user?.full_name?.charAt(0) || 'T'}
                </span>
              </div>
              <button className="absolute -bottom-1 -right-1 p-1.5 bg-white dark:bg-surface-800 rounded-lg shadow border border-surface-200 dark:border-surface-700">
                <Camera className="h-3.5 w-3.5 text-surface-500" />
              </button>
            </div>
            <h3 className="font-bold text-surface-900 dark:text-white mt-4">{user?.full_name}</h3>
            <p className="text-sm text-surface-500 dark:text-surface-400">{user?.email}</p>
            <div className="flex items-center justify-center gap-1 mt-2">
              <Shield className="h-4 w-4 text-tenant-500" />
              <span className="text-xs font-semibold text-tenant-600 dark:text-tenant-400">
                {user?.subscription_plan?.toUpperCase() || 'FREE'} Plan
              </span>
            </div>
          </div>

          {/* Tab nav */}
          <nav className="space-y-1">
            {tabs.map(t => {
              const Icon = t.icon
              return (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-colors ${
                    tab === t.id
                      ? 'bg-tenant-50 text-tenant-700 dark:bg-tenant-950/50 dark:text-tenant-300'
                      : 'text-surface-600 hover:bg-surface-100 dark:text-surface-400 dark:hover:bg-surface-800'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {t.label}
                </button>
              )
            })}
          </nav>
        </div>

        {/* Content */}
        <div className="lg:col-span-3">
          {tab === 'profile' && (
            <div className="card p-6">
              <h3 className="text-lg font-bold text-surface-900 dark:text-white mb-6">Personal Information</h3>
              <div className="grid sm:grid-cols-2 gap-5">
                <div>
                  <label className="input-label">Full Name</label>
                  <div className="relative">
                    <User className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
                    <input type="text" value={fullName} onChange={e => setFullName(e.target.value)} className="input-field pl-12" placeholder="Your full name" />
                  </div>
                </div>
                <div>
                  <label className="input-label">Email</label>
                  <div className="relative">
                    <Mail className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
                    <input type="email" value={user?.email || ''} className="input-field pl-12 bg-surface-50 dark:bg-surface-800" disabled />
                  </div>
                </div>
                <div>
                  <label className="input-label">Phone Number</label>
                  <div className="relative">
                    <Phone className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
                    <input type="tel" value={phone} onChange={e => setPhone(e.target.value)} className="input-field pl-12" placeholder="+1 (416) 555-0123" />
                  </div>
                </div>
                <div>
                  <label className="input-label">Address</label>
                  <div className="relative">
                    <MapPin className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
                    <input type="text" value={address} onChange={e => setAddress(e.target.value)} className="input-field pl-12" placeholder="City, Province" />
                  </div>
                </div>
                <div className="sm:col-span-2">
                  <label className="input-label">Bio</label>
                  <textarea value={bio} onChange={e => setBio(e.target.value)} className="input-field h-24 resize-none" placeholder="Tell landlords a bit about yourself..." />
                </div>
              </div>
              <div className="flex justify-end mt-6">
                <button onClick={handleSaveProfile} disabled={saving} className="btn btn-primary btn-md">
                  <Save className="h-4 w-4" />
                  {saving ? 'Saving...' : 'Save Changes'}
                </button>
              </div>
            </div>
          )}

          {tab === 'security' && (
            <div className="card p-6">
              <h3 className="text-lg font-bold text-surface-900 dark:text-white mb-6">Change Password</h3>
              <div className="max-w-md space-y-5">
                <div>
                  <label className="input-label">Current Password</label>
                  <div className="relative">
                    <Key className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-surface-400" />
                    <input type={showPasswords ? 'text' : 'password'} value={currentPassword} onChange={e => setCurrentPassword(e.target.value)} className="input-field pl-12 pr-12" />
                    <button onClick={() => setShowPasswords(!showPasswords)} className="absolute right-4 top-1/2 -translate-y-1/2 text-surface-400">
                      {showPasswords ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>
                <div>
                  <label className="input-label">New Password</label>
                  <input type={showPasswords ? 'text' : 'password'} value={newPassword} onChange={e => setNewPassword(e.target.value)} className="input-field" />
                </div>
                <div>
                  <label className="input-label">Confirm New Password</label>
                  <input type={showPasswords ? 'text' : 'password'} value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} className="input-field" />
                </div>
                <button onClick={handleChangePassword} disabled={saving} className="btn btn-primary btn-md">
                  <Key className="h-4 w-4" />
                  {saving ? 'Updating...' : 'Update Password'}
                </button>
              </div>

              <div className="mt-8 pt-6 border-t border-surface-200 dark:border-surface-700">
                <h4 className="font-semibold text-surface-900 dark:text-white mb-4">Sessions</h4>
                <div className="card p-4 bg-surface-50 dark:bg-surface-800">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-green-100 dark:bg-green-950/30 rounded-lg">
                        <CheckCircle className="h-4 w-4 text-green-600" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-surface-900 dark:text-white">Current Session</p>
                        <p className="text-xs text-surface-500">{navigator.userAgent.includes('Windows') ? 'Windows' : navigator.userAgent.includes('Mac') ? 'macOS' : 'Linux'} • {navigator.userAgent.includes('Chrome') ? 'Chrome' : navigator.userAgent.includes('Firefox') ? 'Firefox' : 'Browser'}</p>
                      </div>
                    </div>
                    <span className="badge-success">Active</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {tab === 'notifications' && (
            <div className="card p-6">
              <h3 className="text-lg font-bold text-surface-900 dark:text-white mb-6">Notification Preferences</h3>
              <div className="space-y-4">
                {[
                  { key: 'emailAlerts', label: 'Email Alerts', desc: 'Receive important account and security alerts via email' },
                  { key: 'riskAlerts', label: 'Risk Alerts', desc: 'Get notified when a listing you saved is flagged as risky' },
                  { key: 'applicationUpdates', label: 'Application Updates', desc: 'Receive updates on your rental applications' },
                  { key: 'newsletter', label: 'Newsletter', desc: 'Monthly digest of rental market trends and tips' },
                  { key: 'marketingEmails', label: 'Marketing Emails', desc: 'Promotional offers and product updates' },
                ].map(item => (
                  <div key={item.key} className="flex items-center justify-between p-4 rounded-xl bg-surface-50 dark:bg-surface-800">
                    <div>
                      <p className="text-sm font-medium text-surface-900 dark:text-white">{item.label}</p>
                      <p className="text-xs text-surface-500 dark:text-surface-400">{item.desc}</p>
                    </div>
                    <button
                      onClick={() => setNotifications({ ...notifications, [item.key]: !notifications[item.key] })}
                      className={`relative w-11 h-6 rounded-full transition-colors ${
                        notifications[item.key] ? 'bg-tenant-500' : 'bg-surface-300 dark:bg-surface-600'
                      }`}
                    >
                      <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                        notifications[item.key] ? 'translate-x-5' : ''
                      }`} />
                    </button>
                  </div>
                ))}
              </div>
              <div className="flex justify-end mt-6">
                <button onClick={() => toast.success('Preferences saved locally')} className="btn btn-primary btn-md">
                  <Save className="h-4 w-4" />
                  Save Preferences
                </button>
                <p className="text-xs text-surface-400 mt-2">Notification preferences are stored in your browser</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </TenantLayout>
  )
}

export default Profile
