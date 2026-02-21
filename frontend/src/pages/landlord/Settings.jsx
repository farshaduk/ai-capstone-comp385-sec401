import { useState, useEffect } from 'react'
import LandlordLayout from '../../components/layouts/LandlordLayout'
import { profileAPI } from '../../services/api'
import { useAuthStore } from '../../store/authStore'
import { User, Mail, Shield, Save, Key, Eye, EyeOff, Loader, CheckCircle } from 'lucide-react'
import toast from 'react-hot-toast'

const Settings = () => {
  const { user } = useAuthStore()
  const [tab, setTab] = useState('profile')
  const [saving, setSaving] = useState(false)
  const [profile, setProfile] = useState(null)
  const [loading, setLoading] = useState(true)

  // Password
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPasswords, setShowPasswords] = useState(false)

  useEffect(() => {
    const load = async () => {
      try {
        const { data } = await profileAPI.getProfile()
        setProfile(data)
      } catch (err) {
        console.error(err)
        toast.error('Failed to load profile')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const handleProfileSave = async () => {
    setSaving(true)
    try {
      await profileAPI.updateProfile({ full_name: profile.full_name })
      toast.success('Profile updated')
    } catch (err) {
      toast.error('Failed to update profile')
    } finally {
      setSaving(false)
    }
  }

  const handlePasswordChange = async () => {
    if (newPassword !== confirmPassword) return toast.error('Passwords do not match')
    if (newPassword.length < 6) return toast.error('Password must be at least 6 characters')
    setSaving(true)
    try {
      await profileAPI.changePassword({
        current_password: currentPassword,
        new_password: newPassword,
      })
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

  if (loading) return (
    <LandlordLayout title="Settings" subtitle="Manage your account settings">
      <div className="card p-16 text-center"><Loader className="h-8 w-8 animate-spin text-landlord-500 mx-auto mb-3" /><p className="text-surface-500">Loading...</p></div>
    </LandlordLayout>
  )

  return (
    <LandlordLayout title="Settings" subtitle="Manage your account settings">
      {/* Tabs */}
      <div className="flex items-center gap-2 mb-6">
        {['profile', 'security'].map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${tab === t
              ? 'bg-landlord-500 text-white'
              : 'bg-surface-100 dark:bg-surface-800 text-surface-600 dark:text-surface-400 hover:bg-surface-200 dark:hover:bg-surface-700'}`}>
            {t === 'profile' ? 'Profile' : 'Security'}
          </button>
        ))}
      </div>

      {tab === 'profile' && profile && (
        <div className="card p-6 max-w-xl">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 rounded-full bg-landlord-100 dark:bg-landlord-900/30 flex items-center justify-center">
              <User className="h-8 w-8 text-landlord-600 dark:text-landlord-400" />
            </div>
            <div>
              <h3 className="font-semibold text-surface-900 dark:text-white">{profile.full_name}</h3>
              <p className="text-sm text-surface-500">{profile.email}</p>
              <div className="flex items-center gap-2 mt-1">
                <span className="badge-success flex items-center gap-1"><Shield className="h-3 w-3" />{profile.role}</span>
                <span className="text-xs text-surface-400">{profile.subscription_plan} plan</span>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="input-label">Full Name</label>
              <input type="text" value={profile.full_name || ''} onChange={e => setProfile({ ...profile, full_name: e.target.value })} className="input-field" />
            </div>
            <div>
              <label className="input-label">Email Address</label>
              <input type="email" value={profile.email || ''} disabled className="input-field bg-surface-50 dark:bg-surface-800" />
              <p className="text-xs text-surface-400 mt-1">Email cannot be changed</p>
            </div>
          </div>

          <button onClick={handleProfileSave} disabled={saving} className="btn btn-primary btn-md mt-6">
            {saving ? <Loader className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />} Save Changes
          </button>
        </div>
      )}

      {tab === 'security' && (
        <div className="card p-6 max-w-xl">
          <h3 className="font-semibold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
            <Key className="h-5 w-5 text-landlord-500" /> Change Password
          </h3>
          <div className="space-y-4">
            <div>
              <label className="input-label">Current Password</label>
              <div className="relative">
                <input type={showPasswords ? 'text' : 'password'} value={currentPassword} onChange={e => setCurrentPassword(e.target.value)} className="input-field" />
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
            <label className="flex items-center gap-2 cursor-pointer text-sm text-surface-500">
              <input type="checkbox" checked={showPasswords} onChange={() => setShowPasswords(!showPasswords)} className="rounded" />
              Show passwords
            </label>
          </div>
          <button onClick={handlePasswordChange} disabled={saving} className="btn btn-primary btn-md mt-6">
            {saving ? <Loader className="h-4 w-4 animate-spin" /> : <Key className="h-4 w-4" />} Change Password
          </button>
        </div>
      )}
    </LandlordLayout>
  )
}

export default Settings
