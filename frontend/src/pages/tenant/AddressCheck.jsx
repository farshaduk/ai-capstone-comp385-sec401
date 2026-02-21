import { useState } from 'react'
import TenantLayout from '../../components/layouts/TenantLayout'
import { renterAPI } from '../../services/api'
import {
  MapPin, Search, Loader, CheckCircle, AlertTriangle, XCircle, Shield,
  Navigation, Home
} from 'lucide-react'
import toast from 'react-hot-toast'

const statusIcon = { pass: CheckCircle, warn: AlertTriangle, fail: XCircle }
const statusColor = {
  pass: 'text-green-500',
  warn: 'text-yellow-500',
  fail: 'text-red-500',
}
const riskBadge = {
  low: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  medium: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  high: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
}

const AddressCheck = () => {
  const [address, setAddress] = useState('')
  const [city, setCity] = useState('Toronto')
  const [postalCode, setPostalCode] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const handleVerify = async (e) => {
    e?.preventDefault()
    if (!address.trim()) return toast.error('Please enter a street address')
    setLoading(true)
    setResult(null)

    try {
      const { data } = await renterAPI.verifyAddress({
        address: address.trim(),
        city: city.trim(),
        postal_code: postalCode.trim(),
      })
      setResult(data)
      toast.success('Address verified')
    } catch (err) {
      console.error(err)
      toast.error('Verification failed — try again')
    } finally {
      setLoading(false)
    }
  }

  return (
    <TenantLayout title="Address Verification" subtitle="Verify if a rental address is real before committing">
      <div className="max-w-3xl mx-auto">
        {/* Form */}
        <form onSubmit={handleVerify} className="card p-6 mb-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-tenant-100 dark:bg-tenant-900/30 rounded-xl">
              <Navigation className="h-6 w-6 text-tenant-600 dark:text-tenant-400" />
            </div>
            <div>
              <h2 className="font-semibold text-surface-900 dark:text-white">Verify Rental Address</h2>
              <p className="text-sm text-surface-500">Check postal code validity, geocoding, and scam database</p>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-4 mb-4">
            <div className="md:col-span-3">
              <label className="input-label">Street Address</label>
              <div className="relative">
                <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-surface-400" />
                <input type="text" value={address} onChange={e => setAddress(e.target.value)}
                  placeholder="e.g. 123 King St W, Unit 4" className="input-field pl-10" required />
              </div>
            </div>
            <div>
              <label className="input-label">City</label>
              <input type="text" value={city} onChange={e => setCity(e.target.value)} className="input-field" />
            </div>
            <div>
              <label className="input-label">Province</label>
              <input type="text" value="Ontario" disabled className="input-field bg-surface-50 dark:bg-surface-800" />
            </div>
            <div>
              <label className="input-label">Postal Code</label>
              <input type="text" value={postalCode} onChange={e => setPostalCode(e.target.value)}
                placeholder="M5V 1A1" className="input-field" maxLength={7} />
            </div>
          </div>

          <button type="submit" disabled={loading} className="btn btn-primary btn-lg w-full">
            {loading ? <><Loader className="h-5 w-5 animate-spin" /> Verifying...</> : <><Search className="h-5 w-5" /> Verify Address</>}
          </button>
        </form>

        {/* Results */}
        {result && (
          <div className="space-y-6 animate-in fade-in">
            {/* Overall */}
            <div className="card p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Home className="h-5 w-5 text-surface-400" />
                  <div>
                    <p className="text-xs text-surface-500 uppercase tracking-wider">Full Address</p>
                    <p className="font-semibold text-surface-900 dark:text-white">{result.full_address}</p>
                  </div>
                </div>
                <span className={`px-4 py-1.5 rounded-full text-sm font-semibold ${riskBadge[result.risk_level] || riskBadge.medium}`}>
                  {result.risk_level.toUpperCase()} RISK
                </span>
              </div>

              <div className="grid grid-cols-3 gap-4 mt-4">
                <div className="text-center py-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                  <p className="text-xs text-surface-500">Valid</p>
                  <p className={`text-lg font-bold ${result.is_valid ? 'text-green-600' : 'text-red-500'}`}>
                    {result.is_valid ? 'Yes' : 'No'}
                  </p>
                </div>
                <div className="text-center py-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                  <p className="text-xs text-surface-500">Geocoded</p>
                  <p className={`text-lg font-bold ${result.geocoded ? 'text-green-600' : 'text-red-500'}`}>
                    {result.geocoded ? 'Yes' : 'No'}
                  </p>
                </div>
                <div className="text-center py-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                  <p className="text-xs text-surface-500">Coordinates</p>
                  <p className="text-sm font-medium text-surface-700 dark:text-surface-300">
                    {result.latitude ? `${result.latitude.toFixed(4)}, ${result.longitude.toFixed(4)}` : '—'}
                  </p>
                </div>
              </div>

              {/* Extra engine insights */}
              <div className="grid grid-cols-3 gap-4 mt-3">
                <div className="text-center py-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                  <p className="text-xs text-surface-500">Confidence</p>
                  <p className="text-lg font-bold text-tenant-600">
                    {result.confidence != null ? `${Math.round(result.confidence * 100)}%` : '—'}
                  </p>
                </div>
                <div className="text-center py-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                  <p className="text-xs text-surface-500">Residential</p>
                  <p className={`text-lg font-bold ${result.is_residential ? 'text-green-600' : 'text-amber-500'}`}>
                    {result.is_residential ? 'Yes' : 'Unknown'}
                  </p>
                </div>
                <div className="text-center py-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                  <p className="text-xs text-surface-500">Engine Status</p>
                  <p className="text-sm font-semibold text-surface-700 dark:text-surface-300 capitalize">
                    {result.engine_status || '—'}
                  </p>
                </div>
              </div>

              {result.explanation && (
                <div className="mt-3 p-3 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                  <p className="text-xs font-medium text-blue-700 dark:text-blue-300 uppercase tracking-wider mb-1">Engine Analysis</p>
                  <p className="text-sm text-blue-800 dark:text-blue-200">{result.explanation}</p>
                </div>
              )}
            </div>

            {/* Individual checks */}
            <div className="card p-6">
              <h3 className="font-semibold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
                <Shield className="h-5 w-5 text-tenant-500" /> Verification Checks
              </h3>
              <div className="space-y-4">
                {(result.checks || []).map((check, i) => {
                  const Icon = statusIcon[check.status] || AlertTriangle
                  return (
                    <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                      <Icon className={`h-5 w-5 mt-0.5 ${statusColor[check.status]}`} />
                      <div>
                        <p className="font-medium text-sm text-surface-900 dark:text-white">{check.name}</p>
                        <p className="text-xs text-surface-500 mt-0.5">{check.detail}</p>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </TenantLayout>
  )
}

export default AddressCheck
