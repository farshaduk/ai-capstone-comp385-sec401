import { Link } from 'react-router-dom'
import PublicNavbar from '../../components/public/PublicNavbar'
import Footer from '../../components/public/Footer'
import { Building2, UserCheck, ArrowRight, Check, Shield } from 'lucide-react'

const GetStarted = () => {
  return (
    <div className="min-h-screen bg-white dark:bg-surface-950">
      <PublicNavbar />

      <section className="pt-24 lg:pt-32 pb-20">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-50 dark:bg-primary-950/50 border border-primary-200 dark:border-primary-800 mb-6">
              <Shield className="h-4 w-4 text-primary-600 dark:text-primary-400" />
              <span className="text-sm font-semibold text-primary-700 dark:text-primary-300">Choose Your Role</span>
            </div>
            <h1 className="text-display-lg font-display font-bold text-surface-900 dark:text-white mb-4">
              How Will You Use RentalGuard?
            </h1>
            <p className="text-lg text-surface-500 dark:text-surface-400 max-w-xl mx-auto">
              Select your role to get a tailored experience. Each role has its own dedicated dashboard and tools.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {/* Landlord */}
            <Link
              to="/register?role=landlord"
              className="group relative bg-white dark:bg-surface-800 rounded-3xl border-2 border-surface-200 dark:border-surface-700 p-8 lg:p-10
                       hover:border-landlord-500 dark:hover:border-landlord-500 hover:shadow-glow-blue
                       transition-all duration-300"
            >
              <div className="absolute top-6 right-6 opacity-0 group-hover:opacity-100 transition-opacity">
                <ArrowRight className="h-5 w-5 text-landlord-500" />
              </div>
              <div className="w-20 h-20 bg-gradient-to-br from-landlord-100 to-landlord-200 dark:from-landlord-950/50 dark:to-landlord-900/30 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <Building2 className="h-10 w-10 text-landlord-600 dark:text-landlord-400" />
              </div>
              <h2 className="text-2xl font-display font-bold text-surface-900 dark:text-white mb-3">
                I'm a Landlord
              </h2>
              <p className="text-surface-500 dark:text-surface-400 mb-8">
                Manage properties, screen tenants with AI, verify documents, and protect your investments.
              </p>
              <ul className="space-y-3 mb-8">
                {[
                  'Create & manage property listings',
                  'AI-powered tenant screening',
                  'Document & identity verification',
                  'Fraud risk analysis per applicant',
                  'Lease & payment management',
                  'Analytics dashboard',
                ].map(f => (
                  <li key={f} className="flex items-start gap-2 text-sm text-surface-600 dark:text-surface-300">
                    <Check className="h-4 w-4 text-landlord-500 mt-0.5 flex-shrink-0" />
                    {f}
                  </li>
                ))}
              </ul>
              <div className="btn btn-lg btn-landlord w-full">
                Continue as Landlord
                <ArrowRight className="h-4 w-4" />
              </div>
            </Link>

            {/* Tenant */}
            <Link
              to="/register?role=renter"
              className="group relative bg-white dark:bg-surface-800 rounded-3xl border-2 border-surface-200 dark:border-surface-700 p-8 lg:p-10
                       hover:border-tenant-500 dark:hover:border-tenant-500 hover:shadow-glow-green
                       transition-all duration-300"
            >
              <div className="absolute top-6 right-6 opacity-0 group-hover:opacity-100 transition-opacity">
                <ArrowRight className="h-5 w-5 text-tenant-500" />
              </div>
              <div className="w-20 h-20 bg-gradient-to-br from-tenant-100 to-tenant-200 dark:from-tenant-950/50 dark:to-tenant-900/30 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <UserCheck className="h-10 w-10 text-tenant-600 dark:text-tenant-400" />
              </div>
              <h2 className="text-2xl font-display font-bold text-surface-900 dark:text-white mb-3">
                I'm a Tenant
              </h2>
              <p className="text-surface-500 dark:text-surface-400 mb-8">
                Analyze listings for fraud, check landlord credibility, and rent with confidence.
              </p>
              <ul className="space-y-3 mb-8">
                {[
                  'AI-powered listing analysis',
                  'Scam & fraud detection',
                  'Trust score visibility',
                  'Save & compare listings',
                  'Payment tracking',
                  'Direct landlord messaging',
                ].map(f => (
                  <li key={f} className="flex items-start gap-2 text-sm text-surface-600 dark:text-surface-300">
                    <Check className="h-4 w-4 text-tenant-500 mt-0.5 flex-shrink-0" />
                    {f}
                  </li>
                ))}
              </ul>
              <div className="btn btn-lg btn-tenant w-full">
                Continue as Tenant
                <ArrowRight className="h-4 w-4" />
              </div>
            </Link>
          </div>

          {/* Admin link */}
          <div className="text-center mt-10">
            <Link
              to="/login?role=admin"
              className="text-sm text-surface-400 hover:text-surface-600 dark:hover:text-surface-300 transition-colors inline-flex items-center gap-1"
            >
              Admin Access <ArrowRight className="h-3 w-3" />
            </Link>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  )
}

export default GetStarted
