import { Link } from 'react-router-dom'
import { useState, useEffect } from 'react'
import PublicNavbar from '../../components/public/PublicNavbar'
import Footer from '../../components/public/Footer'
import {
  Shield, Brain, Search, Lock, BarChart3, Users,
  ChevronRight, Star, Check, ArrowRight, Zap,
  FileCheck, AlertTriangle, Eye, Globe, Award,
  MessageSquare, Building2, UserCheck, TrendingUp,
  Loader2
} from 'lucide-react'
import api from '../../services/api'

const LandingPage = () => {
  const [plans, setPlans] = useState([])
  const [plansLoading, setPlansLoading] = useState(true)

  useEffect(() => {
    const fetchPlans = async () => {
      try {
        const { data } = await api.get('/renter/subscription/plans')
        setPlans(data)
      } catch (err) {
        console.error('Failed to load plans:', err)
      } finally {
        setPlansLoading(false)
      }
    }
    fetchPlans()
  }, [])

  // Build display-friendly feature list from plan features object
  const buildFeatureList = (plan) => {
    const list = []
    list.push(`${plan.scans_per_month.toLocaleString()} scans/month`)
    const featureLabels = {
      basic_analysis: 'Basic Risk Analysis',
      risk_score: 'Fraud Risk Score',
      detailed_indicators: 'Detailed Indicators',
      image_analysis: 'Image Forensics',
      advanced_ml: 'Advanced ML Models',
      message_analysis: 'Message Analysis',
      address_validation: 'Address Validation',
      price_anomaly: 'Price Anomaly Detection',
      xai_explanations: 'XAI Explanations',
      export_reports: 'Export Reports',
      api_access: 'API Access',
      custom_models: 'Custom Models',
    }
    if (plan.features) {
      Object.entries(featureLabels).forEach(([key, label]) => {
        if (plan.features[key]) list.push(label)
      })
      if (plan.features.support) {
        list.push(`${plan.features.support.charAt(0).toUpperCase() + plan.features.support.slice(1)} support`)
      }
    }
    return list
  }

  return (
    <div className="min-h-screen bg-white dark:bg-surface-950">
      <PublicNavbar />

      {/* ===== HERO ===== */}
      <section className="relative pt-24 lg:pt-32 pb-20 overflow-hidden">
        {/* Background effects */}
        <div className="absolute inset-0 bg-mesh" />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-gradient-radial from-primary-500/10 via-transparent to-transparent" />
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-4xl mx-auto">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-50 dark:bg-primary-950/50 border border-primary-200 dark:border-primary-800 mb-8 animate-fade-in">
              <Zap className="h-4 w-4 text-primary-600 dark:text-primary-400" />
              <span className="text-sm font-semibold text-primary-700 dark:text-primary-300">
                AI-Powered Fraud Detection
              </span>
              <ChevronRight className="h-4 w-4 text-primary-400" />
            </div>

            {/* Headline */}
            <h1 className="text-display-lg lg:text-display-2xl font-display font-bold text-surface-900 dark:text-white mb-6 animate-fade-in-up">
              Protect Every Rental
              <br />
              <span className="text-gradient-hero">Transaction with AI</span>
            </h1>

            <p className="text-lg lg:text-xl text-surface-500 dark:text-surface-400 mb-10 max-w-2xl mx-auto animate-fade-in-up stagger-1">
              Enterprise-grade fraud detection, trust scoring, and verification
              for landlords and tenants. Analyze listings, verify identities,
              and make confident rental decisions.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16 animate-fade-in-up stagger-2">
              <Link to="/get-started" className="btn btn-xl btn-primary shadow-glow-primary">
                Get Started Free
                <ArrowRight className="h-5 w-5" />
              </Link>
              <a href="#how-it-works" className="btn btn-xl btn-outline">
                See How It Works
              </a>
            </div>

            {/* Trust bar */}
            <div className="flex flex-wrap items-center justify-center gap-6 text-sm text-surface-500 dark:text-surface-400 animate-fade-in stagger-3">
              <div className="flex items-center gap-2">
                <Check className="h-4 w-4 text-accent-green" />
                <span>100% Free to Use</span>
              </div>
              <div className="flex items-center gap-2">
                <Check className="h-4 w-4 text-accent-green" />
                <span>4-Signal Fraud Detection</span>
              </div>
              <div className="flex items-center gap-2">
                <Check className="h-4 w-4 text-accent-green" />
                <span>AI-Powered Document Verification</span>
              </div>
              <div className="flex items-center gap-2">
                <Check className="h-4 w-4 text-accent-green" />
                <span>Toronto Market Data</span>
              </div>
            </div>
          </div>

          {/* Hero mockup */}
          <div className="mt-20 relative">
            <div className="bg-gradient-to-b from-surface-100 to-surface-50 dark:from-surface-800 dark:to-surface-900 rounded-3xl border border-surface-200 dark:border-surface-700 shadow-soft-2xl p-2 lg:p-3">
              <div className="bg-white dark:bg-surface-900 rounded-2xl overflow-hidden">
                {/* Mock header */}
                <div className="flex items-center gap-2 px-4 py-3 bg-surface-50 dark:bg-surface-800 border-b border-surface-200 dark:border-surface-700">
                  <div className="flex gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-red-400" />
                    <div className="w-3 h-3 rounded-full bg-amber-400" />
                    <div className="w-3 h-3 rounded-full bg-green-400" />
                  </div>
                  <div className="flex-1 mx-4">
                    <div className="bg-surface-100 dark:bg-surface-700 rounded-lg h-7 max-w-md mx-auto flex items-center px-3">
                      <Lock className="h-3 w-3 text-surface-400 mr-2" />
                      <span className="text-xs text-surface-400">app.rentalguard.ai</span>
                    </div>
                  </div>
                </div>
                {/* Dashboard mockup */}
                <div className="p-6 lg:p-8 min-h-[300px] lg:min-h-[400px]">
                  <div className="grid grid-cols-4 gap-4 mb-6">
                    {[
                      { label: 'Total Scans', value: '12,847', color: 'primary' },
                      { label: 'Fraud Detected', value: '234', color: 'red' },
                      { label: 'Trust Score Avg', value: '87.3', color: 'green' },
                      { label: 'Active Users', value: '3,891', color: 'blue' },
                    ].map((stat, i) => (
                      <div key={i} className="bg-surface-50 dark:bg-surface-800 rounded-xl p-4 animate-fade-in" style={{ animationDelay: `${0.5 + i * 0.1}s` }}>
                        <p className="text-xs text-surface-500 dark:text-surface-400">{stat.label}</p>
                        <p className="text-2xl font-bold text-surface-900 dark:text-white mt-1">{stat.value}</p>
                      </div>
                    ))}
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="col-span-2 bg-surface-50 dark:bg-surface-800 rounded-xl p-4 h-48">
                      <p className="text-sm font-semibold text-surface-700 dark:text-surface-300 mb-3">Risk Analysis Trend</p>
                      <div className="flex items-end gap-1 h-32">
                        {[40, 55, 35, 65, 45, 70, 50, 80, 60, 75, 55, 85].map((h, i) => (
                          <div key={i} className="flex-1 bg-primary-500/20 dark:bg-primary-500/30 rounded-t" style={{ height: `${h}%` }} />
                        ))}
                      </div>
                    </div>
                    <div className="bg-surface-50 dark:bg-surface-800 rounded-xl p-4 h-48">
                      <p className="text-sm font-semibold text-surface-700 dark:text-surface-300 mb-3">Risk Levels</p>
                      <div className="space-y-3 mt-4">
                        {[
                          { label: 'Very Low', pct: '45%', color: 'bg-green-500' },
                          { label: 'Medium', pct: '30%', color: 'bg-amber-500' },
                          { label: 'High', pct: '15%', color: 'bg-red-500' },
                          { label: 'Critical', pct: '10%', color: 'bg-red-700' },
                        ].map((item, i) => (
                          <div key={i} className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${item.color}`} />
                            <span className="text-xs text-surface-500 dark:text-surface-400 flex-1">{item.label}</span>
                            <span className="text-xs font-semibold text-surface-700 dark:text-surface-300">{item.pct}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            {/* Gradient fade at bottom */}
            <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-white dark:from-surface-950" />
          </div>
        </div>
      </section>

      {/* ===== ROLE SELECTION CTA ===== */}
      <section id="get-started" className="py-20 bg-surface-50 dark:bg-surface-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-display-md font-display font-bold text-surface-900 dark:text-white mb-4">
              Choose Your Path
            </h2>
            <p className="text-lg text-surface-500 dark:text-surface-400 max-w-xl mx-auto">
              RentalGuard is built for everyone in the rental ecosystem. Select your role to get started.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {/* Landlord Card */}
            <Link
              to="/register?role=landlord"
              className="group relative bg-white dark:bg-surface-800 rounded-3xl border-2 border-surface-200 dark:border-surface-700 p-8 lg:p-10
                       hover:border-landlord-500 dark:hover:border-landlord-500 hover:shadow-glow-blue
                       transition-all duration-300 cursor-pointer"
            >
              <div className="absolute top-6 right-6 opacity-0 group-hover:opacity-100 transition-opacity">
                <ArrowRight className="h-5 w-5 text-landlord-500" />
              </div>
              <div className="w-16 h-16 bg-landlord-100 dark:bg-landlord-950/50 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <Building2 className="h-8 w-8 text-landlord-600 dark:text-landlord-400" />
              </div>
              <h3 className="text-2xl font-display font-bold text-surface-900 dark:text-white mb-3">
                I'm a Landlord
              </h3>
              <p className="text-surface-500 dark:text-surface-400 mb-6">
                Screen tenants, verify documents, detect fraud, and manage your properties with AI-powered insights.
              </p>
              <ul className="space-y-2">
                {['AI tenant screening', 'Document verification', 'Fraud risk analysis', 'Property management'].map(f => (
                  <li key={f} className="flex items-center gap-2 text-sm text-surface-600 dark:text-surface-300">
                    <Check className="h-4 w-4 text-landlord-500" />
                    {f}
                  </li>
                ))}
              </ul>
              <div className="mt-8 btn btn-lg btn-landlord w-full">
                Get Started as Landlord
                <ArrowRight className="h-4 w-4" />
              </div>
            </Link>

            {/* Tenant Card */}
            <Link
              to="/register?role=renter"
              className="group relative bg-white dark:bg-surface-800 rounded-3xl border-2 border-surface-200 dark:border-surface-700 p-8 lg:p-10
                       hover:border-tenant-500 dark:hover:border-tenant-500 hover:shadow-glow-green
                       transition-all duration-300 cursor-pointer"
            >
              <div className="absolute top-6 right-6 opacity-0 group-hover:opacity-100 transition-opacity">
                <ArrowRight className="h-5 w-5 text-tenant-500" />
              </div>
              <div className="w-16 h-16 bg-tenant-100 dark:bg-tenant-950/50 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <UserCheck className="h-8 w-8 text-tenant-600 dark:text-tenant-400" />
              </div>
              <h3 className="text-2xl font-display font-bold text-surface-900 dark:text-white mb-3">
                I'm a Tenant
              </h3>
              <p className="text-surface-500 dark:text-surface-400 mb-6">
                Analyze listings for fraud, check landlord credibility, and protect yourself from rental scams.
              </p>
              <ul className="space-y-2">
                {['AI listing analysis', 'Scam detection', 'Trust score checking', 'Safe rental discovery'].map(f => (
                  <li key={f} className="flex items-center gap-2 text-sm text-surface-600 dark:text-surface-300">
                    <Check className="h-4 w-4 text-tenant-500" />
                    {f}
                  </li>
                ))}
              </ul>
              <div className="mt-8 btn btn-lg btn-tenant w-full">
                Get Started as Tenant
                <ArrowRight className="h-4 w-4" />
              </div>
            </Link>
          </div>

          {/* Admin access */}
          <div className="text-center mt-8">
            <Link to="/login?role=admin" className="text-sm text-surface-400 hover:text-surface-600 dark:hover:text-surface-300 transition-colors">
              Admin Access →
            </Link>
          </div>
        </div>
      </section>

      {/* ===== FEATURES ===== */}
      <section id="features" className="py-20 bg-white dark:bg-surface-950">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <div className="badge-primary mb-4 inline-flex">Platform Features</div>
            <h2 className="text-display-md font-display font-bold text-surface-900 dark:text-white mb-4">
              Everything You Need for Safe Rentals
            </h2>
            <p className="text-lg text-surface-500 dark:text-surface-400 max-w-2xl mx-auto">
              Comprehensive tools for fraud detection, document verification, and property management — all powered by AI.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              { icon: Brain, title: 'AI Fraud Detection', desc: 'A 4-signal fusion pipeline combining DistilBERT text classification, Isolation Forest anomaly detection, price benchmarking, and rule-based indicator matching.', color: 'text-primary-600 dark:text-primary-400 bg-primary-100 dark:bg-primary-950/50' },
              { icon: FileCheck, title: 'Document Verification', desc: 'OCR engine extracts text from uploaded IDs and income proofs. Cross-document engine automatically checks consistency across multiple documents.', color: 'text-landlord-600 dark:text-landlord-400 bg-landlord-100 dark:bg-landlord-950/50' },
              { icon: Shield, title: 'Trust Score', desc: 'Dynamic trust score calculated from profile completeness, document verification status, and account history — displayed in real time on the tenant dashboard.', color: 'text-tenant-600 dark:text-tenant-400 bg-tenant-100 dark:bg-tenant-950/50' },
              { icon: Eye, title: 'Image Forensics', desc: 'Detects duplicate and stolen listing photos using perceptual hashing (pHash) and validates image metadata via EXIF analysis.', color: 'text-accent-purple bg-purple-100 dark:bg-purple-950/50' },
              { icon: MessageSquare, title: 'Message Risk Analysis', desc: 'Scans landlord–tenant messages for high-pressure tactics, urgency language, and known scam phrases using NLP pattern detection.', color: 'text-accent-amber bg-amber-100 dark:bg-amber-950/50' },
              { icon: BarChart3, title: 'Price Benchmarking', desc: 'Compares listing rent against Toronto neighbourhood-level benchmarks to flag statistically anomalous pricing in real time.', color: 'text-accent-cyan bg-cyan-100 dark:bg-cyan-950/50' },
            ].map((feature, i) => (
              <div
                key={i}
                className="card-hover p-6 lg:p-8 group"
              >
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 ${feature.color} group-hover:scale-110 transition-transform`}>
                  <feature.icon className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-display font-bold text-surface-900 dark:text-white mb-2">{feature.title}</h3>
                <p className="text-sm text-surface-500 dark:text-surface-400 leading-relaxed">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== HOW IT WORKS ===== */}
      <section id="how-it-works" className="py-20 bg-surface-50 dark:bg-surface-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <div className="badge-primary mb-4 inline-flex">How It Works</div>
            <h2 className="text-display-md font-display font-bold text-surface-900 dark:text-white mb-4">
              Simple. Powerful. Secure.
            </h2>
          </div>

          <div className="grid lg:grid-cols-2 gap-16 items-start">
            {/* For Tenants */}
            <div>
              <h3 className="text-xl font-display font-bold text-tenant-600 dark:text-tenant-400 mb-8 flex items-center gap-2">
                <UserCheck className="h-6 w-6" />
                For Tenants
              </h3>
              <div className="space-y-8">
                {[
                  { step: '01', title: 'Browse Listings', desc: 'Explore available rental properties and view AI-generated fraud risk scores for each listing.' },
                  { step: '02', title: 'Review Risk Analysis', desc: 'See detailed fraud indicators from our 4-signal pipeline — text analysis, price anomaly, address validation, and pattern matching.' },
                  { step: '03', title: 'Apply & Upload Documents', desc: 'Submit your application and upload ID and income documents. Our OCR engine verifies them automatically.' },
                  { step: '04', title: 'Message & Sign Lease', desc: 'Chat with the landlord directly, get approved, and sign your lease — all within the platform.' },
                ].map((item, i) => (
                  <div key={i} className="flex gap-4">
                    <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-tenant-100 dark:bg-tenant-950/50 flex items-center justify-center">
                      <span className="text-sm font-bold text-tenant-600 dark:text-tenant-400">{item.step}</span>
                    </div>
                    <div>
                      <h4 className="font-semibold text-surface-900 dark:text-white mb-1">{item.title}</h4>
                      <p className="text-sm text-surface-500 dark:text-surface-400">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* For Landlords */}
            <div>
              <h3 className="text-xl font-display font-bold text-landlord-600 dark:text-landlord-400 mb-8 flex items-center gap-2">
                <Building2 className="h-6 w-6" />
                For Landlords
              </h3>
              <div className="space-y-8">
                {[
                  { step: '01', title: 'Create Your Listing', desc: 'Post a property listing with details, images, and pricing. An admin reviews and approves it.' },
                  { step: '02', title: 'Review Applicants', desc: 'View tenant applications, chat with applicants, and review their uploaded documents side by side.' },
                  { step: '03', title: 'Verify Documents', desc: 'Our OCR engine extracts data from IDs and income proofs, and cross-document checks flag any inconsistencies.' },
                  { step: '04', title: 'Approve & Manage Leases', desc: 'Approve an applicant to auto-generate a lease, auto-reject others, and deactivate the listing — all in one click.' },
                ].map((item, i) => (
                  <div key={i} className="flex gap-4">
                    <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-landlord-100 dark:bg-landlord-950/50 flex items-center justify-center">
                      <span className="text-sm font-bold text-landlord-600 dark:text-landlord-400">{item.step}</span>
                    </div>
                    <div>
                      <h4 className="font-semibold text-surface-900 dark:text-white mb-1">{item.title}</h4>
                      <p className="text-sm text-surface-500 dark:text-surface-400">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ===== SECURITY / TRUST ===== */}
      <section id="security" className="py-20 bg-white dark:bg-surface-950">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <div className="badge-primary mb-4 inline-flex">Security & Trust</div>
              <h2 className="text-display-md font-display font-bold text-surface-900 dark:text-white mb-6">
                Built-In Security
              </h2>
              <p className="text-lg text-surface-500 dark:text-surface-400 mb-8">
                Your data is protected with industry-standard authentication, role-based access control, and isolated storage for every account.
              </p>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { icon: Lock, label: 'Bcrypt Password Hashing' },
                  { icon: Shield, label: 'JWT Token Authentication' },
                  { icon: Users, label: 'Role-Based Access Control' },
                  { icon: FileCheck, label: 'Isolated Document Storage' },
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 rounded-xl bg-surface-50 dark:bg-surface-800">
                    <item.icon className="h-5 w-5 text-primary-600 dark:text-primary-400 flex-shrink-0" />
                    <span className="text-sm font-medium text-surface-700 dark:text-surface-300">{item.label}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="relative">
              <div className="bg-gradient-to-br from-primary-100 to-primary-50 dark:from-primary-950/50 dark:to-surface-800 rounded-3xl p-8 lg:p-12">
                <div className="text-center space-y-6">
                  <div className="w-20 h-20 mx-auto bg-white dark:bg-surface-800 rounded-2xl shadow-soft-lg flex items-center justify-center">
                    <Shield className="h-10 w-10 text-primary-600 dark:text-primary-400" />
                  </div>
                  <div>
                    <p className="text-4xl font-display font-bold text-surface-900 dark:text-white">4-Signal</p>
                    <p className="text-sm text-surface-500 dark:text-surface-400 mt-1">Fraud Detection Pipeline</p>
                  </div>
                  <div className="grid grid-cols-2 gap-4 pt-4 border-t border-primary-200 dark:border-primary-900">
                    <div>
                      <p className="text-2xl font-bold text-surface-900 dark:text-white">9</p>
                      <p className="text-xs text-surface-500 dark:text-surface-400">AI Engines</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-surface-900 dark:text-white">3</p>
                      <p className="text-xs text-surface-500 dark:text-surface-400">User Roles</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ===== AI TECHNOLOGY ===== */}
      <section className="py-20 bg-surface-900 dark:bg-surface-950">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="badge bg-primary-500/20 text-primary-300 mb-4 inline-flex">AI Technology</div>
          <h2 className="text-display-md font-display font-bold text-white mb-4">
            Powered by Advanced AI
          </h2>
          <p className="text-lg text-surface-400 mb-16 max-w-2xl mx-auto">
            Our 4-signal fusion pipeline combines NLP, statistical analysis, price benchmarking, and rule-based detection to produce a single fraud risk score.
          </p>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { icon: Brain, label: 'DistilBERT NLP', desc: 'Fine-tuned transformer model classifies listing text as fraudulent or legitimate' },
              { icon: TrendingUp, label: 'Isolation Forest', desc: 'Anomaly detection on tabular features like price, word count, and listing metadata' },
              { icon: BarChart3, label: 'Price Anomaly', desc: 'Compares rent against Toronto neighbourhood benchmarks to flag unrealistic pricing' },
              { icon: AlertTriangle, label: 'Indicator Engine', desc: 'Rule-based pattern matching for known scam phrases, urgency language, and red flags' },
              { icon: Eye, label: 'Image Forensics', desc: 'Detects duplicate images via perceptual hashing and checks EXIF metadata integrity' },
              { icon: FileCheck, label: 'OCR & Document AI', desc: 'Extracts text from uploaded IDs and income docs, cross-checks consistency across documents' },
              { icon: Globe, label: 'Address Validation', desc: 'Geocodes addresses via Nominatim and validates against known scam location patterns' },
              { icon: Shield, label: '4-Signal Fusion', desc: 'Combines all engine outputs into a weighted fraud risk score with XAI explanations' },
            ].map((item, i) => (
              <div key={i} className="bg-surface-800 dark:bg-surface-900 rounded-2xl p-6 border border-surface-700 dark:border-surface-800 hover:border-primary-500/50 transition-colors">
                <div className="w-12 h-12 mx-auto mb-4 bg-primary-500/20 rounded-xl flex items-center justify-center">
                  <item.icon className="h-6 w-6 text-primary-400" />
                </div>
                <h3 className="font-display font-bold text-white mb-2">{item.label}</h3>
                <p className="text-sm text-surface-400">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== PRICING ===== */}
      <section id="pricing" className="py-20 bg-white dark:bg-surface-950">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <div className="badge-primary mb-4 inline-flex">Pricing</div>
            <h2 className="text-display-md font-display font-bold text-surface-900 dark:text-white mb-4">
              Plans for Every Need
            </h2>
            <p className="text-lg text-surface-500 dark:text-surface-400 max-w-xl mx-auto">
              Start free and scale as you grow. No hidden fees.
            </p>
          </div>

          {plansLoading ? (
            <div className="flex justify-center py-16">
              <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
            </div>
          ) : plans.length === 0 ? (
            <p className="text-center text-surface-500">No plans available at the moment.</p>
          ) : (
            <div className={`grid gap-8 max-w-6xl mx-auto ${
              plans.length === 1 ? 'max-w-md' :
              plans.length === 2 ? 'md:grid-cols-2 max-w-3xl' :
              plans.length === 3 ? 'md:grid-cols-3 max-w-5xl' :
              'md:grid-cols-2 lg:grid-cols-4'
            }`}>
              {plans.map((plan, i) => {
                const isPopular = plan.name === 'premium'
                const features = buildFeatureList(plan)
                return (
                  <div
                    key={plan.id}
                    className={`relative rounded-3xl p-8 ${
                      isPopular
                        ? 'bg-surface-900 dark:bg-white text-white dark:text-surface-900 border-2 border-primary-500 shadow-glow-primary scale-105'
                        : 'bg-white dark:bg-surface-800 border-2 border-surface-200 dark:border-surface-700'
                    }`}
                  >
                    {isPopular && (
                      <div className="absolute -top-3 left-1/2 -translate-x-1/2 badge bg-primary-600 text-white px-4 py-1">
                        Most Popular
                      </div>
                    )}
                    <h3 className={`text-xl font-display font-bold ${isPopular ? '' : 'text-surface-900 dark:text-white'}`}>
                      {plan.display_name}
                    </h3>
                    <div className="flex items-baseline gap-1 my-4">
                      <span className={`text-4xl font-display font-bold ${isPopular ? '' : 'text-surface-900 dark:text-white'}`}>
                        {plan.price === 0 ? '$0' : `$${plan.price.toFixed(2)}`}
                      </span>
                      <span className={`text-sm ${isPopular ? 'text-surface-300 dark:text-surface-500' : 'text-surface-500 dark:text-surface-400'}`}>
                        {plan.price === 0 ? 'forever' : '/month'}
                      </span>
                    </div>
                    <ul className="space-y-3 mb-8">
                      {features.map((f, j) => (
                        <li key={j} className="flex items-center gap-2 text-sm">
                          <Check className={`h-4 w-4 flex-shrink-0 ${isPopular ? 'text-primary-400 dark:text-primary-600' : 'text-accent-green'}`} />
                          <span>{f}</span>
                        </li>
                      ))}
                    </ul>
                    <Link
                      to="/get-started"
                      className={`btn btn-lg w-full ${
                        isPopular
                          ? 'bg-primary-600 dark:bg-primary-600 text-white hover:bg-primary-700'
                          : 'btn-outline'
                      }`}
                    >
                      Get Started
                    </Link>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </section>

      {/* ===== TESTIMONIALS ===== */}
      <section className="py-20 bg-surface-50 dark:bg-surface-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-display-md font-display font-bold text-surface-900 dark:text-white mb-4">
              What Our Users Say
            </h2>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                quote: "The fraud analysis flagged a listing that had stolen photos and a suspicious price. It helped me avoid a rental scam.",
                name: "Tenant User",
                role: "Renter, Toronto",
                rating: 5,
              },
              {
                quote: "I can review applicant documents and the AI cross-checks them automatically. It saves me time verifying each tenant manually.",
                name: "Landlord User",
                role: "Property Owner",
                rating: 5,
              },
              {
                quote: "The admin dashboard gives a clear view of flagged listings, user activity, and model performance all in one place.",
                name: "Admin User",
                role: "System Administrator",
                rating: 5,
              },
            ].map((testimonial, i) => (
              <div key={i} className="card p-8">
                <div className="flex gap-1 mb-4">
                  {Array.from({ length: testimonial.rating }).map((_, j) => (
                    <Star key={j} className="h-4 w-4 text-amber-400 fill-amber-400" />
                  ))}
                </div>
                <p className="text-surface-600 dark:text-surface-300 mb-6 italic">"{testimonial.quote}"</p>
                <div>
                  <p className="font-semibold text-surface-900 dark:text-white">{testimonial.name}</p>
                  <p className="text-sm text-surface-500 dark:text-surface-400">{testimonial.role}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== FAQ ===== */}
      <section id="faq" className="py-20 bg-white dark:bg-surface-950">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-display-md font-display font-bold text-surface-900 dark:text-white mb-4">
              Frequently Asked Questions
            </h2>
          </div>
          <div className="space-y-4">
            {[
              { q: 'How does the AI fraud detection work?', a: 'We use a 4-signal fusion pipeline: a DistilBERT NLP classifier analyzes listing text, an Isolation Forest model detects statistical anomalies, a price engine compares against Toronto market benchmarks, and a rule-based indicator engine catches known scam patterns. All signals combine into a single fraud risk score.' },
              { q: 'Is my personal data safe?', a: 'Yes. All passwords are hashed with bcrypt, sessions are managed via signed JWT tokens, and role-based access control ensures users only see their own data. Uploaded documents are stored server-side and never shared across accounts.' },
              { q: 'Is RentalGuard free to use?', a: 'Yes, RentalGuard is completely free. Register as a Tenant to analyze listings and verify documents, or as a Landlord to manage properties and screen applicants. No subscription or payment required.' },
              { q: 'How does tenant screening work?', a: 'Landlords can review applicant profiles and uploaded documents. Our OCR engine extracts data from ID and income documents, and the cross-document engine checks consistency across multiple uploads to flag mismatches automatically.' },
              { q: 'What markets do you support?', a: 'Our price benchmarking data currently covers the Greater Toronto Area with neighbourhood-level rent comparisons. The fraud detection AI (text analysis, image forensics, and document verification) works with any Canadian rental listing.' },
            ].map((faq, i) => (
              <details key={i} className="group card p-6 cursor-pointer">
                <summary className="flex items-center justify-between font-semibold text-surface-900 dark:text-white list-none">
                  {faq.q}
                  <ChevronRight className="h-5 w-5 text-surface-400 group-open:rotate-90 transition-transform" />
                </summary>
                <p className="mt-4 text-sm text-surface-500 dark:text-surface-400 leading-relaxed">{faq.a}</p>
              </details>
            ))}
          </div>
        </div>
      </section>

      {/* ===== FINAL CTA ===== */}
      <section className="py-20 bg-gradient-to-br from-primary-600 via-primary-700 to-primary-800">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-display-md font-display font-bold text-white mb-6">
            Ready to Protect Your Rental Journey?
          </h2>
          <p className="text-lg text-primary-100 mb-10 max-w-2xl mx-auto">
            Join thousands of landlords and tenants who trust RentalGuard for safe, fraud-free rental experiences.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link to="/get-started" className="btn btn-xl bg-white text-primary-700 hover:bg-primary-50 shadow-soft-xl">
              Get Started Free
              <ArrowRight className="h-5 w-5" />
            </Link>
            <Link to="/contact" className="btn btn-xl border-2 border-white/30 text-white hover:bg-white/10">
              Talk to Sales
            </Link>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  )
}

export default LandingPage
