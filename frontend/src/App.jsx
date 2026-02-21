import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { useEffect } from 'react'
import { useThemeStore } from './store/themeStore'

// Public Pages
import LandingPage from './pages/public/LandingPage'
import GetStarted from './pages/public/GetStarted'
import Login from './pages/Login'
import Register from './pages/Register'

// Route Guards
import RoleRoute from './components/guards/RoleRoute'
import PublicRoute from './components/guards/PublicRoute'

// Tenant Pages
import TenantDashboard from './pages/tenant/Dashboard'
import TenantAnalyze from './pages/tenant/Analyze'
import TenantHistory from './pages/tenant/History'
import TenantSubscription from './pages/tenant/Subscription'
import TenantListings from './pages/tenant/Listings'
import TenantSavedListings from './pages/tenant/SavedListings'
import TenantApplications from './pages/tenant/Applications'
import TenantPayments from './pages/tenant/Payments'
import TenantProfile from './pages/tenant/Profile'
import TenantImageVerification from './pages/tenant/ImageVerification'
import TenantAddressCheck from './pages/tenant/AddressCheck'
import TenantListingDetail from './pages/tenant/ListingDetail'

// Landlord Pages
import LandlordDashboard from './pages/landlord/DashboardNew'
import LandlordDocumentVerification from './pages/landlord/DocumentVerificationNew'
import LandlordTenantVerification from './pages/landlord/TenantVerificationNew'
import LandlordPropertyVerification from './pages/landlord/PropertyVerificationNew'
import LandlordVerificationHistory from './pages/landlord/VerificationHistoryNew'
import LandlordMyListings from './pages/landlord/MyListings'
import LandlordCreateListing from './pages/landlord/CreateListing'
import LandlordApplicants from './pages/landlord/Applicants'
import LandlordRiskAnalysis from './pages/landlord/RiskAnalysis'
import LandlordLeases from './pages/landlord/Leases'
import LandlordPayments from './pages/landlord/LandlordPayments'
import LandlordAnalytics from './pages/landlord/Analytics'
import LandlordSettings from './pages/landlord/Settings'

// Admin Pages
import AdminDashboard from './pages/admin/DashboardNew'
import AdminDatasets from './pages/admin/DatasetsNew'
import AdminUsers from './pages/admin/UsersNew'
import AdminAuditLogs from './pages/admin/AuditLogsNew'
import AdminPlans from './pages/admin/PlansNew'
import AdminAIEngines from './pages/admin/AIEnginesNew'
import AdminTrainedModels from './pages/admin/TrainedModelsNew'
import AdminAnalytics from './pages/admin/AnalyticsNew'
import AdminMonitoring from './pages/admin/MonitoringNew'
import AdminSettings from './pages/admin/SettingsNew'
import AdminFeedbackReview from './pages/admin/FeedbackReviewNew'
import AdminListingApproval from './pages/admin/ListingApprovalNew'

function App() {
  const initTheme = useThemeStore((s) => s.initTheme)

  useEffect(() => {
    initTheme()
  }, [initTheme])

  return (
    <BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          className: 'text-sm font-medium',
          style: {
            borderRadius: '12px',
            padding: '12px 16px',
          },
        }}
      />
      <Routes>
        {/* ============ PUBLIC ROUTES ============ */}
        <Route path="/" element={<PublicRoute><LandingPage /></PublicRoute>} />
        <Route path="/get-started" element={<PublicRoute><GetStarted /></PublicRoute>} />
        <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
        <Route path="/register" element={<PublicRoute><Register /></PublicRoute>} />

        {/* ============ TENANT ROUTES (renter role) ============ */}
        <Route path="/tenant" element={<RoleRoute allowedRoles={['renter', 'tenant']}><TenantDashboard /></RoleRoute>} />
        <Route path="/tenant/listings" element={<RoleRoute allowedRoles={['renter', 'tenant']}><TenantListings /></RoleRoute>} />
        <Route path="/tenant/listings/:id" element={<RoleRoute allowedRoles={['renter', 'tenant']}><TenantListingDetail /></RoleRoute>} />
        <Route path="/tenant/analyze" element={<RoleRoute allowedRoles={['renter', 'tenant']}><TenantAnalyze /></RoleRoute>} />
        <Route path="/tenant/verify-images" element={<RoleRoute allowedRoles={['renter', 'tenant']}><TenantImageVerification /></RoleRoute>} />
        <Route path="/tenant/verify-address" element={<RoleRoute allowedRoles={['renter', 'tenant']}><TenantAddressCheck /></RoleRoute>} />
        <Route path="/tenant/saved" element={<RoleRoute allowedRoles={['renter', 'tenant']}><TenantSavedListings /></RoleRoute>} />
        <Route path="/tenant/applications" element={<RoleRoute allowedRoles={['renter', 'tenant']}><TenantApplications /></RoleRoute>} />
        <Route path="/tenant/payments" element={<RoleRoute allowedRoles={['renter', 'tenant']}><TenantPayments /></RoleRoute>} />
        <Route path="/tenant/history" element={<RoleRoute allowedRoles={['renter', 'tenant']}><TenantHistory /></RoleRoute>} />
        <Route path="/tenant/subscription" element={<RoleRoute allowedRoles={['renter', 'tenant']}><TenantSubscription /></RoleRoute>} />
        <Route path="/tenant/profile" element={<RoleRoute allowedRoles={['renter', 'tenant']}><TenantProfile /></RoleRoute>} />

        {/* ============ LANDLORD ROUTES ============ */}
        <Route path="/landlord" element={<RoleRoute allowedRoles={['landlord']}><LandlordDashboard /></RoleRoute>} />
        <Route path="/landlord/listings" element={<RoleRoute allowedRoles={['landlord']}><LandlordMyListings /></RoleRoute>} />
        <Route path="/landlord/listings/new" element={<RoleRoute allowedRoles={['landlord']}><LandlordCreateListing /></RoleRoute>} />
        <Route path="/landlord/applicants" element={<RoleRoute allowedRoles={['landlord']}><LandlordApplicants /></RoleRoute>} />
        <Route path="/landlord/risk-analysis" element={<RoleRoute allowedRoles={['landlord']}><LandlordRiskAnalysis /></RoleRoute>} />
        <Route path="/landlord/documents" element={<RoleRoute allowedRoles={['landlord']}><LandlordDocumentVerification /></RoleRoute>} />
        <Route path="/landlord/tenants" element={<RoleRoute allowedRoles={['landlord']}><LandlordTenantVerification /></RoleRoute>} />
        <Route path="/landlord/property-images" element={<RoleRoute allowedRoles={['landlord']}><LandlordPropertyVerification /></RoleRoute>} />
        <Route path="/landlord/leases" element={<RoleRoute allowedRoles={['landlord']}><LandlordLeases /></RoleRoute>} />
        <Route path="/landlord/payments" element={<RoleRoute allowedRoles={['landlord']}><LandlordPayments /></RoleRoute>} />
        <Route path="/landlord/analytics" element={<RoleRoute allowedRoles={['landlord']}><LandlordAnalytics /></RoleRoute>} />
        <Route path="/landlord/history" element={<RoleRoute allowedRoles={['landlord']}><LandlordVerificationHistory /></RoleRoute>} />
        <Route path="/landlord/settings" element={<RoleRoute allowedRoles={['landlord']}><LandlordSettings /></RoleRoute>} />

        {/* ============ ADMIN ROUTES ============ */}
        <Route path="/admin" element={<RoleRoute allowedRoles={['admin']}><AdminDashboard /></RoleRoute>} />
        <Route path="/admin/analytics" element={<RoleRoute allowedRoles={['admin']}><AdminAnalytics /></RoleRoute>} />
        <Route path="/admin/monitoring" element={<RoleRoute allowedRoles={['admin']}><AdminMonitoring /></RoleRoute>} />
        <Route path="/admin/datasets" element={<RoleRoute allowedRoles={['admin']}><AdminDatasets /></RoleRoute>} />
        <Route path="/admin/users" element={<RoleRoute allowedRoles={['admin']}><AdminUsers /></RoleRoute>} />
        <Route path="/admin/audit-logs" element={<RoleRoute allowedRoles={['admin']}><AdminAuditLogs /></RoleRoute>} />
        <Route path="/admin/plans" element={<RoleRoute allowedRoles={['admin']}><AdminPlans /></RoleRoute>} />
        <Route path="/admin/ai-engines" element={<RoleRoute allowedRoles={['admin']}><AdminAIEngines /></RoleRoute>} />
        <Route path="/admin/trained-models" element={<RoleRoute allowedRoles={['admin']}><AdminTrainedModels /></RoleRoute>} />
        <Route path="/admin/settings" element={<RoleRoute allowedRoles={['admin']}><AdminSettings /></RoleRoute>} />
        <Route path="/admin/feedback-review" element={<RoleRoute allowedRoles={['admin']}><AdminFeedbackReview /></RoleRoute>} />
        <Route path="/admin/listing-approval" element={<RoleRoute allowedRoles={['admin']}><AdminListingApproval /></RoleRoute>} />

        {/* ============ LEGACY REDIRECTS ============ */}
        <Route path="/dashboard" element={<Navigate to="/tenant" replace />} />
        <Route path="/analyze" element={<Navigate to="/tenant/analyze" replace />} />
        <Route path="/history" element={<Navigate to="/tenant/history" replace />} />
        <Route path="/subscription" element={<Navigate to="/tenant/subscription" replace />} />

        {/* ============ CATCH-ALL ============ */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App

